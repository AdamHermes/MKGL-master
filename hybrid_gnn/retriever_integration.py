"""
HybridRetriever Integration for MKGL
====================================

This module provides integration between the HybridRetriever and the 
existing MKGL retriever interface.

Usage in main.py:
    from hybrid_gnn.retriever_integration import HybridScoreRetriever
    
    # In config, set use_hybrid: yes under score_retriever
    if cfg.score_retriever.get('use_hybrid', False):
        model.score_retriever = HybridScoreRetriever(cfg.score_retriever, ...)

Author: MKGL Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .sampler import HybridSampler
from .layers import PEARL_GIN, HybridBlock
from .models import Scorer


class HybridScoreRetriever(nn.Module):
    """
    Score retriever using Hybrid GNN architecture.
    
    Drop-in replacement for ScoreRetriever that uses:
    - Local PNA on real KG edges
    - Global Sparse GT on augmented edges
    - PEARL positional encoding
    
    Attributes:
        config: Configuration dict/namespace
        hidden_dim: GNN hidden dimension (typically 32)
        llm_hidden_dim: LLM embedding dimension (e.g., 2048 for TinyLlama)
    """
    
    def __init__(
        self,
        config,
        text_embeddings: torch.Tensor,
        kgl2token: torch.Tensor,
        orig_vocab_size: int,
    ):
        """
        Initialize HybridScoreRetriever.
        
        Args:
            config: Configuration with score_retriever settings
            text_embeddings: LLM embedding weights [vocab_size, llm_hidden_dim]
            kgl2token: KGL token to text token mapping
            orig_vocab_size: Original vocabulary size
        """
        super(HybridScoreRetriever, self).__init__()
        
        self.config = config
        self.text_embeddings = text_embeddings
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        # Dimensions
        self.llm_hidden_dim = config.llm_hidden_dim
        self.hidden_dim = config.r  # Use r as hidden_dim for consistency
        
        # Get hybrid config
        hybrid_cfg = config.get('hybrid', {})
        self.num_layers = hybrid_cfg.get('num_layers', 6)
        self.num_heads = hybrid_cfg.get('num_heads', 4)
        self.num_edge_types = hybrid_cfg.get('num_edge_types', 3)
        self.k_semantic = hybrid_cfg.get('k_semantic', 10)
        self.k_random = hybrid_cfg.get('k_random', 5)
        self.use_pearl = hybrid_cfg.get('use_pearl', True)
        self.dropout = hybrid_cfg.get('dropout', 0.1)
        
        # Text encoder (same as BasePNARetriever)
        self.down_scaling = nn.Linear(
            self.llm_hidden_dim, self.hidden_dim, 
            bias=False, dtype=torch.float
        )
        
        if config.text_encoder == 'pna':
            self.re_scaling = nn.Linear(self.hidden_dim * 12, self.hidden_dim)
        
        # LLM embedding down-scaling
        self.h_down_scaling = nn.Linear(
            self.llm_hidden_dim, self.hidden_dim,
            bias=False, dtype=torch.float
        )
        self.r_down_scaling = nn.Linear(
            self.llm_hidden_dim, self.hidden_dim,
            bias=False, dtype=torch.float
        )
        
        # Relation embedding (learnable)
        num_relations = config.kg_encoder.get('num_relation', 237)
        self.rel_embedding = nn.Embedding(num_relations * 2, self.hidden_dim)
        
        # PEARL positional encoding
        if self.use_pearl:
            self.pearl = PEARL_GIN(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                dropout=self.dropout,
            )
        
        # Hybrid blocks
        self.hybrid_layers = nn.ModuleList([
            HybridBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_edge_types=self.num_edge_types,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])
        
        # Scorer
        self.scorer = Scorer(
            hidden_dim=self.hidden_dim,
            num_mlp_layers=2,
            dropout=self.dropout,
        )
        
        # Sampler (initialized lazily with text embeddings)
        self.sampler = HybridSampler(
            k_semantic=self.k_semantic,
            k_random=self.k_random,
            use_faiss=True,
        )
        self._sampler_initialized = False
        
        # Cache for augmented graph
        self._cached_aug_edges = None
        self._cached_graph_id = None
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.down_scaling.weight)
        nn.init.xavier_uniform_(self.h_down_scaling.weight)
        nn.init.xavier_uniform_(self.r_down_scaling.weight)
        nn.init.normal_(self.rel_embedding.weight, std=0.02)
    
    def aggregate_text(
        self,
        token_ids: torch.Tensor,
        text_embeddings: torch.Tensor,
        method: str = 'pna',
    ) -> torch.Tensor:
        """Aggregate text token embeddings (same as BasePNARetriever)."""
        device = text_embeddings.device
        
        token_ids = token_ids.to(device)
        token_mask = (token_ids > 0).unsqueeze(-1).to(device)
        token_lengths = token_mask.float().sum(axis=1).to(device)
        degree = token_lengths
        token_embs = text_embeddings[token_ids]
        
        mean = (token_embs * token_mask).sum(axis=1) / token_lengths.clamp(min=1)
        
        if method == 'mean':
            return mean
        
        # PNA aggregation
        sq_mean = (token_embs**2 * token_mask).sum(axis=1) / token_lengths.clamp(min=1)
        max_val, _ = (token_embs * token_mask).max(axis=1)
        min_val, _ = (token_embs * token_mask).min(axis=1)
        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
        
        features = torch.cat([mean, max_val, min_val, std], dim=-1)
        
        scale = degree.log().clamp(min=1e-6)
        scale = scale / scale.mean().clamp(min=1e-6)
        scales = torch.cat(
            [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], 
            dim=-1
        )
        
        result = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        return result
    
    def retrieve_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve and aggregate text embeddings."""
        R = self.down_scaling(self.text_embeddings)
        result = self.aggregate_text(token_ids, R, self.config.text_encoder)
        
        if self.config.text_encoder == 'pna':
            result = self.re_scaling(result)
        
        return F.normalize(result, p=2, dim=-1)
    
    def get_text_embeddings(self, all_kgl_index: torch.Tensor) -> torch.Tensor:
        """Get text embeddings for all entities."""
        token_ids = self.kgl2token[all_kgl_index.cpu()]
        return self.retrieve_text(token_ids)
    
    def _initialize_sampler(self, text_embs: torch.Tensor):
        """Initialize sampler with text embeddings."""
        if not self._sampler_initialized:
            self.sampler.build_semantic_index(text_embs)
            self._sampler_initialized = True
    
    def _get_augmented_edges(
        self,
        graph: Data,
        text_embs: torch.Tensor,
    ) -> tuple:
        """Get or compute augmented edges."""
        # First check if graph already has pre-computed augmented edges
        if hasattr(graph, 'aug_edge_index') and graph.aug_edge_index is not None:
            return graph.aug_edge_index, graph.aug_edge_type
        
        # Otherwise, compute on-the-fly (with caching)
        graph_id = (graph.num_nodes, graph.edge_index.shape[1])
        
        if self._cached_graph_id != graph_id or self._cached_aug_edges is None:
            self._initialize_sampler(text_embs)
            aug_edge_index, aug_edge_type = self.sampler.sample_edges(graph, text_embs)
            self._cached_aug_edges = (aug_edge_index, aug_edge_type)
            self._cached_graph_id = graph_id
        
        return self._cached_aug_edges
    
    def forward(
        self,
        h_id: torch.Tensor,
        r_id: torch.Tensor,
        t_id: torch.Tensor,
        hidden_states: torch.Tensor,
        rel_hidden_states: torch.Tensor,
        graph: Data,
        all_index: torch.Tensor,
        all_kgl_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for score prediction.
        
        Args:
            h_id: Head entity indices [batch_size] or [batch_size, num_neg+1]
            r_id: Relation indices [batch_size] or [batch_size, num_neg+1]
            t_id: Tail entity indices [batch_size] or [batch_size, num_neg+1]
            hidden_states: LLM hidden states for (h, r) [batch_size, llm_hidden_dim]
            rel_hidden_states: LLM hidden states for relations [batch_size, llm_hidden_dim]
            graph: PyG Data object with edge_index
            all_index: All entity indices
            all_kgl_index: All KGL token indices
        
        Returns:
            scores: Prediction scores [batch_size, num_candidates]
        """
        device = h_id.device
        batch_size = h_id.size(0)
        num_nodes = graph.num_nodes
        
        # Ensure proper shapes
        if h_id.dim() == 1:
            h_id = h_id.unsqueeze(-1)
        if r_id.dim() == 1:
            r_id = r_id.unsqueeze(-1)
        if t_id.dim() == 1:
            t_id = t_id.unsqueeze(-1)
        
        num_candidates = t_id.size(1)
        
        # ============================================================
        # 1. Get text embeddings for all entities
        # ============================================================
        text_embs = self.get_text_embeddings(all_kgl_index)  # [num_nodes, hidden_dim]
        
        # ============================================================
        # 2. Get augmented edges
        # ============================================================
        aug_edge_index, aug_edge_type = self._get_augmented_edges(graph, text_embs)
        aug_edge_index = aug_edge_index.to(device)
        aug_edge_type = aug_edge_type.to(device)
        
        # ============================================================
        # 3. Down-scale LLM embeddings
        # ============================================================
        head_embeds = self.h_down_scaling(hidden_states.float())  # [batch, hidden_dim]
        rel_embeds = self.r_down_scaling(rel_hidden_states.float())  # [batch, hidden_dim]
        
        # Get learnable relation embeddings
        rel_emb_learned = self.rel_embedding(r_id[:, 0])  # [batch, hidden_dim]
        
        # Combine embeddings
        query = rel_embeds + rel_emb_learned  # [batch, hidden_dim]
        
        # ============================================================
        # 4. PEARL Positional Encoding
        # ============================================================
        if self.use_pearl:
            noise = torch.randn(num_nodes, self.hidden_dim, device=device)
            h_pos = self.pearl(noise, graph.edge_index)
        else:
            h_pos = torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        # ============================================================
        # 5. Initialize node features
        # ============================================================
        x = text_embs.clone().to(device)
        
        # Inject head embeddings
        for i in range(batch_size):
            x[h_id[i, 0]] = head_embeds[i]
        
        # Add positional encoding
        x = x + h_pos
        
        # ============================================================
        # 6. Hybrid message passing
        # ============================================================
        batch_assignment = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        h = x
        for layer in self.hybrid_layers:
            h = layer(
                x=h,
                real_edge_index=graph.edge_index,
                aug_edge_index=aug_edge_index,
                aug_edge_type=aug_edge_type,
                query=query,
                batch=batch_assignment,
            )
        
        # ============================================================
        # 7. Final scoring
        # ============================================================
        # Gather tail embeddings: [batch_size, num_candidates, hidden_dim]
        tail_embeds = h[t_id]
        
        # Compute scores
        scores = self.scorer(tail_embeds, query, normalize=True)
        
        return scores


def create_hybrid_retriever(config, text_embeddings, kgl2token, orig_vocab_size):
    """
    Factory function to create HybridScoreRetriever.
    
    Args:
        config: Score retriever configuration
        text_embeddings: LLM embedding weights
        kgl2token: KGL to token mapping
        orig_vocab_size: Original vocabulary size
    
    Returns:
        HybridScoreRetriever instance
    """
    return HybridScoreRetriever(
        config=config,
        text_embeddings=text_embeddings,
        kgl2token=kgl2token,
        orig_vocab_size=orig_vocab_size,
    )
