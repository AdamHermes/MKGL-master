"""
HybridRetriever - Main Model for KG Reasoning
==============================================

Complete model integrating:
1. PEARL_GIN: Positional encodings
2. HybridBlock x N: Local PNA + Global GT processing
3. Scorer: Final ranking scores


Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional, Dict

from ..layers import PEARL_GIN, HybridBlock


class Scorer(nn.Module):
    """
    Scoring module for ranking tail predictions.
    
    Computes compatibility scores between node embeddings and relation queries.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(Scorer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim * 2
        
        self.linear = nn.Linear(self.feature_dim, hidden_dim)
        
        layers = []
        for i in range(num_mlp_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden: torch.Tensor,
        query: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute scores for node embeddings given a query.
        
        Args:
            hidden: Node embeddings [num_nodes, hidden_dim] or [batch, num_candidates, hidden_dim]
            query: Relation query [batch_size, hidden_dim] or [batch, num_candidates, hidden_dim]
            normalize: Whether to normalize embeddings
        
        Returns:
            scores: Compatibility scores
        """
        if normalize:
            hidden = F.normalize(hidden, p=2, dim=-1)
            query = F.normalize(query, p=2, dim=-1)
        
        # Handle different input shapes
        if hidden.dim() == query.dim():
            # Same dimensions - directly concatenate
            combined = torch.cat([hidden, query], dim=-1)
        elif hidden.dim() == 3 and query.dim() == 2:
            # hidden: [batch, num_candidates, dim], query: [batch, dim]
            query_expanded = query.unsqueeze(1).expand(-1, hidden.size(1), -1)
            combined = torch.cat([hidden, query_expanded], dim=-1)
        elif hidden.dim() == 2 and query.dim() == 3:
            # Unlikely but handle it
            hidden_expanded = hidden.unsqueeze(1).expand(-1, query.size(1), -1)
            combined = torch.cat([hidden_expanded, query], dim=-1)
        else:
            # Fallback: try direct concatenation
            combined = torch.cat([hidden, query], dim=-1)
        
        heuristic = self.linear(combined)
        heuristic = F.normalize(heuristic, p=2, dim=-1)
        
        # Final score - use hidden with same shape as heuristic
        if hidden.shape == heuristic.shape:
            score = self.mlp(hidden * heuristic).squeeze(-1)
        else:
            # If hidden was expanded, use the expanded version
            if hidden.dim() == 2 and heuristic.dim() == 3:
                hidden_for_score = hidden.unsqueeze(1).expand_as(heuristic)
            else:
                hidden_for_score = hidden
            score = self.mlp(hidden_for_score * heuristic).squeeze(-1)
        
        score = score * 10.0
        score = torch.clamp(score, min=-15, max=15)
        
        return score


class HybridRetriever(nn.Module):
    """
    Main model for Knowledge Graph reasoning with hybrid architecture.
    
    Attributes:
        hidden_dim (int): Hidden dimension for GNN processing
        llm_hidden_dim (int): LLM embedding dimension
        num_layers (int): Number of HybridBlock layers
        num_relations (int): Number of relation types
    
    Example:
        >>> model = HybridRetriever(hidden_dim=32, llm_hidden_dim=4096, num_layers=6)
        >>> scores = model(h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, text_embs)
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        llm_hidden_dim: int = 4096,
        num_layers: int = 6,
        num_relations: int = 237,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.1,
        use_pearl: bool = True,
        use_edge_selection: bool = True,
        node_ratio: float = 0.1,
        degree_ratio: float = 1.0,
    ):
        super(HybridRetriever, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.use_pearl = use_pearl
        self.use_edge_selection = use_edge_selection
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        
        # Down-scaling projections
        self.h_down_scaling = nn.Linear(llm_hidden_dim, hidden_dim, bias=False)
        self.r_down_scaling = nn.Linear(llm_hidden_dim, hidden_dim, bias=False)
        
        # Relation embedding
        self.rel_embedding = nn.Embedding(num_relations * 2, hidden_dim)
        
        # PEARL positional encoding
        if use_pearl:
            self.pearl = PEARL_GIN(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
            )
        
        # Hybrid blocks
        self.hybrid_layers = nn.ModuleList([
            HybridBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Scorer
        self.scorer = Scorer(
            hidden_dim=hidden_dim,
            num_mlp_layers=2,
            dropout=dropout,
        )
        
        # Up-scaling projection
        self.up_scaling = nn.Linear(hidden_dim, llm_hidden_dim, bias=False)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.h_down_scaling.weight)
        nn.init.xavier_uniform_(self.r_down_scaling.weight)
        nn.init.xavier_uniform_(self.up_scaling.weight)
        nn.init.normal_(self.rel_embedding.weight, std=0.02)
    
    def forward(
        self,
        h_index: torch.Tensor,
        r_index: torch.Tensor,
        t_index: torch.Tensor,
        hidden_states: torch.Tensor,
        rel_hidden_states: torch.Tensor,
        graph: Data,
        text_embeddings: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for link prediction.
        
        Args:
            h_index: Head entity indices [batch_size] or [batch_size, num_neg+1]
            r_index: Relation indices
            t_index: Tail entity indices
            hidden_states: LLM embeddings for head entities
            rel_hidden_states: LLM embeddings for relations
            graph: PyG Data object with aug_edge_index, aug_edge_type
            text_embeddings: Pre-computed text embeddings [num_entities, hidden_dim]
            return_embeddings: If True, also return final node embeddings
        
        Returns:
            scores: Prediction scores [batch_size, num_candidates]
        """
        device = h_index.device
        batch_size = h_index.size(0)
        num_nodes = graph.num_nodes
        
        # Handle different index shapes
        if h_index.dim() == 1:
            h_index = h_index.unsqueeze(-1)
        if r_index.dim() == 1:
            r_index = r_index.unsqueeze(-1)
        if t_index.dim() == 1:
            t_index = t_index.unsqueeze(-1)
        
        # Down-scale LLM embeddings
        head_embeds = self.h_down_scaling(hidden_states.float())
        rel_embeds = self.r_down_scaling(rel_hidden_states.float())
        rel_emb_learned = self.rel_embedding(r_index[:, 0])
        query = rel_embeds + rel_emb_learned
        
        # PEARL Positional Encoding
        if self.use_pearl:
            noise = torch.randn(num_nodes, self.hidden_dim, device=device)
            h_pos = self.pearl(noise, graph.edge_index)
        else:
            h_pos = torch.zeros(num_nodes, self.hidden_dim, device=device)
        
        # Initialize node features
        x = text_embeddings.clone()
        for i in range(batch_size):
            x[h_index[i, 0]] = head_embeds[i]
        x = x + h_pos
        
        # Get augmented edges
        real_edge_index = graph.edge_index
        if hasattr(graph, 'aug_edge_index') and graph.aug_edge_index is not None:
            aug_edge_index = graph.aug_edge_index
            aug_edge_type = graph.aug_edge_type
        else:
            aug_edge_index = real_edge_index
            aug_edge_type = torch.zeros(real_edge_index.size(1), dtype=torch.long, device=device)
        
        # Hybrid message passing
        batch_assignment = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        h = x
        for layer in self.hybrid_layers:
            h = layer(
                x=h,
                real_edge_index=real_edge_index,
                aug_edge_index=aug_edge_index,
                aug_edge_type=aug_edge_type,
                query=query,
                batch=batch_assignment,
            )
        
        # Final scoring
        num_candidates = t_index.size(1)
        tail_embeds = h[t_index]
        query_expanded = query.unsqueeze(1).expand(-1, num_candidates, -1)
        scores = self.scorer(tail_embeds, query_expanded, normalize=True)
        
        if return_embeddings:
            return scores, h
        
        return scores
    
    def get_embeddings_for_llm(
        self,
        node_embeddings: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project GNN embeddings back to LLM dimension."""
        selected = node_embeddings[node_indices]
        return self.up_scaling(selected)
    
    def __repr__(self) -> str:
        return (
            f"HybridRetriever(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  llm_hidden_dim={self.llm_hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_relations={self.num_relations},\n"
            f"  use_pearl={self.use_pearl}\n"
            f")"
        )
