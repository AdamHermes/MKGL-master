import torch
from torch import nn
import torch.nn.functional as F
import math
#from torchdrug import core
from gnn2.model import *
from gnn2.layer import PNALayer


class SemanticAttentionModule(nn.Module):
    """
    Dynamic Semantic Attention Module.
    
    Enriches head entity representations with semantically similar entities
    using precomputed sparse semantic adjacency (from FAISS offline index).
    
    This module:
    1. Retrieves top-K semantic neighbors for input entities from sparse matrix
    2. Computes attention weights using dot-product + precomputed similarity scores
    3. Aggregates neighbor features via weighted sum
    4. Blends with original representation via learnable gate
    """
    
    def __init__(self, config, hidden_dim):
        """
        Args:
            config: Semantic attention config with keys:
                - k: Number of semantic neighbors to use
                - threshold: Minimum similarity (already applied during index build)
                - temperature: Softmax temperature for attention
                - gate_init: Initial value for blending gate
                - score_weight: Weight for precomputed similarity in attention
            hidden_dim: Dimension of hidden states (e.g., 2048 for TinyLlama)
        """
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.k = config.get('k', 10)
        self.temperature = config.get('temperature', 1.0)
        self.score_weight = config.get('score_weight', 0.5)
        
        # Learnable blending gate: alpha * original + (1-alpha) * semantic
        gate_init = config.get('gate_init', 0.1)
        self.gate = nn.Parameter(torch.tensor([gate_init]))
        
        # Optional: projection layers for attention computation
        self.use_projection = config.get('use_projection', False)
        if self.use_projection:
            proj_dim = config.get('proj_dim', 256)
            self.query_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
            self.key_proj = nn.Linear(hidden_dim, proj_dim, bias=False)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Semantic adjacency (loaded externally)
        self.register_buffer('semantic_indices', None)  # [num_entities, max_k]
        self.register_buffer('semantic_scores', None)   # [num_entities, max_k]
        self.num_entities = 0
        self.max_k = 0
    
    def load_semantic_neighbors(self, semantic_adj_path):
        """
        Load precomputed semantic neighbors from file.
        
        Args:
            semantic_adj_path: Path to .pt file with semantic adjacency
        """
        data = torch.load(semantic_adj_path, map_location='cpu')
        
        self.semantic_indices = data['dense_indices']  # [num_entities, max_k]
        self.semantic_scores = data['dense_scores']    # [num_entities, max_k]
        self.num_entities = data['num_entities']
        self.max_k = self.semantic_indices.shape[1]
        
        print(f"[SemanticAttention] Loaded {self.num_entities} entities, max_k={self.max_k}")
    
    def set_semantic_neighbors(self, indices, scores, num_entities):
        """
        Set semantic neighbors directly (for dynamic loading).
        
        Args:
            indices: [num_entities, max_k] - neighbor indices (-1 for padding)
            scores: [num_entities, max_k] - similarity scores
            num_entities: Number of entities
        """
        self.semantic_indices = indices
        self.semantic_scores = scores
        self.num_entities = num_entities
        self.max_k = indices.shape[1]
    
    def get_semantic_neighbors(self, entity_ids, k=None):
        """
        Retrieve top-k semantic neighbors for given entities.
        
        Args:
            entity_ids: [batch_size] or [batch_size, num_candidates] entity IDs
            k: Number of neighbors (default: self.k)
        
        Returns:
            neighbor_ids: [batch_size, k] or [batch_size, num_candidates, k]
            neighbor_scores: Same shape, similarity scores
            valid_mask: Same shape, True for valid neighbors
        """
        if self.semantic_indices is None:
            raise RuntimeError("Semantic neighbors not loaded. Call load_semantic_neighbors() first.")
        
        k = k or self.k
        k = min(k, self.max_k)
        
        device = entity_ids.device
        original_shape = entity_ids.shape
        entity_ids_flat = entity_ids.view(-1)
        
        # Clamp to valid range
        entity_ids_clamped = entity_ids_flat.clamp(0, self.num_entities - 1)
        
        # Lookup neighbors
        indices = self.semantic_indices.to(device)
        scores = self.semantic_scores.to(device)
        
        neighbor_ids = indices[entity_ids_clamped, :k]  # [flat_batch, k]
        neighbor_scores = scores[entity_ids_clamped, :k]
        
        # Valid mask: neighbor_id != -1
        valid_mask = neighbor_ids >= 0
        
        # Reshape back
        if len(original_shape) > 1:
            neighbor_ids = neighbor_ids.view(*original_shape, k)
            neighbor_scores = neighbor_scores.view(*original_shape, k)
            valid_mask = valid_mask.view(*original_shape, k)
        
        return neighbor_ids, neighbor_scores, valid_mask
    
    def forward(self, entity_ids, hidden_states, all_entity_embeddings, 
                return_attention=False):
        """
        Compute semantic-enriched representations.
        
        Args:
            entity_ids: [batch_size] - head entity IDs
            hidden_states: [batch_size, hidden_dim] - LLM hidden states for heads
            all_entity_embeddings: [num_entities, embed_dim] - all entity embeddings
            return_attention: Whether to return attention weights
        
        Returns:
            enriched_hidden: [batch_size, hidden_dim] - semantically enriched states
            attention_weights: (optional) [batch_size, k] - attention weights
        """
        if self.semantic_indices is None or self.k == 0:
            # No semantic neighbors: return original
            if return_attention:
                return hidden_states, None
            return hidden_states
        
        batch_size = entity_ids.shape[0]
        device = hidden_states.device
        
        # Get semantic neighbors
        neighbor_ids, precomputed_scores, valid_mask = self.get_semantic_neighbors(entity_ids)
        # neighbor_ids: [batch_size, k]
        # precomputed_scores: [batch_size, k]
        # valid_mask: [batch_size, k]
        
        # Replace invalid (-1) with 0 for embedding lookup (will be masked)
        neighbor_ids_safe = neighbor_ids.clamp(min=0)
        
        # Get neighbor embeddings
        neighbor_embeddings = all_entity_embeddings[neighbor_ids_safe]  # [batch, k, embed_dim]
        
        # Compute attention scores
        if self.use_projection:
            # Project to attention space
            queries = self.query_proj(hidden_states)  # [batch, proj_dim]
            keys = self.key_proj(neighbor_embeddings)  # [batch, k, proj_dim]
            
            # Scaled dot-product attention
            attn_scores = torch.bmm(keys, queries.unsqueeze(-1)).squeeze(-1)  # [batch, k]
            attn_scores = attn_scores / math.sqrt(keys.shape[-1])
        else:
            # Simple dot-product with dimension matching
            if hidden_states.shape[-1] != neighbor_embeddings.shape[-1]:
                # Dimension mismatch: fallback to precomputed scores only
                # This can happen if embeddings haven't been up-scaled
                attn_scores = precomputed_scores
            else:
                attn_scores = torch.bmm(
                    neighbor_embeddings, 
                    hidden_states.unsqueeze(-1)
                ).squeeze(-1)  # [batch, k]
                attn_scores = attn_scores / math.sqrt(hidden_states.shape[-1])
        
        # Combine with precomputed similarity scores
        combined_scores = (1 - self.score_weight) * attn_scores + self.score_weight * precomputed_scores
        
        # Apply temperature
        combined_scores = combined_scores / self.temperature
        
        # Mask invalid neighbors
        combined_scores = combined_scores.masked_fill(~valid_mask, float('-inf'))
        
        # Softmax attention
        attention_weights = F.softmax(combined_scores, dim=-1)  # [batch, k]
        
        # Handle all-invalid case (set to zero)
        all_invalid = ~valid_mask.any(dim=-1, keepdim=True)  # [batch, 1]
        attention_weights = attention_weights.masked_fill(all_invalid, 0.0)
        
        # Aggregate neighbor features
        if self.use_projection:
            neighbor_values = self.value_proj(neighbor_embeddings)
        else:
            # Need to project neighbor embeddings to hidden_dim if different
            if neighbor_embeddings.shape[-1] != self.hidden_dim:
                # Dimension mismatch: skip aggregation, return original
                if return_attention:
                    return hidden_states, attention_weights
                return hidden_states
            neighbor_values = neighbor_embeddings
        
        # Weighted sum: [batch, k, hidden] * [batch, k, 1] -> [batch, hidden]
        semantic_context = torch.bmm(
            attention_weights.unsqueeze(1), 
            neighbor_values
        ).squeeze(1)  # [batch, hidden_dim]
        
        # Blend with original via gate
        gate = torch.sigmoid(self.gate)
        enriched_hidden = gate * hidden_states + (1 - gate) * semantic_context
        
        if return_attention:
            return enriched_hidden, attention_weights
        return enriched_hidden
    
    def extra_repr(self):
        return f'k={self.k}, temperature={self.temperature}, score_weight={self.score_weight}'

class BasePNARetriever(nn.Module): 
    '''
    Retrieve text information
    '''

    
    def __init__(self, config, text_embeddings, kgl2token, orig_vocab_size):
        super().__init__()
        self.config = config
        self.text_embeddings =text_embeddings
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        self.down_scaling = nn.Linear(
                self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)
        
        if self.config.text_encoder == 'pna':
            self.re_scaling = nn.Linear(config.r*12, self.config.r)
    
    
    def aggregate_text(self, token_ids, text_embeddings, method='pna'):
        device = text_embeddings.device
        
        token_ids = token_ids.to(device) # Batch x Length
        token_mask = (token_ids > 0).unsqueeze(-1).to(device) # B x L X 1
        token_lengths = token_mask.half().sum(axis=1).to(device) # B X 1
        degree = token_lengths
        token_embs = text_embeddings[token_ids] # B x L x Hidden
        
        mean = (token_embs * token_mask).sum(axis=1) / token_lengths
        if method == 'mean':
            result = mean
        else:
            sq_mean = (token_embs**2 * token_mask).sum(axis=1) / \
                token_lengths
            max, _ = (token_embs*token_mask).max(axis=1)
            min, _ = (token_embs*token_mask).min(axis=1)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            features = torch.cat(
                [mean, max, min, std], dim=-1)
            
            scale = degree.log()
            scale = scale / scale.mean()
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            
            result = (features.unsqueeze(-1) *
                      scales.unsqueeze(-2)).flatten(-2)

        return result 
    
    def retrieve_text(self, token_ids):
        # token_ids: num_kgl_tokens x num_tokens
        R = self.down_scaling(self.text_embeddings)
        
        result = self.aggregate_text(token_ids, R, self.config.text_encoder)
        
        if self.config.text_encoder == 'pna':
            result = self.re_scaling(result)
        
        return self.norm(result)

    def norm(self, x):
        return F.normalize(x, p=2, dim=1)
                
    
    def forward(self, kgl_ids=None):
        if kgl_ids is not None:
            kgl_ids = kgl_ids - self.orig_vocab_size
            token_ids = self.kgl2token[kgl_ids.cpu()]
        else:
            token_ids = self.kgl2token
        return self.retrieve_text(token_ids)
        


class ContextRetriever(BasePNARetriever):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.up_scaling = nn.Linear(
                self.config.r, self.config.llm_hidden_dim, bias=False, dtype=torch.float)

    def forward(self, kgl_ids, graph, all_index, all_kgl_index):
        text_embs = super().forward(kgl_ids)
        context = self.up_scaling(text_embs)
        return context

        

        
class ScoreRetriever(BasePNARetriever):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        cfg_kg = config.kg_encoder
        cfg_base = cfg_kg.base_layer
        
        base_layer = PNALayer(
            input_dim=cfg_base.input_dim,          # 32
            output_dim=cfg_base.output_dim,        # 32
            num_relation=cfg_kg.num_relation,            # PASSED FROM DATASET
            query_input_dim=cfg_base.query_input_dim,
            aggregate_func=cfg_base.get("aggregate_func", "pna"),
            layer_norm=cfg_base.get("layer_norm", "yes"),
            dependent=cfg_base.get("dependent", "yes")
        )
        
        # 3. Initialize ConditionedPNA
        # It will clone 'base_layer' 6 times (num_layer=6)
        self.kg_retriever = ConditionedPNA(
            base_layer=base_layer,
            num_layer=cfg_kg.get("num_layer", 6),
            num_mlp_layer=cfg_kg.get("num_mlp_layer", 2),
            node_ratio=cfg_kg.get("node_ratio", 0.1),
            degree_ratio=cfg_kg.get("degree_ratio", 1),
            remove_one_hop=cfg_kg.get("remove_one_hop", "yes")
        )
        
        # Down-scaling layers
        self.h_down_scaling = nn.Linear(
            self.config.llm_hidden_dim, self.config.r, 
            bias=False, dtype=torch.float
        )
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim, self.config.r, 
            bias=False, dtype=torch.float
        )

    def forward(self, h_id, r_id, t_id, hidden_states, rel_hidden_states, 
                graph, all_index, all_kgl_index):
        
        score_text_embs = super().forward(all_kgl_index)
        
        # Down-scale LLM embeddings to model dimension
        head_embeds = self.h_down_scaling(hidden_states)
        rel_embeds = self.r_down_scaling(rel_hidden_states)
        
        # Get scores from KG retriever
        score = self.kg_retriever(
            h_id, r_id, t_id, 
            head_embeds, rel_embeds, 
            graph, 
            score_text_embs, 
            all_index
        )
        
        return score

class RelScoreRetriever(BasePNARetriever):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.r_down_scaling = nn.Linear(
                self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)

    def forward(self, rel_hidden_states, all_rel_kgl_index):
        score_text_embs = super().forward(all_rel_kgl_index) # num rel, r
        rel_embeds = self.r_down_scaling(rel_hidden_states) # batch size, r
        score = F.linear(rel_embeds, score_text_embs)
        return score