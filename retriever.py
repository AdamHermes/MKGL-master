import torch
from torch import nn
import torch.nn.functional as F
from gnn2.model import ConditionedGAT, GAT


class BasePNARetriever(nn.Module): 
    '''
    Retrieve text information
    '''
    
    def __init__(self, config, text_embeddings, kgl2token, orig_vocab_size):
        super().__init__()
        self.config = config
        self.text_embeddings = text_embeddings
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        self.down_scaling = nn.Linear(
            text_embeddings.shape[1],  # use actual embedding dim instead of config
            self.config.r, 
            bias=False, 
            dtype=torch.float
        )

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
            sq_mean = (token_embs**2 * token_mask).sum(axis=1) / token_lengths
            max, _ = (token_embs*token_mask).max(axis=1)
            min, _ = (token_embs*token_mask).min(axis=1)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            features = torch.cat([mean, max, min, std], dim=-1)
            
            scale = degree.log()
            scale = scale / scale.mean()
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            
            result = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

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
    """
    Context retriever that can optionally use GAT for graph-based context.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.up_scaling = nn.Linear(
            self.config.r, 
            self.config.llm_hidden_dim, 
            bias=False, 
            dtype=torch.float
        )
        
        # Optional: Initialize GAT if kg_encoder is specified
        if hasattr(config, 'kg_encoder'):
            cfg_kg = config.kg_encoder
            base_layer_config = {
                'input_dim': cfg_kg.base_layer.input_dim,
                'output_dim': cfg_kg.base_layer.output_dim,
                'query_input_dim': cfg_kg.base_layer.query_input_dim,
                'layer_norm': cfg_kg.base_layer.layer_norm,
                'dependent': cfg_kg.base_layer.dependent,
                'num_heads': getattr(cfg_kg.base_layer, 'num_heads', 4),
                'dropout': getattr(cfg_kg.base_layer, 'dropout', 0.1),
                'aggregate_func': cfg_kg.base_layer.aggregate_func,
            }
            
            self.kg_retriever = GAT(
                base_layer=base_layer_config,
                num_layer=cfg_kg.num_layer,
                num_mlp_layer=getattr(cfg_kg, 'num_mlp_layer', 2),
                remove_one_hop=cfg_kg.remove_one_hop
            )
        else:
            self.kg_retriever = None

    def forward(self, kgl_ids, graph=None, all_index=None, all_kgl_index=None):
        """
        Forward pass for context retrieval.
        
        Args:
            kgl_ids: KG token IDs
            graph: Optional graph structure for GAT
            all_index: Optional entity indices
            all_kgl_index: Optional KG lookup indices
            
        Returns:
            context: Context embeddings [batch_size, llm_hidden_dim]
        """
        text_embs = super().forward(kgl_ids)
        
        # If graph is provided and we have a kg_retriever, use it
        if graph is not None and self.kg_retriever is not None:
            # Apply GAT to enhance embeddings with graph structure
            text_embs = self.kg_retriever(graph, text_embs)
        
        context = self.up_scaling(text_embs)
        return context


class ScoreRetriever(BasePNARetriever):
    """
    Score retriever using GAT for knowledge graph reasoning.
    Compatible with the provided YAML config structure.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        cfg_kg = config.kg_encoder
        
        # Build base_layer config dict from YAML config
        base_layer_config = {
            'input_dim': cfg_kg.base_layer.input_dim,
            'output_dim': cfg_kg.base_layer.output_dim,
            'query_input_dim': cfg_kg.base_layer.query_input_dim,
            'layer_norm': cfg_kg.base_layer.layer_norm,
            'dependent': cfg_kg.base_layer.dependent,
            'num_heads': getattr(cfg_kg.base_layer, 'num_heads', 4),
            'dropout': getattr(cfg_kg.base_layer, 'dropout', 0.1),
            'aggregate_func': cfg_kg.base_layer.aggregate_func,
        }
        
        # Initialize ConditionedGAT (replaces ConditionedPNA)
        self.kg_retriever = ConditionedGAT(
            base_layer=base_layer_config,
            num_layer=cfg_kg.num_layer,
            num_mlp_layer=getattr(cfg_kg, 'num_mlp_layer', 2),
            node_ratio=cfg_kg.node_ratio,
            degree_ratio=getattr(cfg_kg, 'degree_ratio', 1.0),
            test_node_ratio=getattr(cfg_kg, 'test_node_ratio', None),
            test_degree_ratio=getattr(cfg_kg, 'test_degree_ratio', None),
            remove_one_hop=cfg_kg.remove_one_hop,
            break_tie=getattr(cfg_kg, 'break_tie', False)
        )
        
        # Down-scaling projections
        self.h_down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float
        )
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim,
            self.config.r,
            bias=False,
            dtype=torch.float
        )

    def forward(self, h_id, r_id, t_id, hidden_states, rel_hidden_states, 
                graph, all_index, all_kgl_index):
        """
        Forward pass for scoring triples.
        
        Args:
            h_id: Head entity IDs [batch_size, num_neg]
            r_id: Relation IDs [batch_size, num_neg]
            t_id: Tail entity IDs [batch_size, num_neg]
            hidden_states: Entity hidden states from LLM [num_entities, llm_hidden_dim]
            rel_hidden_states: Relation hidden states from LLM [num_relations, llm_hidden_dim]
            graph: Knowledge graph structure
            all_index: All entity indices
            all_kgl_index: Knowledge graph lookup indices
            
        Returns:
            score: Scores for each triple [batch_size, num_neg]
        """
        # Get text embeddings from parent class
        score_text_embs = super().forward(all_kgl_index)
        
        # Project hidden states to reduced dimension
        head_embeds = self.h_down_scaling(hidden_states) 
        rel_embeds = self.r_down_scaling(rel_hidden_states)
        
        # Score using GAT-based KG retriever
        score = self.kg_retriever(
            h_id, r_id, t_id, 
            head_embeds, rel_embeds, 
            graph, score_text_embs, all_index
        )
        
        return score


class RelScoreRetriever(BasePNARetriever):
    """
    Relation score retriever for scoring relations.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.r_down_scaling = nn.Linear(
            self.config.llm_hidden_dim, 
            self.config.r, 
            bias=False, 
            dtype=torch.float
        )

    def forward(self, rel_hidden_states, all_rel_kgl_index):
        """
        Forward pass for relation scoring.
        
        Args:
            rel_hidden_states: Relation hidden states [batch_size, llm_hidden_dim]
            all_rel_kgl_index: Relation KG lookup indices
            
        Returns:
            score: Relation scores [batch_size, num_relations]
        """
        # Get text embeddings for all relations
        score_text_embs = super().forward(all_rel_kgl_index)  # [num_rel, r]
        
        # Project relation hidden states
        rel_embeds = self.r_down_scaling(rel_hidden_states)  # [batch_size, r]
        
        # Compute similarity scores
        score = F.linear(rel_embeds, score_text_embs)
        
        return score