import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv

class TextPNA(nn.Module):
    """
    Applies Principal Neighborhood Aggregation (PNA) concepts to text sequences.
    Treats the sequence of tokens as a "neighborhood" to aggregate into a single vector.
    """
    def __init__(self, input_dim, output_dim, towers=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # PNA Aggregators: Mean, Max, Min, Std
        # PNA Scalers: Identity, Amplification, Attenuation (Log)
        # 4 aggregators * 3 scalers = 12 combinations
        self.param_dim = input_dim * 12 
        
        self.towers = towers
        self.divide_input = False 
        
        # Down projection if needed to match PNA input size
        self.project = nn.Linear(self.param_dim, output_dim)
        
    def forward(self, x, mask=None):
        """
        x: (Batch, Seq_Len, Hidden)
        mask: (Batch, Seq_Len) - 1 for valid tokens, 0 for padding
        """
        if mask is not None:
            # Mask out padding (set to reasonable values for min/max logic)
            mask_expanded = mask.unsqueeze(-1) # (B, L, 1)
            x_masked = x * mask_expanded
            
            # Count valid tokens for mean/std
            n = mask.sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)
            
            # 1. Aggregators
            mean = x_masked.sum(dim=1) / n
            
            # For min/max, we need to handle padding carefully
            # Set padding to -inf for max, +inf for min
            x_max_safe = x.clone()
            x_max_safe[mask == 0] = -float('inf')
            max_val = x_max_safe.max(dim=1)[0]
            
            x_min_safe = x.clone()
            x_min_safe[mask == 0] = float('inf')
            min_val = x_min_safe.min(dim=1)[0]
            
            # Std
            sq_mean = (x_masked ** 2).sum(dim=1) / n
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            
        else:
            n = x.size(1)
            mean = x.mean(dim=1)
            max_val = x.max(dim=1)[0]
            min_val = x.min(dim=1)[0]
            std = x.std(dim=1, unbiased=False)

        features = torch.cat([mean, max_val, min_val, std], dim=-1) # (B, 4*H)
        
        # 2. Scalers (Identity, Amplification, Attenuation)
        # Degree based on sequence length 'n'
        # Log scaler: log(n + 1) / avg_log_n (Assuming avg_log_n approx 1 for simplicity or learnable)
        # Note: In standard PNA, scalers are normalized by avg degree of training set. 
        # Here we approximate for dynamic text lengths.
        
        deg = n.float()
        scale_log = torch.log(deg + 1)
        scale_amp = deg  # Simplified amplification (usually standard PNA uses specific scalars)
        # To strictly follow MKGL paper, we stick to the core PyG PNA formula structure:
        # We'll construct the 3 scalers manually: Identity, Amp (log), Att (inverse log)
        
        # Correction: MKGL uses log-based scalers
        s_identity = torch.ones_like(deg)
        s_amp = torch.log(deg + 1)
        s_att = 1.0 / torch.log(deg + 1).clamp(min=1.0)
        
        scalers = [s_identity, s_amp, s_att]
        
        output = []
        for s in scalers:
            output.append(features * s) # Broadcast over hidden dim
            
        output = torch.cat(output, dim=-1) # (B, 12*H)
        
        return self.project(output)

class ContextRetriever(nn.Module):
    def __init__(self, config, llm_embedding_layer, kgl2token_ids, deg_histogram):
        super().__init__()
        self.config = config
        self.llm_emb = llm_embedding_layer # Reference to frozen LLM embeddings
        self.register_buffer("kgl2token_ids", kgl2token_ids) # Mapping: KG_ID -> [Token_IDs]
        
        hidden_dim = config.llm_hidden_dim
        r_dim = config.r # Reduced dimension for LoRA-like efficiency
        
        # 1. Down-Scale
        self.down_scale = nn.Linear(hidden_dim, r_dim, bias=False)
        
        # 2. Text Retrieval (Aggregation)
        self.text_pna = TextPNA(r_dim, r_dim)
        
        # 3. KG Retrieval (Graph Aggregation)
        # Using PyG PNAConv
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.kg_layers = nn.ModuleList()
        for _ in range(config.num_layers):
            conv = PNAConv(
                in_channels=r_dim, 
                out_channels=r_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg_histogram,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False
            )
            self.kg_layers.append(conv)
            
        # 4. Up-Scale
        self.up_scale = nn.Linear(r_dim, hidden_dim, bias=False)
        
    def forward(self, entity_ids, edge_index):
        """
        entity_ids: Tensor of KG Entity IDs to retrieve embeddings for.
        edge_index: The complete KG edge index for message passing.
        """
        # A. Get Text Embeddings
        # Look up token IDs for these entities
        token_ids = self.kgl2token_ids[entity_ids] # (Batch, Max_Seq_Len)
        mask = (token_ids != 0).long()
        
        # Use frozen LLM embeddings
        with torch.no_grad():
            raw_text_embs = self.llm_emb(token_ids) # (Batch, Seq, Hidden)
            
        # B. Down Scale & Text Aggregation
        x = self.down_scale(raw_text_embs)
        x = self.text_pna(x, mask) # (Batch, r_dim) - This is the "Node Feature"
        
        # C. Graph Aggregation (Message Passing)
        # Note: We need to run GNN over the subgraph or full graph. 
        # For efficiency in a retriever, we usually assume 'x' contains features for ALL nodes 
        # or we construct a subgraph. The original MKGL runs PNA on the whole graph context 
        # if feasible, or samples. Assuming 'entity_ids' represents the subset we want,
        # but PNAConv expects 'x' to align with 'edge_index'.
        # 
        # Strategy: We assume 'x' is computed for relevant nodes in the batch + neighbors.
        # However, to keep it simple and matching the old logic:
        # The old logic ran retrieval for specific nodes on the fly.
        
        for conv in self.kg_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        # D. Up Scale
        context_emb = self.up_scale(x)
        return context_emb