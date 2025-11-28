import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv
from new_retriever import ContextRetriever

class GatedPNALayer(nn.Module):
    """
    A custom GNN layer where the message passing is 'gated' or 'conditioned' 
    by a query vector (from the LLM).
    
    Logic preserved from MKGL:
    1. Calculate a compatibility score (heuristic) between Node Features and LLM Query.
    2. Use this score to scale (gate) the Node Features.
    3. Aggregate the gated features using PNA.
    """
    def __init__(self, in_channels, out_channels, aggregators, scalers, deg, towers=1):
        super().__init__()
        # The 'Query' (LLM state) interacts with Node Features to create a gate
        self.gate_linear = nn.Linear(in_channels * 2, in_channels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )
        
        # Standard PNA Conv from PyG
        self.pna_conv = PNAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=towers,
            pre_layers=1,
            post_layers=1,
            divide_input=False
        )

    def forward(self, x, edge_index, query_vector):
        """
        x: Node features (Num_Nodes, Hidden)
        edge_index: Graph connectivity
        query_vector: LLM Hidden State (Batch, Hidden) -> Broadcasted to nodes
        """
        # 1. Conditioning / Gating
        # Expand query to match number of nodes (assuming batch 1 for simplicity in inference, 
        # or graph nodes are batched appropriately)
        # Note: In a real batch, x is (Batch*Nodes, Dim). Query needs to align.
        # For this implementation, we assume x contains candidates relevant to the current query.
        
        num_nodes = x.size(0)
        # Broadcast query: If query is (1, Dim) and x is (N, Dim)
        query_expanded = query_vector.expand(num_nodes, -1)
        
        # Calculate Gate: Heuristic = MLP(Linear(cat(x, query)))
        combined = torch.cat([x, query_expanded], dim=-1)
        gate_input = F.relu(self.gate_linear(combined))
        gate_score = torch.sigmoid(self.gate_mlp(gate_input)) # (N, 1)
        
        # Modulate node features based on relevance to the LLM query
        x_gated = x * gate_score
        
        # 2. Message Passing (PNA)
        # We aggregate the *gated* features. Nodes irrelevant to the query get suppressed.
        out = self.pna_conv(x_gated, edge_index)
        
        # Residual connection + Update
        out = out + x # Simple residual
        return out, gate_score

class ScoreRetriever(nn.Module):
    def __init__(self, config, context_retriever, deg_histogram):
        super().__init__()
        self.config = config
        self.deg = deg_histogram
        
        # Reuse the Text processing parts from the ContextRetriever
        # (The Score Retriever also needs to understand entity names)
        self.context_retriever = context_retriever 
        
        hidden_dim = config.llm_hidden_dim
        r_dim = config.r
        
        # Down-project LLM hidden state to match GNN dimension
        self.query_proj = nn.Linear(hidden_dim, r_dim, bias=False)
        
        # Gated GNN Layers
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                GatedPNALayer(r_dim, r_dim, aggregators, scalers, deg_histogram)
            )
            
        # Final Score Head
        self.score_head = nn.Linear(r_dim, 1)

    def forward(self, llm_hidden_state, candidate_ids, edge_index):
        """
        llm_hidden_state: The last hidden state from the LLM (Batch, Hidden)
        candidate_ids: The entity IDs we are scoring (N)
        edge_index: The subgraph connectivity
        """
        # 1. Get Initial Embeddings for Candidates
        # We reuse the ContextRetriever's text logic to get base features for candidates
        # (Note: In optimization, these can be cached)
        with torch.no_grad():
            token_ids = self.context_retriever.kgl2token_ids[candidate_ids]
            raw_embs = self.context_retriever.llm_emb(token_ids)
            
        x = self.context_retriever.down_scale(raw_embs)
        mask = (token_ids != 0).long()
        x = self.context_retriever.text_pna(x, mask) # (N, r_dim)
        
        # 2. Project LLM Query
        query = self.query_proj(llm_hidden_state) # (Batch, r_dim)
        
        # 3. Gated Message Passing
        # We assume batch size 1 for graph operations or that candidate_ids belongs to the single query.
        # Handling Batched Graphs in PyG requires 'batch' vector. Simplified here for clarity.
        
        for layer in self.layers:
            x, gates = layer(x, edge_index, query)
            x = F.relu(x)
            
        # 4. Final Scoring
        # The score is a projection of the refined node features
        logits = self.score_head(x).squeeze(-1) # (N,)
        
        return logits