import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import PNAConv
from torch_geometric.utils import to_undirected, degree
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import BatchNorm, global_add_pool

#############################################
# Helper Functions
#############################################

def repeat_graph(x: torch.Tensor, edge_index: torch.Tensor, batch_size: int):
    """
    Repeat node features x (N, D) and edge_index (2, E) batch_size times.
    Returns:
      x_rep: (B*N, D)
      edge_index_rep: (2, B*E)
      node_batch: (B*N,) node->graph mapping (0..B-1)
    """
    device = x.device
    N = x.size(0)
    E = edge_index.size(1)

    x_rep = x.repeat(batch_size, 1)  # (B*N, D)

    offsets = (torch.arange(batch_size, device=device, dtype=edge_index.dtype) * N).unsqueeze(1)
    edge_index_rep = edge_index.unsqueeze(0) + offsets.unsqueeze(-1)
    edge_index_rep = edge_index_rep.view(2, -1).to(edge_index.dtype)

    node_batch = torch.arange(batch_size, device=device).repeat_interleave(N)

    return x_rep, edge_index_rep, node_batch


def neighbors(edge_index, nodes):
    """
    Returns (edge_idx, neighbor_nodes)
    """
    row, col = edge_index
    mask = torch.isin(row, nodes)
    out_edges = torch.nonzero(mask).squeeze(-1)
    out_nodes = col[mask]
    return out_edges, out_nodes


def select_edges_pyg(edge_index, score, batch, node_ratio=0.1, degree_ratio=1.0):
    """
    Per-graph selection of edges based on node scores.
    """
    device = edge_index.device
    unique_batches = torch.unique(batch)
    selected = []

    for b in unique_batches:
        mask_nodes = (batch == b)
        node_ids = torch.nonzero(mask_nodes, as_tuple=True)[0]
        if node_ids.numel() == 0:
            continue

        scores_b = score[node_ids]
        k = max(1, int(node_ratio * node_ids.numel()))
        if scores_b.numel() <= k:
            top_node_idx = torch.arange(scores_b.numel(), device=device)
        else:
            _, top_node_idx = torch.topk(scores_b, k)

        selected_nodes = node_ids[top_node_idx]

        edge_idxs, cols = neighbors(edge_index, selected_nodes)
        if edge_idxs.numel() == 0:
            continue

        e = max(1, int(len(edge_idxs) * degree_ratio))
        target_scores = score[cols]
        if target_scores.numel() <= e:
            top_e_idx = torch.arange(target_scores.numel(), device=device)
        else:
            _, top_e_idx = torch.topk(target_scores, e)

        selected.append(edge_idxs[top_e_idx])

    if len(selected) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(selected)


#############################################
# PNA Backbone (PyG only)
#############################################

class PNA(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_layers=3):
        super().__init__()
        aggr = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.layers = ModuleList()
        deg_placeholder = torch.ones(1)
        
        for _ in range(num_layers):
            self.layers.append(
                PNAConv(
                    in_dim,
                    out_dim,
                    aggregators=aggr,
                    scalers=scalers,
                    deg=deg_placeholder
                )
            )
        self.short_cut = True

    def forward(self, x, edge_index):
        h = x
        for conv in self.layers:
            h_new = conv(h, edge_index)
            if self.short_cut:
                h_new = h_new + h
            h = h_new
        return h


#############################################
# ConditionedPNA (PyG-based, TorchDrug-free)
#############################################

class ConditionedPNA(PNA):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_relations,
        num_layers=3,
        node_ratio=0.1,
        degree_ratio=1.0,
    ):
        super().__init__(in_dim, out_dim, num_relations, num_layers=num_layers)

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.num_relations = num_relations

        self.rel_embedding = nn.Embedding(num_relations * 2, in_dim)

        # scoring MLP
        feature_dim = in_dim + out_dim
        self.linear = nn.Linear(feature_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

    ###########################################
    # scoring function
    ###########################################
    def score(self, hidden, rel):
        """
        Compute per-node scores based on hidden state and relation embedding.
        Args:
            hidden: (num_nodes, hidden_dim)
            rel: (num_nodes, rel_dim)
        Returns:
            scores: (num_nodes,)
        """
        h = torch.cat([hidden, rel], dim=-1)
        heur = self.linear(h)
        x = heur * hidden
        return self.mlp(x).squeeze(-1)

    ###########################################
    # negative sample swapping
    ###########################################
    def negative_sample_to_tail(self, h, t, r):
        """
        Swap heads and tails for negative samples.
        Assumes first column is positive sample, rest are negatives.
        """
        is_t_neg = (h == h[:, [0]]).all(dim=-1, keepdim=True)
        new_h = torch.where(is_t_neg, h, t)
        new_t = torch.where(is_t_neg, t, h)
        new_r = torch.where(is_t_neg, r, r + self.num_relations)
        return new_h, new_t, new_r

    ###########################################
    # initialize input embeddings
    ###########################################
    def init_input_embeds(self, x, h_emb, h_idx, t_emb, t_idx, rel_emb, batch):
        """
        Initialize embeddings for the repeated graph.
        Place head and tail embeddings at their positions.
        """
        x = x.clone()
        x[t_idx] = t_emb.to(x.dtype)
        x[h_idx] = h_emb.to(x.dtype)

        # Initialize scores: 0 for all nodes except heads
        dummy_hidden = torch.zeros_like(rel_emb)
        background_score = self.score(dummy_hidden, rel_emb)
        score = background_score[batch]
        score = score.to(torch.float32)
        
        calculated_score = self.score(h_emb, rel_emb)
        score[h_idx] = calculated_score.to(dtype=torch.float32)

        return x, score

    ###########################################
    # aggregate via GNN layers
    ###########################################
    def aggregate_pyg(self, x_rep, e_rep, batch_rep, rel_embeds, input_embeds, init_score):
        """
        Run GNN aggregation with edge selection based on scores.
        Args:
            x_rep: (B*N, dim) repeated node features
            e_rep: (2, B*E) repeated edge indices
            batch_rep: (B*N,) node->graph mapping
            rel_embeds: (batch_size, rel_dim) relation embeddings
            input_embeds: (B*N, dim) initialized embeddings
            init_score: (B*N,) initial scores
        Returns:
            score: (B*N,) final per-node scores
        """
        device = x_rep.device
        num_nodes_rep = x_rep.size(0)

        boundary = input_embeds.clone()
        score = init_score.clone()
        hidden = boundary.clone()

        # Relation embedding per node (expand batch dimension)
        rel_per_node = rel_embeds[batch_rep]  # (B*N, rel_dim)

        # Iterate through GNN layers
        for i, layer in enumerate(self.layers):
            # Select edges based on current scores
            sel_edge_idx = select_edges_pyg(e_rep, score, batch_rep, 
                                           self.node_ratio, self.degree_ratio)

            if sel_edge_idx.numel() > 0:
                e_sub = e_rep[:, sel_edge_idx]
            else:
                e_sub = e_rep

            # GNN forward pass
            new_hidden = layer(hidden, e_sub)

            # Update hidden states (accumulate)
            if e_sub.size(1) > 0:
                out_deg = torch.zeros(num_nodes_rep, device=device, dtype=torch.float32)
                out_deg.scatter_add_(0, e_sub[0], torch.ones(e_sub.size(1), device=device))
                node_out = torch.nonzero(out_deg > 0, as_tuple=True)[0]
                hidden[node_out] = hidden[node_out] + new_hidden[node_out]
            else:
                hidden = hidden + new_hidden

            # Recompute scores
            score = self.score(hidden, rel_per_node).type(score.dtype)

        return score

    ###########################################
    # main forward pass
    ###########################################
    def forward(self, h_index, r_index, t_index,
                hidden_states, rel_hidden_states,
                graph, score_text_embs, all_index):
        """
        Forward pass for link prediction scoring.
        
        Args:
            h_index: (batch_size, k) head indices (k = 1 + num_negative)
            r_index: (batch_size, k) relation indices
            t_index: (batch_size, k) tail indices
            hidden_states: (batch_size, in_dim) head embeddings from LLM
            rel_hidden_states: unused
            graph: PyG Data object with .x, .edge_index, .num_nodes
            score_text_embs: (num_nodes, in_dim) node features
            all_index: node indices (not used)
        
        Returns:
            final: (batch_size, k) predicted scores
        """
        
        # 1) Swap negatives to tail
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        batch_size = h_index.size(0)
        device = score_text_embs.device
        num_nodes = score_text_embs.size(0)

        # 2) Extract original indices before offsetting
        head_orig_idx = h_index[:, 0].clone()
        tail_orig_idx = t_index[:, 0].clone()

        # 3) Build repeated graph tensors
        x_rep, e_rep, batch_rep = repeat_graph(graph.x, graph.edge_index, batch_size)

        # 4) Offset indices to repeated graph indexing
        offsets = (torch.arange(batch_size, device=device, dtype=torch.long) * num_nodes).unsqueeze(1)
        h_index = h_index + offsets.to(h_index.dtype).to(device)
        t_index = t_index + offsets.to(t_index.dtype).to(device)

        # 5) Get relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0]).to(hidden_states.dtype)  # (batch_size, in_dim)

        # 6) Prepare head and tail embeddings
        head_embs = hidden_states.to(x_rep.dtype)
        tail_embs = score_text_embs[tail_orig_idx].to(x_rep.dtype)

        # 7) Initialize input embeddings
        input_embeds, init_score = self.init_input_embeds(
            x_rep,
            head_embs,
            h_index[:, 0],
            tail_embs,
            t_index[:, 0],
            rel_embeds.to(x_rep.dtype),
            batch_rep
        )

        # 8) Run GNN aggregation
        score = self.aggregate_pyg(
            x_rep=x_rep,
            e_rep=e_rep,
            batch_rep=batch_rep,
            rel_embeds=rel_embeds,
            input_embeds=input_embeds,
            init_score=init_score
        )

        # 9) Index to get scores for requested tails
        final = score[t_index[:, 0]]

        return final