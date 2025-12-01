import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import PNAConv
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.utils import degree
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential

from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
#############################################
# Helper Functions (TorchDrug → PyG)
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

    # repeat node features efficiently
    x_rep = x.repeat(batch_size, 1)  # (B*N, D)

    # offsets for edges
    offsets = (torch.arange(batch_size, device=device, dtype=edge_index.dtype) * N).unsqueeze(1)  # (B,1)
    edge_index_rep = edge_index.unsqueeze(0) + offsets.unsqueeze(-1)  # (B, 2, E)
    edge_index_rep = edge_index_rep.view(2, -1).to(edge_index.dtype)  # (2, B*E)

    # node -> batch mapping
    node_batch = torch.arange(batch_size, device=device).repeat_interleave(N)  # (B*N,)

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
    Per-graph selection of edges:
      - For each graph instance b in the batch, pick top-k nodes by 'score' (k = node_ratio * num_nodes_b)
      - Gather outgoing edges of those nodes (edge_index refers to the full repeated graph)
      - From those candidate edges, choose top-e edges by node scores of their target nodes (e = degree_ratio * number_of_candidate_edges)
    Returns: concatenated selected edge indices (1D long tensor, indices into the columns of `edge_index`)
    """
    device = edge_index.device
    unique_batches = torch.unique(batch)
    selected = []

    for b in unique_batches:
        mask_nodes = (batch == b)
        node_ids = torch.nonzero(mask_nodes, as_tuple=True)[0]  # indices of nodes in rep graph
        if node_ids.numel() == 0:
            continue

        scores_b = score[node_ids]  # per-node scores for this graph
        k = max(1, int(node_ratio * node_ids.numel()))
        if scores_b.numel() <= k:
            top_node_idx = torch.arange(scores_b.numel(), device=device)
        else:
            _, top_node_idx = torch.topk(scores_b, k)

        selected_nodes = node_ids[top_node_idx]  # actual node ids in rep graph

        # neighbors: edges where row (src) in selected_nodes
        edge_idxs, cols = neighbors(edge_index, selected_nodes)
        if edge_idxs.numel() == 0:
            continue

        e = max(1, int(len(edge_idxs) * degree_ratio))
        # choose top edges by score of their target nodes (cols gives targets)
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
# PyG PNA (simple backbone)
#############################################

class PNA(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=6):
        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(num_layers):
            conv = PNAConv(in_channels=in_dim, out_channels=out_dim,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch, score, rel_per_node):
        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            edge_index = select_edges_pyg(x, score)
            x = batch_norm(conv(x, edge_index, edge_attr))
            score = self.score(x, rel_per_node).type(score.dtype)


        #x = global_add_pool(x, batch)
        return score
    
#############################################
# PyG ConditionedPNA (Faithful Rewrite)
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
        super().__init__(in_dim, out_dim, num_relations, num_layers=num_layers, avg_deg=1.0)

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.num_relations = num_relations
        # backbone GNN

        self.rel_embedding = nn.Embedding(num_relations * 2, in_dim)

        # scoring MLP
        feature_dim = in_dim + out_dim
        self.linear = nn.Linear(feature_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_layers)])

    ###########################################
    # scoring
    ###########################################
    def score(self, hidden, rel):
        h = torch.cat([hidden, rel], dim=-1)
        heur = self.linear(h)
        x = heur * hidden
        return self.mlp(x).squeeze(-1)

    ###########################################
    # negative sample swapping
    ###########################################
    def negative_sample_to_tail(self, h, t, r):
        is_t_neg = (h == h[:, [0]]).all(dim=-1, keepdim=True)
        new_h = torch.where(is_t_neg, h, t)
        new_t = torch.where(is_t_neg, t, h)
        new_r = torch.where(is_t_neg, r, r + self.num_relations)
        return new_h, new_t, new_r

    ###########################################
    # init input embeddings
    ###########################################
    def init_input_embeds(self, x, h_emb, h_idx, t_emb, t_idx, rel_emb, batch):
        # Tạo x với dtype mặc định
        x = x.clone()
        # Đảm bảo t_emb và h_emb cùng kiểu với x trước khi gán
        x[t_idx] = t_emb.to(x.dtype)
        x[h_idx] = h_emb.to(x.dtype)
        # Tính score "nền" (background) dựa trên relation embedding và vector 0
        dummy_hidden = torch.zeros_like(rel_emb)
        background_score = self.score(dummy_hidden, rel_emb)
        # Mở rộng background score cho từng node dựa trên batch index
        score = background_score[batch]
        score = score.to(torch.float32)
        # score = torch.zeros(batch.size(0), device=x.device, dtype=torch.float32)
        calculated_score = self.score(h_emb, rel_emb)
        score[h_idx] = calculated_score.to(dtype=torch.float32)
        # Mở rộng score
        # score = score.repeat_interleave((batch == batch.unique()[0]).sum())
        return x, score

    ###########################################
    # select nodes/edges based on score
    ###########################################

    def aggregate_pyg(self, graph, h_index_rep, r_index_rep, input_embeds, rel_embeds, init_score, batch_rep):
        
        device = graph.x.device
        num_nodes_rep = graph.num_nodes

        input_embeds = input_embeds.float()
        init_score = init_score.float()
        rel_embeds = rel_embeds.float()

        # boundary and score equivalent
        boundary = input_embeds.clone()
        score = init_score.clone()
        hidden = boundary.clone()

        # The rel embedding per-node (index by batch)
        rel_per_node = rel_embeds[batch_rep]  # shape (B*N, rel_dim)

        # compute pna_degree_mean if needed (approx)
        # use out-degree on e_rep
        score = super().forward(graph.x, graph.edge_index, graph.edge_attr, batch_rep, score, rel_per_node)
        # iterate layers
        

    ###########################################
    # main forward
    ###########################################
    def forward(self, h_index, r_index, t_index,
                hidden_states, rel_hidden_states,
                graph, score_text_embs, all_index):
        """
        h_index, r_index, t_index : (batch_size, k) indices (k = 1 + num_negative)
        hidden_states: head embeddings (batch_size, in_dim)  -- from LLM mapping
        rel_hidden_states: (unused)
        graph: PyG Data (we use graph.edge_index)
        score_text_embs: node features (num_nodes, in_dim)  <- THIS is x
        all_index: original node indices (torch.arange(num_nodes))  (not used here)
        """

        # 1) negative-sample swapping (same as before)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)

        batch_size = h_index.size(0)
        device = score_text_embs.device
        num_nodes = score_text_embs.size(0)

        # 2) IMPORTANT: take the head/tail *original* node indices BEFORE offsets
        #    we'll need original tail embeddings (from score_text_embs) to place into repeated graph
        head_orig_idx = h_index[:, 0].clone()   # shape (batch_size,)
        tail_orig_idx = t_index[:, 0].clone()   # shape (batch_size,)

        # 3) build repeated graph tensors (repeat node features and edges)
        # edge_index_undirected = to_undirected(graph.edge_index)
        #graph = repeat_graph(score_text_embs, edge_index_undirected, batch_size)
        # x_rep: (B * num_nodes, in_dim)
        # e_rep: (2, B * num_edges)
        # batch_rep: (B * num_nodes,
        N = graph.num_nodes
        batch_rep = torch.arange(batch_size, device=device).repeat_interleave(N)  # (B*N,)

        # 4) compute offsets and shift h/t indices to the repeated graph indexing
        offsets = (torch.arange(batch_size, device=device, dtype=torch.long) * num_nodes).unsqueeze(1)  # (B,1)
        h_index = h_index + offsets.to(h_index.dtype).to(device)
        t_index = t_index + offsets.to(t_index.dtype).to(device)

        # 5) relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0]).to(hidden_states.dtype)  # (batch_size, in_dim)

        # 6) prepare head and tail embeddings for init_input_embeds
        # head embeddings come from 'hidden_states' (already per-batch)
        head_embs = hidden_states.to(input_embeds.dtype)

        # tail embeddings must be collected from the original node features score_text_embs
        tail_embs = score_text_embs[tail_orig_idx].to(input_embeds.dtype)  # (batch_size, in_dim)

        # 7) init input embeddings for the repeated graph
        # signature: init_input_embeds(self, x, h_emb, h_idx, t_emb, t_idx, rel_emb, batch)
        input_embeds, init_score = self.init_input_embeds(
            graph.x,                # repeated node features
            head_embs,            # head embeddings (batch_size, in_dim)
            h_index[:, 0],        # indices in repeated graph for heads (batch_size,)
            tail_embs,            # tail embeddings (batch_size, in_dim)
            t_index[:, 0],        # indices in repeated graph for tails (batch_size,)
            rel_embeds,
            batch_rep
        )
        
        score = self.aggregate_pyg(
                            graph = graph,
                            h_index_rep=h_index[:, 0],   # already shifted indexes into rep graph
                            r_index_rep=r_index[:, 0],
                            input_embeds=input_embeds,
                            rel_embeds=rel_embeds,       # shape (batch_size, dim)
                            init_score=init_score,
                            batch_rep=batch_rep)
        # After computing hidden or score
        print("Score stats:", score.min().item(), score.max().item(), torch.isnan(score).any())

        final = score[t_index]   # t_index already shifted earlier

        # 10) final indexing to get scores for requested tails (t_index already shifted to repeated index)
        return final
    