import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import PNAConv
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.utils import degree

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


def degree(edge_index, num_nodes):
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=row.device)
    deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
    return deg

def neighbors(edge_index, nodes):
    """
    Returns (edge_idx, neighbor_nodes)
    """
    row, col = edge_index
    mask = torch.isin(row, nodes)
    out_edges = torch.nonzero(mask).squeeze(-1)
    out_nodes = col[mask]
    return out_edges, out_nodes


#############################################
# PyG PNA (simple backbone)
#############################################




class PNA(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations, num_layers=3):
        super().__init__()
        aggr = ['mean', 'max', 'min', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.layers = nn.ModuleList()
        deg_placeholder = torch.ones(1)  # temp placeholder
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
        # Compute degrees dynamically
        num_nodes = x.size(0)
        deg = degree(edge_index[0], num_nodes=num_nodes, dtype=x.dtype)

        h = x
        for conv in self.layers:
            h_new = conv(h, edge_index, deg=deg)  # pass deg here
            if self.short_cut:
                h_new += h
            h = h_new
        return h




#############################################
# PyG ConditionedPNA (Faithful Rewrite)
#############################################

class ConditionedPNA(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_relations,
        num_layers=3,
        node_ratio=0.1,
        degree_ratio=1.0
    ):
        super().__init__()

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.num_relations = num_relations

        # backbone GNN
        self.gnn = PNA(in_dim, out_dim, num_relations, num_layers)

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
        x = torch.zeros_like(x)
        #x = x.to(torch.bfloat16)
        x[t_idx] = t_emb
        x[h_idx] = h_emb

        # score = 0 for all nodes, except head
        score = torch.zeros(batch.size(0), device=x.device)
        #score = score.to(torch.bfloat16)
        score[h_idx] = self.score(h_emb, rel_emb)
        score = score.repeat_interleave((batch == batch.unique()[0]).sum())

        return x, score

    ###########################################
    # select nodes/edges based on score
    ###########################################
    def select_edges(self, edge_index, score, batch, node_ratio, degree_ratio):
        """
        Memory-safe edge selection (PyG rewrite).
        Performs per-batch top-k instead of global top-k.
        """

        selected_edges = []

        # loop over each graph instance in the minibatch
        for b in batch.unique():
            mask = (batch == b)                    # nodes belonging to graph b
            nodes_b = mask.nonzero(as_tuple=True)[0]

            scores_b = score[nodes_b]
            k_b = max(1, int(self.node_ratio * nodes_b.numel()))

            # top-k nodes of this graph
            _, idx_b = torch.topk(scores_b, k_b)
            selected_nodes = nodes_b[idx_b]

            # gather neighbors
            eidx, cols = neighbors(edge_index, selected_nodes)

            if len(eidx) == 0:
                continue  # skip empty subgraphs

            # pick top edges among neighbors by score
            e = max(1, int(len(eidx) * self.degree_ratio))
            _, top = torch.topk(score[cols], e)

            selected_edges.append(eidx[top])
            print("Score NaNs:", torch.isnan(score).any())
            print("Edge index:", edge_index.shape)
            print("Selected edges empty?", len(selected_edges) == 0)
            print("k_b values:", k_b)
            print("e values:", e)


        if len(selected_edges) == 0:
            return torch.empty(0, dtype=torch.long, device=edge_index.device)

        return torch.cat(selected_edges)


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
        x_rep, e_rep, batch_rep = repeat_graph(score_text_embs, graph.edge_index, batch_size)
        # x_rep: (B * num_nodes, in_dim)
        # e_rep: (2, B * num_edges)
        # batch_rep: (B * num_nodes,)

        # 4) compute offsets and shift h/t indices to the repeated graph indexing
        offsets = (torch.arange(batch_size, device=device, dtype=torch.long) * num_nodes).unsqueeze(1)  # (B,1)
        h_index = h_index + offsets.to(h_index.dtype).to(device)
        t_index = t_index + offsets.to(t_index.dtype).to(device)

        # 5) relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0]).to(hidden_states.dtype)  # (batch_size, in_dim)

        # 6) prepare head and tail embeddings for init_input_embeds
        # head embeddings come from 'hidden_states' (already per-batch)
        head_embs = hidden_states.to(x_rep.dtype)

        # tail embeddings must be collected from the original node features score_text_embs
        tail_embs = score_text_embs[tail_orig_idx].to(x_rep.dtype)  # (batch_size, in_dim)

        # 7) init input embeddings for the repeated graph
        # signature: init_input_embeds(self, x, h_emb, h_idx, t_emb, t_idx, rel_emb, batch)
        input_embeds, init_score = self.init_input_embeds(
            x_rep,                # repeated node features
            head_embs,            # head embeddings (batch_size, in_dim)
            h_index[:, 0],        # indices in repeated graph for heads (batch_size,)
            tail_embs,            # tail embeddings (batch_size, in_dim)
            t_index[:, 0],        # indices in repeated graph for tails (batch_size,)
            rel_embeds.to(x_rep.dtype),
            batch_rep
        )

        # 8) aggregate / run the conditioned PNA logic (your aggregate expects graph-like object)
        # The original TorchDrug code passes a graph-like object. Here we must adapt:
        # We will create a tiny namespace 'rep_graph' with the minimal attributes aggregate() expects.
        # aggregate() in your code expects attributes like: .num_node, .num_nodes?, .edge_index, .degree_out, .node2graph, etc.
        # If your aggregate() implementation uses PyG-style tensors, adapt accordingly.
        # For now, call your aggregate with (we assume it accepts (edge_index, x_rep, batch_rep, ...)).
        #
        # NOTE: If your aggregate() is the PyG rewrite that expects (graph, ...), you will likely need to adapt it similarly.
        #
        # For simplicity, if aggregate expects (graph, ...), create a small object:
        rep_graph = type("RG", (), {})()
        rep_graph.x = x_rep
        rep_graph.edge_index = e_rep
        rep_graph.batch = batch_rep
        rep_graph.num_node = x_rep.size(0)
        rep_graph.num_nodes = x_rep.size(0)
        rep_graph.num_edge = e_rep.size(1)
        rep_graph.node2graph = batch_rep  # mapping node->graph
        rep_graph.edge_list = e_rep.t().contiguous()
        # if aggregate expects degree_out use degree() helper to set it:
        try:
            rep_graph.degree_out = degree(e_rep[0], rep_graph.num_node)
        except Exception:
            rep_graph.degree_out = None

        # 9) call aggregate (your aggregate signature in the PyG rewrite earlier was aggregate(graph, h_index, r_index, input_embeds, rel_embeds, init_score))
        score = self.aggregate(rep_graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)

        # 10) final indexing to get scores for requested tails (t_index already shifted to repeated index)
        final = score[t_index]   # shape: (batch_size, k) presumably
        return final
