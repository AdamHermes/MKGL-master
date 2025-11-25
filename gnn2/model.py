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

def repeat_graph(x, edge_index, batch_size):
    """
    TorchDrug RepeatGraph → PyG implementation.
    Replicates a graph batch_size times.
    """
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    all_x = []
    all_edge = []
    all_batch = []

    for i in range(batch_size):
        node_offset = i * num_nodes
        x_i = x.clone()
        edge_i = edge_index + node_offset

        all_x.append(x_i)
        all_edge.append(edge_i)
        all_batch.append(torch.full((num_nodes,), i, dtype=torch.long, device=x.device))

    return (
        torch.cat(all_x, dim=0),
        torch.cat(all_edge, dim=1),
        torch.cat(all_batch, dim=0),
    )

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
        print("hidden shape:", hidden.shape)
        print("rel shape:", rel.shape)
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
        x = x.to(torch.bfloat16)
        x[t_idx] = t_emb
        x[h_idx] = h_emb

        # score = 0 for all nodes, except head
        score = torch.zeros(batch.size(0), device=x.device)
        score = score.to(torch.bfloat16)
        score[h_idx] = self.score(h_emb, rel_emb)
        score = score.repeat_interleave((batch == batch.unique()[0]).sum())

        return x, score

    ###########################################
    # select nodes/edges based on score
    ###########################################
    def select_edges(self, edge_index, score, batch, node_ratio, degree_ratio):
        """
        PyG rewrite of TorchDrug's select_edges().
        """
        num_nodes = score.size(0)
        k = int(node_ratio * num_nodes)

        # top-k nodes
        _, idx = torch.topk(score, k)
        selected = idx

        # neighbors of selected nodes
        eidx, cols = neighbors(edge_index, selected)

        # top edges by score on neighbor nodes
        e = int(len(eidx) * degree_ratio)
        _, top = torch.topk(score[cols], e)

        return eidx[top]

    ###########################################
    # main forward
    ###########################################
    def forward(self, h_index, r_index, t_index,
                hidden_states, rel_hidden_states,
                edge_index, x, batch):
        """
        x: initial graph node features
        edge_index: graph edges
        hidden_states: head entity embedding
        rel_hidden_states: unused in original (so same here)
        """

        # negative sampling
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index
        )

        # get relation embedding
        rel_emb = self.rel_embedding(r_index[:, 0])

        # Repeat graph (batch_size copies)
        batch_size = h_index.size(0)
        x_rep, e_rep, b_rep = repeat_graph(x, edge_index, batch_size)

        # adjust indices
        num_nodes = x.size(0)
        offsets = torch.arange(batch_size, device=x.device) * num_nodes
        h_index = h_index + offsets[:, None]
        t_index = t_index + offsets[:, None]

        # Init embeddings
        x_rep, score = self.init_input_embeds(
            x_rep, hidden_states, h_index[:, 0],
            rel_hidden_states, t_index[:, 0],
            rel_emb, b_rep
        )

        # Run several PNA layers with dynamic edge pruning
        curr_x = x_rep.clone()
        curr_edge = e_rep.clone()

        for _layer in self.gnn.layers:
            # prune edges
            selected_edges = self.select_edges(
                curr_edge, score, b_rep,
                self.node_ratio, self.degree_ratio
            )

            e_sub = curr_edge[:, selected_edges]

            # gnn update
            new_x = _layer(curr_x, e_sub)

            # update x and score
            curr_x = curr_x + new_x
            # Expand rel_emb to match curr_x's node dimension
            rel_expanded = rel_emb.unsqueeze(1).repeat(1, num_nodes, 1)  # [batch, num_nodes, hidden]
            rel_expanded = rel_expanded.view(-1, rel_emb.size(-1))       # [batch*num_nodes, hidden]

            score = self.score(curr_x, rel_expanded)


        # final score for tails
        final = score[t_index]
        return final
