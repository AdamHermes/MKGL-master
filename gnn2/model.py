import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from .util import VirtualTensor, bincount, variadic_topks, to_undirected_with_inverse
import copy
from .layer import MLP

def print_stat(name, tensor):
    if tensor is None:
        print(f"DEBUG: {name} is None")
        return
    t = tensor.float() # Convert to float for stat calculation to avoid overflow/underflow issues in stats
    print(f"DEBUG: {name} | Shape: {list(t.shape)} | Min: {t.min().item():.4f} | Max: {t.max().item():.4f} | Mean: {t.mean().item():.4f} | NaNs: {torch.isnan(t).sum().item()}")

class PNA(nn.Module):
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()        
        self.num_relation = getattr(base_layer, 'num_relation', None) 
        self.remove_one_hop = remove_one_hop
        self.layers = nn.ModuleList()        
        for i in range(num_layer):
            self.layers.append(copy.deepcopy(base_layer))            
        feature_dim = base_layer.output_dim + base_layer.input_dim        
        from .layer import MLP 
        self.mlp = MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])
        self.short_cut = getattr(base_layer, 'short_cut', False)

    def aggregate(self, graph, input_embeds):
        layer_input = input_embeds
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut:
                hidden = hidden + layer_input
            layer_input = hidden
        return hidden

    def init_input_embeds(self, graph, input_embeds, input_index):
        input_embeds = torch.zeros(graph.num_node, input_embeds.shape[-1], device=input_embeds.device)
        input_embeds[input_index] = input_embeds
        return input_embeds

    def forward(self, graph, input_embeds, input_index):
        new_index, new_attr = to_undirected_with_inverse(
            graph.edge_index, 
            graph.edge_attr, 
            num_relations = self.num_relation
        )
        graph.edge_index = new_index
        graph.edge_attr = new_attr        
        input_embeds = self.init_input_embeds(graph, input_embeds, input_index)
        output = self.aggregate(graph, input_embeds)
        return output





class ConditionedPNA(PNA):
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, node_ratio=0.1, degree_ratio=1, test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, **kwargs):
        super().__init__(base_layer, num_layer, num_mlp_layer=num_mlp_layer, **kwargs)

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie

        feature_dim = base_layer.output_dim + base_layer.input_dim        
        #self.rel_embedding = nn.Embedding(base_layer.num_relation * 2, base_layer.input_dim)
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        
        self.mlp = MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])


    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, score_text_embs, all_index):

        graph = graph.clone()
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        new_index, new_attr = to_undirected_with_inverse(
            graph.edge_index,
            graph.edge_attr,
            num_relations = self.num_relation
        )
        graph.edge_index = new_index
        graph.edge_attr = new_attr

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        
        batch_size = len(h_index)
        graph_list = [graph.clone() for _ in range(batch_size)]
        graph = Batch.from_data_list(graph_list)
        graph.node2graph = graph.batch
        
        node_counts = graph.ptr[:-1] 
        h_index = h_index + node_counts.unsqueeze(-1).to(h_index.device)
        t_index = t_index + node_counts.unsqueeze(-1).to(t_index.device)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        rel_embeds = rel_hidden_states
        rel_embeds = rel_embeds.type(hidden_states.dtype)

        input_embeds, init_score = self.init_input_embeds(graph, hidden_states, h_index[:, 0], score_text_embs, all_index, rel_embeds)
        
        score = self.aggregate(graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)
        score = score[t_index]
        return score

    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        batch_size = len(rel_embeds)  # 32
        
        query = rel_embeds  # [32, 32]
        boundary, score = input_embeds, init_score
        hidden = boundary.clone()
        
        # Store query as [batch_size, dim], NOT expanded
        graph.query = query  # [32, 32] ✓
        graph.boundary = boundary
        graph.hidden = hidden
        graph.score = score
        
        graph.node_id = torch.arange(graph.num_nodes, device=h_index.device)
        
        graph.degree_out = degree(graph.edge_index[0], graph.num_nodes)
        graph.pna_degree_out = graph.degree_out.unsqueeze(-1)  # Add this!
        pna_degree_mean = (graph.degree_out + 1).log().mean()
        
        # Add this for proper batch detection
        graph.num_graphs = batch_size

        for i, layer in enumerate(self.layers):
            edge_id_subset = self.select_edges(graph, graph.score)
           
            sub_edge_index = graph.edge_index[:, edge_id_subset]
            sub_edge_attr = graph.edge_attr[edge_id_subset]
            
            
            
            unique_nodes, new_edge_index = sub_edge_index.unique(return_inverse=True)
            new_edge_index = new_edge_index.reshape(2, -1)
            
            subgraph = Data(
                edge_index=new_edge_index,
                edge_type=sub_edge_attr,  # Use edge_type, not edge_attr
                num_nodes=unique_nodes.size(0)
            )
            
            # Set edge_attr as well for compatibility
            if sub_edge_attr is not None:
                subgraph.edge_attr = sub_edge_attr
            
            subgraph.score = graph.score[unique_nodes]
            subgraph.hidden = graph.hidden[unique_nodes]
            subgraph.boundary = graph.boundary[unique_nodes]
            subgraph.degree_out = degree(subgraph.edge_index[0], subgraph.num_nodes)
            subgraph.pna_degree_out = subgraph.degree_out.unsqueeze(-1)
            subgraph.pna_degree_mean = pna_degree_mean
            
            # CRITICAL FIX: Keep query as [batch_size, dim], don't expand per-node
            subgraph.query = graph.query  # [32, 32] ✓ NOT per-node!
            
            # But store node-to-batch mapping for message passing
            subgraph.batch = graph.batch[unique_nodes]
            subgraph.node2graph = subgraph.batch  # For relation lookup
            subgraph.node_id = unique_nodes
            subgraph.num_graphs = batch_size  # For batch size detection
            
            # Gating
            gate = F.sigmoid(subgraph.score).unsqueeze(-1)
            layer_input = gate * subgraph.hidden
            
            # Run layer
            hidden_out = layer(subgraph, layer_input.type(torch.float32))
            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]
            
            graph.hidden[node_out] = (graph.hidden[node_out] + hidden_out[out_mask]).type(graph.hidden[node_out].dtype)
            
            index = graph.node2graph[node_out]
            
            new_scores = self.score(graph.hidden[node_out],query[index])         
            graph.score[node_out] = new_scores.type(graph.score[node_out].dtype)

        return graph.score


    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index,  rel_embeds):
        input_embeds = VirtualTensor.zeros(graph.num_nodes, rel_embeds.shape[1], device=rel_embeds.device, dtype=rel_embeds.dtype)
        
        
        input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        input_embeds[head_index] = head_embeds

        score = VirtualTensor.gather(self.score(torch.zeros_like(rel_embeds), rel_embeds), graph.node2graph) # zero all
        score[head_index] = self.score(head_embeds, rel_embeds)
            
        return input_embeds, score

    def score(self, hidden, rel_embeds):
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score

    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        num_nodes_per_graph = bincount(graph.batch, minlength=graph.num_graphs)
        
        edge_batch_ids = graph.batch[graph.edge_index[0]]
        total_edges_per_graph = bincount(edge_batch_ids, minlength=graph.num_graphs)

        ks = (num_nodes_per_graph.float() * node_ratio).long()
        ks = torch.clamp(ks, min=1)
        ks = torch.min(ks, num_nodes_per_graph)

        index = variadic_topks(score, num_nodes_per_graph, ks=ks, break_tie=self.break_tie)[1]
        node_in = index 

        src_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.edge_index.device)
        src_mask[node_in] = True
        
        edge_mask_in = src_mask[graph.edge_index[0]]
        
        candidate_edge_batch = graph.batch[graph.edge_index[0][edge_mask_in]]
        num_candidate_edges = bincount(candidate_edge_batch, minlength=graph.num_graphs)
        
        avg_degree = total_edges_per_graph.float() / num_nodes_per_graph.float().clamp(min=1)
        es = (degree_ratio * ks.float() * avg_degree).long()
        
        es = torch.clamp(es, min=1)
        es = torch.min(es, num_candidate_edges)

    
        valid_edge_indices = torch.nonzero(edge_mask_in).squeeze()
        
        node_out = graph.edge_index[1][valid_edge_indices]
        score_edge = score[node_out]
        
        final_edge_indices = variadic_topks(score_edge, num_candidate_edges, ks=es, break_tie=self.break_tie)[1]
        
        return valid_edge_indices[final_edge_indices]
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        edge_rels = graph.edge_attr.squeeze()
        num_nodes = graph.num_nodes
        num_rels = max(edge_rels.max().item(), r_index.max().item()) + 1

        if self.remove_one_hop:
            h_ext = torch.cat([h_index, t_index], dim=-1)
            t_ext = torch.cat([t_index, h_index], dim=-1)
            
            graph_hashes = graph.edge_index[0].long() * num_nodes + graph.edge_index[1].long()
            batch_hashes = h_ext.long() * num_nodes + t_ext.long()
        else:
            graph_hashes = (graph.edge_index[0].long() * num_nodes + graph.edge_index[1].long()) * num_rels + edge_rels.long()
            batch_hashes = (h_index.long() * num_nodes + t_index.long()) * num_rels + r_index.long()

        mask_to_remove = torch.isin(graph_hashes, batch_hashes)
        final_mask = ~mask_to_remove
        
        graph.edge_index = graph.edge_index[:, final_mask]
        graph.edge_attr = graph.edge_attr[final_mask]
        
        return graph

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index
    
