import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

# PyG Imports
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, degree

# Local Utils (Assumed available)
from .util import VirtualTensor, bincount, variadic_topks
# Note: RepeatGraph is usually replaced by PyG's Batch.from_data_list([g]*n)
# but if you have a custom RepeatGraph for PyG, you can import it.

class PNA(nn.Module):
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()
        
        # In TorchDrug, base_layer is often a config dict or prototype.
        # We assume base_layer is an instantiated object or class we can copy.
        import copy
        
        self.num_relation = getattr(base_layer, 'num_relation', None) 
        self.remove_one_hop = remove_one_hop
        self.layers = nn.ModuleList()
        
        for i in range(num_layer):
            self.layers.append(copy.deepcopy(base_layer))
            
        feature_dim = base_layer.output_dim + base_layer.input_dim
        
        # Assuming you have a similar MLP class in your utils or use standard nn.Sequential
        from .layer import MLP # or torchdrug.layers if you still import it
        self.mlp = MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])
        self.short_cut = getattr(base_layer, 'short_cut', False)

    def aggregate(self, graph, input_embeds):
        layer_input = input_embeds
        for layer in self.layers:
            # PyG layers typically expect: x, edge_index, edge_attr
            # We assume your base_layer.__call__ handles PyG Data objects
            hidden = layer(graph, layer_input)
            
            if self.short_cut:
                hidden = hidden + layer_input
            layer_input = hidden
            
        return hidden

    def init_input_embeds(self, graph, input_embeds, input_index):
        # Using VirtualTensor as requested
        # Ensure graph.num_nodes is used (PyG syntax) vs graph.num_node (TorchDrug)
        input_embeds_full = VirtualTensor.zeros(graph.num_nodes, input_embeds.shape[-1], 
                                                device=input_embeds.device, dtype=input_embeds.dtype)
        input_embeds_full[input_index] = input_embeds
        return input_embeds_full

    def forward(self, graph, input_embeds, input_index):
        # 1. Undirected
        if graph.edge_attr is not None:
             edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
             graph.edge_attr = edge_attr
        else:
             edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        graph.edge_index = edge_index
        
        # 2. Init
        input_embeds = self.init_input_embeds(graph, input_embeds, input_index)
        
        # 3. Aggregate
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
        
        self.rel_embedding = nn.Embedding(self.num_relation * 2, base_layer.input_dim)
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        
        from .layer import MLP
        self.mlp = MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])


    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, score_text_embs, all_index):
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        # PyG: To Undirected
        if graph.edge_attr is not None:
            edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
            graph.edge_attr = edge_attr
        else:
            edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        graph.edge_index = edge_index

        # Negative Sampling
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        
        # Batching (RepeatGraph)
        batch_size = len(h_index)
        # Using PyG Batch to replicate the graph logic
        graph_list = [graph.clone() for _ in range(batch_size)]
        graph = Batch.from_data_list(graph_list)
        
        # Offset Logic (PyG uses graph.ptr for offsets [0, N, 2N...])
        node_counts = graph.ptr[:-1] 
        h_index = h_index + node_counts.unsqueeze(-1).to(h_index.device)
        t_index = t_index + node_counts.unsqueeze(-1).to(t_index.device)
        
        # Assertions
        # assert (h_index[:, [0]] == h_index).all() # Optional safety check

        rel_embeds = self.rel_embedding(r_index[:, 0]) 
        rel_embeds = rel_embeds.type(hidden_states.dtype) #+ rel_hidden_states

        input_embeds, init_score = self.init_input_embeds(graph, hidden_states, h_index[:, 0], score_text_embs, all_index, rel_embeds)
        
        score = self.aggregate(graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)
        score = score[t_index]
        return score

    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        # Store context
        query = rel_embeds
        boundary, score = input_embeds, init_score
        hidden = boundary.clone()
        
        # In PyG, we attach these to the Data object directly
        graph.query = query
        graph.boundary = boundary
        graph.hidden = hidden
        graph.score = score
        
        # PyG equivalent of Range(graph.num_node)
        graph.node_id = torch.arange(graph.num_nodes, device=h_index.device)
        
        # Calculate Degree
        # Assuming undirected, so out_degree == degree
        graph.degree_out = degree(graph.edge_index[0], graph.num_nodes)
        graph.pna_degree_out = graph.degree_out

        # PNA Degree Mean (mean of log(d+1))
        pna_degree_mean = (graph.degree_out + 1).log().mean()

        for layer in self.layers:
            # 1. Select Edges (returns 1D indices of edges to keep)
            edge_id_subset = self.select_edges(graph, graph.score)
            
            # 2. Create Compact Subgraph (Manual logic to preserve mappings)
            sub_edge_index = graph.edge_index[:, edge_id_subset]
            sub_edge_attr = graph.edge_attr[edge_id_subset] if graph.edge_attr is not None else None
            
            # unique_nodes maps: 0..SubN -> Original_Graph_Node_ID
            unique_nodes, new_edge_index = sub_edge_index.unique(return_inverse=True)
            
            subgraph = Data(edge_index=new_edge_index, edge_attr=sub_edge_attr)
            subgraph.num_nodes = unique_nodes.size(0)
            
            # Transfer attributes
            subgraph.score = graph.score[unique_nodes]
            subgraph.hidden = graph.hidden[unique_nodes]
            subgraph.degree_out = graph.degree_out[unique_nodes]
            subgraph.node_id = graph.node_id[unique_nodes]
            subgraph.pna_degree_mean = pna_degree_mean
            
            # Prepare Layer Input
            layer_input = F.sigmoid(subgraph.score).unsqueeze(-1) * subgraph.hidden
            
            # Layer Forward
            hidden_out = layer(subgraph, layer_input.type(torch.float32))
            
            # 3. Scatter Updates back to original graph
            out_mask = subgraph.degree_out > 0
            
            # Which nodes in the ORIGINAL graph do we update?
            active_original_ids = unique_nodes[out_mask]
            
            # Update Hidden
            graph.hidden[active_original_ids] = (graph.hidden[active_original_ids] + hidden_out[out_mask]).type(graph.hidden.dtype)

            # Update Scores
            # Map original node IDs to their batch index to get the correct Query
            batch_idx = graph.batch[active_original_ids]
            
            new_scores = self.score(graph.hidden[active_original_ids], graph.query[batch_idx])
            graph.score[active_original_ids] = new_scores.type(graph.score.dtype)

        return graph.score

    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index, rel_embeds):
        # Using VirtualTensor
        input_embeds = VirtualTensor.zeros(graph.num_nodes, rel_embeds.shape[1], 
                                           device=rel_embeds.device, dtype=rel_embeds.dtype)
        
        # Assuming tail_index matches PyG broadcast rules or is handled by VirtualTensor
        # If VirtualTensor supports the original TorchDrug broadcasting logic, this line is fine:
        input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        input_embeds[head_index] = head_embeds

        # Init Score
        # Zero all scores
        # We rely on VirtualTensor.gather or standard indexing
        # Note: 'graph.node2graph' in TorchDrug is 'graph.batch' in PyG
        
        # Recreating logic: score = VirtualTensor.gather(..., graph.batch)
        # Using standard PyG approach for the 'zero all' base:
        # We create scores for ALL nodes based on the query of their respective graph
        # query[graph.batch] expands Query [BatchSize, D] -> [NumNodes, D]
        
        # If you want to use VirtualTensor for scores too:
        # score = VirtualTensor.gather(self.score(torch.zeros_like(rel_embeds), rel_embeds), graph.batch) 
        # But standard tensor is likely safer here:
        
        expanded_query = rel_embeds[graph.batch]
        score_all = self.score(torch.zeros(graph.num_nodes, rel_embeds.shape[1], device=rel_embeds.device), expanded_query)
        
        # Use VirtualTensor for the result if preferred, or standard tensor
        # The original code returned a tensor for 'score'.
        
        # Update specific head scores
        score_all[head_index] = self.score(head_embeds, rel_embeds)
            
        return input_embeds, score_all

    def score(self, hidden, rel_embeds):
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score

    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        ks = (node_ratio * graph.num_nodes).long()
        es = (degree_ratio * ks * graph.num_edges / graph.num_nodes).long()

        # PyG: Count per graph
        num_nodes_per_graph = bincount(graph.batch, minlength=graph.num_graphs)
        ks = torch.min(ks, num_nodes_per_graph)
        
        # TopK Nodes
        index = variadic_topks(score, num_nodes_per_graph, ks=ks, break_tie=self.break_tie)[1]
        node_in = index 
        
        # Count Edges for chosen nodes
        # Create mask for source nodes
        src_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)
        src_mask[node_in] = True
        edge_mask_in = src_mask[graph.edge_index[0]]
        
        # Edges per graph count
        edge_batch = graph.batch[graph.edge_index[0][edge_mask_in]]
        num_edges_per_graph = bincount(edge_batch, minlength=graph.num_graphs)
        
        es = torch.min(es, num_edges_per_graph)

        # TopK Edges
        valid_edge_indices = torch.nonzero(edge_mask_in).squeeze()
        node_out = graph.edge_index[1][valid_edge_indices]
        score_edge = score[node_out]
        
        final_edge_indices = variadic_topks(score_edge, num_edges_per_graph, ks=es, break_tie=self.break_tie)[1]
        
        # Return Global Edge IDs
        return valid_edge_indices[final_edge_indices]
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        # PyG Version: Uses vectorized hashing instead of loop
        if graph.edge_attr is None:
             raise ValueError("Graph must have edge_attr (relation IDs) for remove_easy_edges")

        edge_rels = graph.edge_attr.squeeze()
        num_nodes = graph.num_nodes
        num_rels = max(edge_rels.max().item(), r_index.max().item()) + 1

        if self.remove_one_hop:
            h_ext = torch.cat([h_index, t_index], dim=0)
            t_ext = torch.cat([t_index, h_index], dim=0)
            
            # Hash (h, t)
            graph_hashes = graph.edge_index[0].long() * num_nodes + graph.edge_index[1].long()
            batch_hashes = h_ext.long() * num_nodes + t_ext.long()
        else:
            # Hash (h, t, r)
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