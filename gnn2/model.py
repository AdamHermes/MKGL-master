import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, degree
from .util import VirtualTensor, bincount, variadic_topks

def print_stat(name, tensor):
    if tensor is None:
        print(f"DEBUG: {name} is None")
        return
    t = tensor.float() # Convert to float for stat calculation to avoid overflow/underflow issues in stats
    print(f"DEBUG: {name} | Shape: {list(t.shape)} | Min: {t.min().item():.4f} | Max: {t.max().item():.4f} | Mean: {t.mean().item():.4f} | NaNs: {torch.isnan(t).sum().item()}")

class PNA(nn.Module):
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()
        import copy
        
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
        input_embeds_full = VirtualTensor.zeros(graph.num_nodes, input_embeds.shape[-1], 
                                                device=input_embeds.device, dtype=input_embeds.dtype)
        input_embeds_full[input_index] = input_embeds
        return input_embeds_full

    def forward(self, graph, input_embeds, input_index):
        if graph.edge_attr is not None:
             edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
             graph.edge_attr = edge_attr
        else:
             edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        graph.edge_index = edge_index
        
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
        
        self.rel_embedding = nn.Embedding(self.num_relation * 2, base_layer.input_dim)
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        
        from .layer import MLP
        self.mlp = MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])


    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, score_text_embs, all_index):
        print(f"DEBUG: START FORWARD | h_max={h_index.max()} t_max={t_index.max()} r_max={r_index.max()}")
        print(f"DEBUG: GRAPH STATS | num_nodes={graph.num_nodes} edge_index_max={graph.edge_index.max()} edge_attr_max={graph.edge_attr.max() if graph.edge_attr is not None else 'None'}")
        graph = graph.clone()
        if r_index.max() >= self.num_relation * 2:
            print(f"CRASH PENDING: r_index {r_index.max()} >= limit {self.num_relation * 2}")
        print("Got1")
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        print("Got2")
        max_id = graph.edge_index.max().item()
        if max_id >= graph.num_nodes:
            print(f"CRASH PENDING: Max Node ID ({max_id}) >= graph.num_nodes ({graph.num_nodes})")
            graph.num_nodes = max_id + 1
        if graph.edge_index.min() < 0:
            print(f"CRASH CAUSE: edge_index contains negative values! Min: {graph.edge_index.min()}")
            
        if graph.edge_attr is not None:
            if graph.edge_attr.min() < 0:
                print(f"CRASH CAUSE: edge_attr contains negative values! Min: {graph.edge_attr.min()}")
            
            if graph.edge_index.size(1) != graph.edge_attr.size(0):
                print(f"CRASH CAUSE: Shape Mismatch! edge_index cols={graph.edge_index.size(1)} vs edge_attr rows={graph.edge_attr.size(0)}")
        if graph.edge_attr is not None:
            print("Got3")
            if graph.edge_attr.max() >= self.num_relation:
                print(f"WARNING: edge_attr max {graph.edge_attr.max()} >= num_relation {self.num_relation} before undirected")
            try:
                edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
                graph.edge_attr = edge_attr
            except RuntimeError as e:
                print(f"CRASH IN TO_UNDIRECTED: {e}")
                print(f"State: nodes={graph.num_nodes}, edge_index_shape={graph.edge_index.shape}, edge_attr_shape={graph.edge_attr.shape}")
                raise e
        else:
            print("Got4")
            edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
            print("Got5")
        graph.edge_index = edge_index

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        
        batch_size = len(h_index)
        graph_list = [graph.clone() for _ in range(batch_size)]
        graph = Batch.from_data_list(graph_list)
        
        node_counts = graph.ptr[:-1] 
        h_index = h_index + node_counts.unsqueeze(-1).to(h_index.device)
        t_index = t_index + node_counts.unsqueeze(-1).to(t_index.device)
        
        if r_index[:, 0].max() >= self.rel_embedding.num_embeddings:
             print(f"CRASH PENDING: Rel Embedding Index {r_index[:, 0].max()} >= {self.rel_embedding.num_embeddings}")

        rel_embeds = self.rel_embedding(r_index[:, 0]) 
        rel_embeds = rel_embeds.type(hidden_states.dtype)
        # DEBUG: Check initial embeddings
        print_stat("Forward: Initial hidden_states", hidden_states)
        print_stat("Forward: Initial rel_embeds", rel_embeds)

        input_embeds, init_score = self.init_input_embeds(graph, hidden_states, h_index[:, 0], score_text_embs, all_index, rel_embeds)
        
        score = self.aggregate(graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)
        score = score[t_index]
        return score

    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        query = rel_embeds
        boundary, score = input_embeds, init_score
        hidden = boundary.clone()
        
        graph.query = query
        graph.boundary = boundary
        graph.hidden = hidden
        graph.score = score
        
        graph.node_id = torch.arange(graph.num_nodes, device=h_index.device)
        
        graph.degree_out = degree(graph.edge_index[0], graph.num_nodes)
        graph.pna_degree_out = graph.degree_out

        pna_degree_mean = (graph.degree_out + 1).log().mean()
        print("\n--- START AGGREGATE ---")
        print_stat("Aggregate: Init Score", graph.score)

        for i, layer in enumerate(self.layers):
            print(f"\n--- LAYER {i} START ---")
            print_stat(f"Layer {i}: graph.score (Start of Loop)", graph.score)
            
            # If this prints -62k, the corruption happened in init_input_embeds or passed init_score
            
            edge_id_subset = self.select_edges(graph, graph.score)
            
            # ... inside your aggregate loop ...
            
            sub_edge_index = graph.edge_index[:, edge_id_subset]
            sub_edge_attr = graph.edge_attr[edge_id_subset] if graph.edge_attr is not None else None
            
            # --- INSERT THIS DEBUG BLOCK ---
            if sub_edge_attr is not None:
                max_val = sub_edge_attr.max().item()
                limit = self.num_relation * 2
                print(f"DEBUG: Layer {i} | Edge Attr Max: {max_val} | Limit: {limit}")
                
                if max_val >= limit:
                    # This print proves the config is the issue, not the subgraph code
                    print(f"!!! CRASH DETECTED !!!")
                    print(f"You have a Relation ID {max_val} but only configured {limit} slots.")
                    print(f"Your 'sub_edge_attr' logic is correct, but the DATA is out of bounds.")
                    # We exit explicitly to avoid the confusing CUDA error
                    import sys; sys.exit(1)
            # -------------------------------

            unique_nodes, new_edge_index = sub_edge_index.unique(return_inverse=True)
            # ... continue ...
            
            subgraph = Data(edge_index=new_edge_index, edge_attr=sub_edge_attr)
            subgraph.num_nodes = unique_nodes.size(0)
            
            subgraph.score = graph.score[unique_nodes]
            subgraph.hidden = graph.hidden[unique_nodes]
            subgraph.degree_out = graph.degree_out[unique_nodes]
            subgraph.node_id = graph.node_id[unique_nodes]
            subgraph.pna_degree_mean = pna_degree_mean
            
            # Gating mechanism: check if sigmoid is saturating due to high score
            gate = F.sigmoid(subgraph.score).unsqueeze(-1)
            print_stat(f"Layer {i}: Gate (Sigmoid output)", gate)
            
            layer_input = gate * subgraph.hidden
            
            hidden_out = layer(subgraph, layer_input.type(torch.float32))
            
            out_mask = subgraph.degree_out > 0
            active_original_ids = unique_nodes[out_mask]
            
            # Update Hidden
            prev_hidden = graph.hidden[active_original_ids]
            update_delta = hidden_out[out_mask]
            
            # Check for explosion in hidden states (often causes score explosion next)
            if update_delta.abs().max() > 100:
                print(f"WARNING: Layer {i} hidden update delta is large!")
                print_stat(f"Layer {i}: Update Delta", update_delta)
                
            graph.hidden[active_original_ids] = (prev_hidden + update_delta).type(graph.hidden.dtype)
            print_stat(f"Layer {i}: Updated Hidden (Subset)", graph.hidden[active_original_ids])

            batch_idx = graph.batch[active_original_ids]
            
            # Update Score
            print(f"DEBUG: Layer {i} | Calculating new scores...")
            new_scores = self.score(graph.hidden[active_original_ids], graph.query[batch_idx])
            
            # Track the new scores BEFORE they go back into the graph
            print_stat(f"Layer {i}: New Scores Calculated", new_scores)
            
            graph.score[active_original_ids] = new_scores.type(graph.score.dtype)

        print("--- END AGGREGATE ---\n")
        return graph.score

    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index, rel_embeds):
        if tail_embeds.dtype != head_embeds.dtype:
            tail_embeds = tail_embeds.to(head_embeds.dtype)

        batch_size = rel_embeds.size(0)
        input_embeds_full = tail_embeds.repeat(batch_size, 1)
        
        if input_embeds_full.size(0) != graph.num_nodes:
             input_embeds_full = tail_embeds.repeat(batch_size, 1)

        input_embeds_full[head_index] = head_embeds

        expanded_query = rel_embeds[graph.batch]
        zero_embeds = torch.zeros(graph.num_nodes, rel_embeds.shape[1], 
                                  device=rel_embeds.device, dtype=rel_embeds.dtype)
        
        print("\nDEBUG: init_input_embeds calc start")
        score_all = self.score(zero_embeds, expanded_query)
        print_stat("init_input_embeds: Raw Score (Zero Embeds)", score_all)
        
        score_head = self.score(head_embeds, rel_embeds)
        print_stat("init_input_embeds: Raw Score (Head Embeds)", score_head)
        
        score_all[head_index] = score_head
        
        # Check before clamp
        print_stat("init_input_embeds: Score All (Pre-Clamp)", score_all)
        
        score_all = torch.clamp(score_all, min=-15, max=15)
        
        # Check after clamp
        print_stat("init_input_embeds: Score All (Post-Clamp)", score_all)
            
        return input_embeds_full, score_all

    def score(self, hidden, rel_embeds):
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        raw_score = self.mlp(x).squeeze(-1)
        if raw_score.abs().max() > 50 or torch.isnan(raw_score).any():
            print("  DEBUG: score() internal tracking:")
            print_stat("    score input: hidden", hidden)
            print_stat("    score input: heuristic", heuristic)
            print_stat("    score input: x (hidden*heuristic)", x)
            print_stat("    score output", raw_score)
        return raw_score

    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        num_nodes_per_graph = bincount(graph.batch, minlength=graph.num_graphs)
        
        ks = (num_nodes_per_graph.float() * node_ratio).long()
        ks = torch.clamp(ks, min=1) # Ensure at least 1 node kept per graph
        
        ks = torch.min(ks, num_nodes_per_graph)

        
        index = variadic_topks(score, num_nodes_per_graph, ks=ks, break_tie=self.break_tie)[1]
        node_in = index 
        
        # 3. Mask Sources
        src_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.edge_index.device)
        src_mask[node_in] = True
        
        edge_mask_in = src_mask[graph.edge_index[0]]
        
        edge_batch = graph.batch[graph.edge_index[0][edge_mask_in]]
        num_edges_per_graph = bincount(edge_batch, minlength=graph.num_graphs)
        
        es = (degree_ratio * ks.float() * (graph.num_edges / graph.num_nodes)).long()
        es = torch.clamp(es, min=1)
        
        if es.size(0) != num_edges_per_graph.size(0):
            
             es_aligned = torch.zeros_like(num_edges_per_graph)
             # Fill based on graph indices available, or just recalculate 'es' using full batch stats
             # Simpler approach: Recalculate 'es' entirely using batch-wise stats
             es = (degree_ratio * ks.float() * (num_edges_per_graph.float() / num_nodes_per_graph.float().clamp(min=1))).long()
             es = torch.clamp(es, min=1)

        es = torch.min(es, num_edges_per_graph)

        # 5. Select Top-K Edges
        valid_edge_indices = torch.nonzero(edge_mask_in).squeeze()
        node_out = graph.edge_index[1][valid_edge_indices]
        score_edge = score[node_out]
        
        final_edge_indices = variadic_topks(score_edge, num_edges_per_graph, ks=es, break_tie=self.break_tie)[1]
        
        # Map back to global edge indices
        return valid_edge_indices[final_edge_indices]
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        if graph.edge_attr is None:
             raise ValueError("Graph must have edge_attr (relation IDs) for remove_easy_edges")

        edge_rels = graph.edge_attr.squeeze()
        num_nodes = graph.num_nodes
        num_rels = max(edge_rels.max().item(), r_index.max().item()) + 1

        if self.remove_one_hop:
            h_ext = torch.cat([h_index, t_index], dim=0)
            t_ext = torch.cat([t_index, h_index], dim=0)
            
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