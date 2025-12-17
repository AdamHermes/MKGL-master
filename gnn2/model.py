import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, degree
from .util import VirtualTensor, bincount, variadic_topks
from .layer import MLP 
from torch_geometric.utils import subgraph as pyg_subgraph

def print_stat(name, tensor):
    if tensor is None:
        print(f"DEBUG: {name} is None")
        return
    t = tensor.float()
    print(f"DEBUG: {name} | Shape: {list(t.shape)} | Min: {t.min().item():.4f} | Max: {t.max().item():.4f} | Mean: {t.mean().item():.4f} | NaNs: {torch.isnan(t).sum().item()}")


class ConditionedPNA(nn.Module):
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, node_ratio=0.1, degree_ratio=1, 
                 test_node_ratio=None, test_degree_ratio=None, break_tie=False, remove_one_hop=False, **kwargs):
        super().__init__()
        import copy
        
        self.num_relation = getattr(base_layer, 'num_relation', None)
        self.remove_one_hop = remove_one_hop
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie
        
        # Clone layers
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(copy.deepcopy(base_layer))
        
        # Relation embeddings
        self.rel_embedding = nn.Embedding(self.num_relation * 2, base_layer.input_dim)
        
        # Scoring network
        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        self.mlp = MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, score_text_embs, all_index):
        graph = graph.clone()
        
        # Remove easy edges during training
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        
        # Make undirected with inverse relations
        if graph.edge_attr is not None:
            # Create reverse edges with num_relation offset
            reverse_edge_index = torch.stack([graph.edge_index[1], graph.edge_index[0]], dim=0)
            reverse_edge_attr = graph.edge_attr + self.num_relation  # Shift for reverse direction
            
            graph.edge_index = torch.cat([graph.edge_index, reverse_edge_index], dim=1)
            graph.edge_attr = torch.cat([graph.edge_attr, reverse_edge_attr], dim=0)

        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        
        # Create batched graph
        batch_size = len(h_index)
        graph_list = [graph.clone() for _ in range(batch_size)]
        graph = Batch.from_data_list(graph_list)
        
        # Adjust indices for batched graph
        node_counts = graph.ptr[:-1]
        h_index = h_index + node_counts.unsqueeze(-1).to(h_index.device)
        t_index = t_index + node_counts.unsqueeze(-1).to(t_index.device)
        
        # Get relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0])
        rel_embeds = rel_embeds.type(hidden_states.dtype)
        
        # Initialize embeddings and scores
        input_embeds, init_score = self.init_input_embeds(
            graph, hidden_states, h_index[:, 0], 
            score_text_embs, all_index, rel_embeds
        )
        
        # Run aggregation
        score = self.aggregate(graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)
        
        # Get scores for target entities
        score = score[t_index]
        return score

    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        """
        Main aggregation loop with score-based edge selection
        """
        # Initialize states
        query = rel_embeds
        boundary = input_embeds  # Keep original embeddings as boundary
        hidden = input_embeds.clone()  # Working hidden state
        score = init_score
        
        # Precompute degree statistics
        graph.degree_out = degree(graph.edge_index[0], graph.num_nodes)
        pna_degree_mean = (graph.degree_out + 1).log().mean()
        
        for i, layer in enumerate(self.layers):
            print(f"\n--- LAYER {i} START ---")
            print_stat(f"Layer {i}: graph.score (Start of Loop)", graph.score)
            
            # If this prints -62k, the corruption happened in init_input_embeds or passed init_score
            
            edge_id_subset = self.select_edges(graph, graph.score)
            
            # ... inside your aggregate loop ...
            
            edge_index_full = graph.edge_index
            edge_subset = edge_id_subset

            # 1. Build node mask induced by selected edges
            node_mask = torch.zeros(
                graph.num_nodes,
                dtype=torch.bool,
                device=edge_index_full.device,
            )

            node_mask[edge_index_full[0, edge_subset]] = True
            node_mask[edge_index_full[1, edge_subset]] = True

            # 2. Compact subgraph (TorchDrug edge_mask(compact=True))
            new_edge_index, _, edge_mask = pyg_subgraph(
                node_mask,
                edge_index_full,
                relabel_nodes=True,
                return_edge_mask=True,
            )

            # 3. Node mapping (TorchDrug: subgraph.node_id)
            node_id = torch.nonzero(node_mask, as_tuple=True)[0]

            # 4. Edge attributes
            sub_edge_attr = (
                graph.edge_attr[edge_mask]
                if graph.edge_attr is not None
                else None
            )

            subgraph = Data(
                edge_index=new_edge_index,
                edge_attr=sub_edge_attr,
                num_nodes=node_id.size(0),
            )


            # ... continue ...
            
            subgraph.num_nodes = node_id.size(0)
            
            subgraph.score = score[node_id]
            subgraph.hidden = hidden[node_id]
            subgraph.degree_out = graph.degree_out[node_id]
            subgraph.batch = graph.batch[node_id]
            subgraph.query = graph.query[graph.batch[node_id]]
            subgraph.pna_degree_out = subgraph.degree_out
            subgraph.node_id = node_id
            subgraph.pna_degree_mean = pna_degree_mean
            
            # Get batch indices for queries
            batch_idx = graph.batch[node_id]
            subgraph.query = query[batch_idx]
            
            # CRITICAL: Apply score-based gating to hidden states
            # This is the key difference from standard message passing
            node_scores = score[node_id]
            score_weights = F.sigmoid(node_scores).unsqueeze(-1)  # [num_subgraph_nodes, 1]
            
            # Gated input: multiply hidden by sigmoid(score)
            layer_input = score_weights * hidden[node_id]
            
            # Boundary for self-loops (original embeddings)
            subgraph.boundary = boundary[node_id]
            
            # Run message passing on subgraph
            hidden_update = layer(subgraph, layer_input.type(torch.float32))
            
            # Only update nodes with outgoing edges
            out_mask = subgraph.degree_out > 0
            active_node_ids = node_id[out_mask]
            
            if len(active_node_ids) == 0:
                continue
            
            # CRITICAL: Accumulate updates (not replace)
            # This allows information to flow across layers
            hidden[active_node_ids] = (
                hidden[active_node_ids] + hidden_update[out_mask]
            ).type(hidden.dtype)
            
            # Update scores based on new hidden states
            batch_idx_active = graph.batch[active_node_ids]
            new_scores = self.score(hidden[active_node_ids], query[batch_idx_active])
            score[active_node_ids] = new_scores.type(score.dtype)
        
        return score

    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index, rel_embeds):
        """Initialize node embeddings and scores"""
        if tail_embeds.dtype != head_embeds.dtype:
            tail_embeds = tail_embeds.to(head_embeds.dtype)
        
        # Initialize all nodes with zeros
        input_embeds = torch.zeros(
            graph.num_nodes,
            head_embeds.shape[-1],
            device=head_embeds.device,
            dtype=head_embeds.dtype
        )
        
        # Set tail embeddings
        input_embeds[tail_index] = tail_embeds
        
        # Set head embeddings (overrides tails if overlap)
        input_embeds[head_index] = head_embeds.to(input_embeds.dtype)
        
        # Initialize scores
        # Default score for all nodes (based on zero embedding)
        zero_hidden = torch.zeros_like(rel_embeds)
        default_scores = self.score(zero_hidden, rel_embeds)  # [batch_size]
        
        # Broadcast to all nodes in their respective graphs
        score_all = default_scores[graph.batch]  # [num_nodes]
        
        # Override with head-specific scores
        score_all[head_index] = self.score(head_embeds, rel_embeds)
        
        # IMPORTANT: Don't clamp too aggressively - allow model to learn full range
        # Only clamp to prevent numerical instability
        score_all = torch.clamp(score_all, min=-50, max=50)
        
        return input_embeds, score_all

    def score(self, hidden, rel_embeds):
        # Normalize inputs
        hidden_norm = F.normalize(hidden, p=2, dim=-1)
        rel_norm = F.normalize(rel_embeds, p=2, dim=-1)
        
        # Concatenate
        combined = torch.cat([hidden_norm, rel_norm], dim=-1)
        heuristic = self.linear(combined)
        heuristic = F.normalize(heuristic, p=2, dim=-1)
        
        # Element-wise multiplication on normalized tensors
        x = hidden_norm * heuristic
        
        # MLP produces output, then scale it
        raw_score = self.mlp(x).squeeze(-1)  # Should be bounded now
        raw_score = raw_score * 10  # Scale to [-10, 10] range
        
        return raw_score

    def select_edges(self, graph, score):
        """Select top-k nodes and their top-k' edges based on scores"""
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        # Get number of nodes per graph in batch
        num_nodes_per_graph = bincount(graph.batch, minlength=graph.num_graphs)
        
        # Compute k (number of nodes to keep per graph)
        ks = (num_nodes_per_graph.float() * node_ratio).long()
        ks = torch.clamp(ks, min=1)
        ks = torch.min(ks, num_nodes_per_graph)
        
        # Select top-k nodes per graph based on scores
        _, index = variadic_topks(score, num_nodes_per_graph, ks=ks, break_tie=self.break_tie)
        node_in = index
        
        # Create mask for edges whose source is in selected nodes
        src_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.edge_index.device)
        src_mask[node_in] = True
        edge_mask_in = src_mask[graph.edge_index[0]]
        
        # Count edges per graph among selected nodes
        edge_batch = graph.batch[graph.edge_index[0][edge_mask_in]]
        num_edges_per_graph = bincount(edge_batch, minlength=graph.num_graphs)
        
        # Compute number of edges to keep per graph
        es = (degree_ratio * ks.float() * (num_edges_per_graph.float() / num_nodes_per_graph.float().clamp(min=1))).long()
        es = torch.clamp(es, min=1)
        es = torch.min(es, num_edges_per_graph)
        
        # Select top-k' edges based on target node scores
        valid_edge_indices = torch.nonzero(edge_mask_in, as_tuple=True)[0]
        node_out = graph.edge_index[1][valid_edge_indices]
        score_edge = score[node_out]
        
        # Get top edges per graph
        _, final_indices = variadic_topks(score_edge, num_edges_per_graph, ks=es, break_tie=self.break_tie)
        
        return valid_edge_indices[final_indices]

    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        """Remove edges that directly connect query heads and tails"""
        if graph.edge_attr is None:
            raise ValueError("Graph must have edge_attr for remove_easy_edges")
        
        edge_rels = graph.edge_attr.squeeze()
        num_nodes = graph.num_nodes
        num_rels = max(edge_rels.max().item(), r_index.max().item()) + 1
        
        if self.remove_one_hop:
            # Remove all edges between h and t (any relation)
            h_ext = torch.cat([h_index, t_index], dim=0)
            t_ext = torch.cat([t_index, h_index], dim=0)
            graph_hashes = graph.edge_index[0].long() * num_nodes + graph.edge_index[1].long()
            batch_hashes = h_ext.long() * num_nodes + t_ext.long()
        else:
            # Remove only edges with specific relation
            graph_hashes = (graph.edge_index[0].long() * num_nodes + graph.edge_index[1].long()) * num_rels + edge_rels.long()
            batch_hashes = (h_index.long() * num_nodes + t_index.long()) * num_rels + r_index.long()
        
        mask_to_remove = torch.isin(graph_hashes, batch_hashes)
        final_mask = ~mask_to_remove
        
        graph.edge_index = graph.edge_index[:, final_mask]
        graph.edge_attr = graph.edge_attr[final_mask]
        
        return graph

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        """Convert negative samples to tail prediction format"""
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index