import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, degree
from .util import VirtualTensor, bincount, variadic_topks
from .layer import MLP 

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
            reverse_edge_index = torch.stack([graph.edge_index[1], graph.edge_index[0]], dim=0)
            reverse_edge_attr = graph.edge_attr + self.num_relation
            graph.edge_index = torch.cat([graph.edge_index, reverse_edge_index], dim=1)
            graph.edge_attr = torch.cat([graph.edge_attr, reverse_edge_attr], dim=0)
        
        # Convert to tail prediction format
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
            # Select important edges based on current scores
            edge_id_subset = self.select_edges(graph, score)
            
            if len(edge_id_subset) == 0:
                continue
            
            # Create subgraph
            sub_edge_index = graph.edge_index[:, edge_id_subset]
            sub_edge_attr = graph.edge_attr[edge_id_subset] if graph.edge_attr is not None else None
            
            # Get unique nodes in subgraph
            unique_nodes, new_edge_index = sub_edge_index.unique(return_inverse=True)
            new_edge_index = new_edge_index.view(2, -1)
            
            subgraph = Data(
                edge_index=new_edge_index,
                edge_attr=sub_edge_attr,
                num_nodes=unique_nodes.size(0)
            )
            
            # Set up subgraph attributes
            subgraph.degree_out = degree(subgraph.edge_index[0], subgraph.num_nodes)
            subgraph.pna_degree_out = subgraph.degree_out
            subgraph.pna_degree_mean = pna_degree_mean
            
            # Get batch indices for queries
            batch_idx = graph.batch[unique_nodes]
            subgraph.query = query[batch_idx]
            
            # CRITICAL: Apply score-based gating to hidden states
            # This is the key difference from standard message passing
            node_scores = score[unique_nodes]
            score_weights = F.sigmoid(node_scores).unsqueeze(-1)  # [num_subgraph_nodes, 1]
            
            # Gated input: multiply hidden by sigmoid(score)
            layer_input = score_weights * hidden[unique_nodes]
            
            # Boundary for self-loops (original embeddings)
            subgraph.boundary = boundary[unique_nodes]
            
            # Run message passing on subgraph
            hidden_update = layer(subgraph, layer_input.type(torch.float32))
            
            # Only update nodes with outgoing edges
            out_mask = subgraph.degree_out > 0
            active_node_ids = unique_nodes[out_mask]
            
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
        """Compute relevance score for nodes given query relation"""
        # Concatenate hidden state with relation embedding
        combined = torch.cat([hidden, rel_embeds], dim=-1)
        
        # Linear projection to get heuristic
        heuristic = self.linear(combined)
        
        # Element-wise multiplication (query-specific gating)
        x = hidden * heuristic
        
        # MLP to get final score
        score = self.mlp(x).squeeze(-1)
        
        return score

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