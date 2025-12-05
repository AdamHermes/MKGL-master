import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, to_undirected

class SimpleMLPLayer(nn.Module):
    """Simple MLP layer for graph neural networks"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """Multi-layer perceptron"""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class PNA(nn.Module):
    """Principal Neighborhood Aggregation without TorchDrug"""
    
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()
        
        self.num_layer = num_layer
        self.num_mlp_layer = num_mlp_layer
        self.remove_one_hop = remove_one_hop
        self.layers = nn.ModuleList()
        
        # Store layer configuration
        self.input_dim = base_layer.in_dim if hasattr(base_layer, 'in_dim') else base_layer.input_dim
        self.output_dim = base_layer.out_dim if hasattr(base_layer, 'out_dim') else base_layer.output_dim
        self.num_relation = base_layer.num_relation if hasattr(base_layer, 'num_relation') else 1
        
        # Create multiple layers
        for i in range(num_layer):
            self.layers.append(self._clone_layer(base_layer))
        
        feature_dim = self.output_dim + self.input_dim
        self.mlp = MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])
    
    def _clone_layer(self, layer):
        """Clone a layer with the same configuration"""
        layer_class = layer.__class__
        return layer_class(
            input_dim=layer.in_dim if hasattr(layer, 'in_dim') else layer.input_dim,
            output_dim=layer.out_dim if hasattr(layer, 'out_dim') else layer.output_dim,
            num_relation=layer.num_relation if hasattr(layer, 'num_relation') else 1
        )
    
    def aggregate(self, graph, input_embeds):
        """Aggregate embeddings through layers"""
        layer_input = input_embeds
        
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if hasattr(self, 'short_cut') and self.short_cut:
                hidden = hidden + layer_input
            layer_input = hidden
        
        return hidden
    
    def init_input_embeds(self, graph, input_embeds, input_index):
        """Initialize input embeddings for all nodes"""
        full_embeds = torch.zeros(
            graph.num_nodes, 
            input_embeds.shape[-1], 
            device=input_embeds.device,
            dtype=input_embeds.dtype
        )
        full_embeds[input_index] = input_embeds
        return full_embeds
    
    def forward(self, graph, input_embeds, input_index):
        """Forward pass"""
        graph = graph.clone()
        
        # Validate and filter edge_index
        if graph.edge_index.shape[1] > 0:
            max_node_idx = graph.edge_index.max().item()
            if max_node_idx >= graph.num_nodes:
                valid_mask = (graph.edge_index[0] < graph.num_nodes) & (graph.edge_index[1] < graph.num_nodes)
                graph.edge_index = graph.edge_index[:, valid_mask]
                if graph.edge_attr is not None:
                    graph.edge_attr = graph.edge_attr[valid_mask]
        
        if graph.edge_index.shape[1] > 0:
            edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
            graph.edge_index = edge_index
            if edge_attr is not None:
                graph.edge_attr = edge_attr
        
        full_embeds = self.init_input_embeds(graph, input_embeds, input_index)
        output = self.aggregate(graph, full_embeds)
        return output


class PNALayer(nn.Module):
    """
    Basic PNA layer for message passing
    Handles both Node-Aligned and Edge-Aligned inputs correctly.
    """
    def __init__(self, input_dim, output_dim, query_input_dim=None, 
                 message_func='distmult', aggregate_func='pna', 
                 layer_norm=True, dependent=True, num_relation=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.query_input_dim = query_input_dim or input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.layer_norm = layer_norm
        self.dependent = dependent
        self.num_relation = num_relation
        
        self.message_transform = nn.Linear(input_dim, output_dim)
        if layer_norm:
            self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, graph, x):
        """Forward pass with intelligent shape handling"""
        edge_index = graph.edge_index
        
        # Transform features
        out = self.message_transform(x)
        
        # Determine num_nodes reliably
        num_nodes = graph.num_nodes if graph.num_nodes is not None else out.size(0)
        
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            
            # ALWAYS aggregate into a tensor sized (Num_Nodes, Dim)
            aggregated = torch.zeros(num_nodes, out.size(1), device=out.device, dtype=out.dtype)
            
            # Check if input x was Node-Aligned (N, D) or Edge-Aligned (E, D)
            if out.size(0) == edge_index.shape[1]:
                # Input was Edge-Aligned (ConditionedPNA optimization)
                msg = out
            else:
                # Input was Node-Aligned (Standard)
                msg = out[src]
                
            aggregated.index_add_(0, dst, msg)
            out = aggregated
        else:
            # If no edges, return zeros for the graph nodes
            out = torch.zeros(num_nodes, out.size(1), device=out.device, dtype=out.dtype)
        
        if self.layer_norm:
            out = self.norm(out)
        
        return out


class ConditionedPNA(PNA):
    """
    Conditioned PNA for knowledge graph completion without TorchDrug
    Includes fixes for:
    1. Num_Nodes batching logic
    2. Edge Attribute Mismatch
    3. Autocast/Dtype Mismatch (Float vs BFloat16)
    """
    
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, node_ratio=0.1, 
                 degree_ratio=1, test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, **kwargs):
        
        super().__init__(base_layer, num_layer, num_mlp_layer=num_mlp_layer, **kwargs)
        
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie
        
        feature_dim = self.output_dim + self.input_dim
        self.rel_embedding = nn.Embedding(self.num_relation * 2, self.input_dim)
        self.linear = nn.Linear(feature_dim, self.output_dim)
        self.mlp = MLP(self.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])
    
    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, 
                graph, score_text_embs, all_index):
        
        graph = graph.clone()
        
        # 1. Normalize inputs to prevent Exploding Gradients/NaN
        hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        score_text_embs = F.normalize(score_text_embs, p=2, dim=-1)
        
        # 2. Correct Num_Nodes Calculation
        num_nodes = score_text_embs.size(0)
        
        # Expand num_nodes if edges point to non-existent nodes (Safety)
        if graph.edge_index.numel() > 0:
            max_edge = graph.edge_index.max().item()
            if max_edge >= num_nodes:
                num_nodes = max_edge + 1
        graph.num_nodes = num_nodes

        # 3. Check for Edge Attribute Mismatch
        if graph.edge_attr is not None:
            if graph.edge_index.size(1) != graph.edge_attr.size(0):
                limit = min(graph.edge_index.size(1), graph.edge_attr.size(0))
                graph.edge_index = graph.edge_index[:, :limit]
                graph.edge_attr = graph.edge_attr[:limit]

        # 4. Remove Easy Edges (Training only)
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        
        # 5. Robust Graph Sanitization
        if graph.edge_index.numel() > 0:
            row, col = graph.edge_index
            mask = (row >= 0) & (row < num_nodes) & (col >= 0) & (col < num_nodes)
            if not mask.all():
                graph.edge_index = graph.edge_index[:, mask]
                if graph.edge_attr is not None:
                    graph.edge_attr = graph.edge_attr[mask]

        # 6. Convert to undirected (Safe)
        if graph.edge_index.numel() > 0:
            edge_index, edge_attr = to_undirected(
                graph.edge_index, 
                graph.edge_attr, 
                num_nodes=num_nodes
            )
            graph.edge_index = edge_index
            if edge_attr is not None:
                graph.edge_attr = edge_attr
        
        # 7. Resample negatives to tail
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        
        # 8. Handle batch processing
        batch_size = len(h_index)
        graph = self._repeat_graph(graph, batch_size) 
        
        # Compute offsets for batched indices
        if not hasattr(graph, 'num_nodes_list'):
            graph.num_nodes_list = [num_nodes] * batch_size

        num_nodes_cumsum = torch.cat([
            torch.tensor([0], device=h_index.device),
            torch.cumsum(torch.tensor(graph.num_nodes_list, device=h_index.device), dim=0)
        ])
        offset = num_nodes_cumsum[:-1]
        
        h_index = h_index + offset.unsqueeze(-1).to(h_index.device)
        t_index = t_index + offset.unsqueeze(-1).to(t_index.device)
        
        # 9. Get relation embeddings
        r_index_clamped = r_index[:, 0].clamp(0, self.num_relation * 2 - 1)
        rel_embeds = self.rel_embedding(r_index_clamped).type(hidden_states.dtype)
        
        # 10. Initialize embeddings
        input_embeds, init_score = self.init_input_embeds(
            graph, hidden_states, h_index[:, 0], score_text_embs, all_index, rel_embeds
        )
        
        # 11. Aggregate
        score = self.aggregate(graph, h_index[:, 0], r_index_clamped, input_embeds, rel_embeds, init_score)
        
        # 12. Return gathered scores
        flat_t_index = t_index.view(-1)
        
        max_valid_idx = score.size(0) - 1
        if flat_t_index.max() > max_valid_idx:
            flat_t_index = flat_t_index.clamp(max=max_valid_idx)

        flat_scores = score[flat_t_index]
        final_score = flat_scores.view(batch_size, -1)
        
        return final_score
    
    def _repeat_graph(self, graph, batch_size):
        """Repeat graph structure for batch processing"""
        edge_index_list = []
        edge_attr_list = []
        x_list = []
        
        for i in range(batch_size):
            offset = i * graph.num_nodes
            edge_index_list.append(graph.edge_index + offset)
            if graph.edge_attr is not None:
                edge_attr_list.append(graph.edge_attr)
            x_list.append(graph.x)
        
        batched_graph = graph.clone()
        batched_graph.edge_index = torch.cat(edge_index_list, dim=1)
        if edge_attr_list:
            batched_graph.edge_attr = torch.cat(edge_attr_list, dim=0)
        batched_graph.x = torch.cat(x_list, dim=0)
        batched_graph.num_nodes_list = [graph.num_nodes] * batch_size
        
        # Fix: Update total nodes for the batched graph
        batched_graph.num_nodes = graph.num_nodes * batch_size
        
        return batched_graph
    
    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        """Aggregate with edge selection"""
        query = rel_embeds
        boundary = input_embeds.clone()
        hidden = boundary.clone()
        score = init_score.clone()
        
        pna_degree_mean = (degree(graph.edge_index[1], num_nodes=graph.num_nodes) + 1).log().mean()
        
        for layer in self.layers:
            # Select important edges
            edge_index = self.select_edges(graph, score)
            
            # Create subgraph
            subgraph = self._create_subgraph(graph, edge_index)
            subgraph.pna_degree_mean = pna_degree_mean
            
            # Layer forward
            layer_input = torch.sigmoid(score[edge_index[0]]).unsqueeze(-1) * hidden[edge_index[0]]
            
            hidden_update = layer(subgraph, layer_input.type(torch.float32))
            
            # Update hidden states for nodes with outgoing edges
            out_mask = degree(edge_index[0], num_nodes=graph.num_nodes) > 0
            node_out = torch.where(out_mask)[0]
            
            # Node-Aligned addition
            hidden[node_out] = (hidden[node_out] + hidden_update[node_out]).type(hidden.dtype)
            
            # Update scores
            index_batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=h_index.device)
            score[node_out] = self.score(hidden[node_out], query[index_batch[node_out]]).type(score.dtype)
        
        return score
    
    def _create_subgraph(self, graph, edge_index):
        """Create subgraph from edge indices"""
        subgraph = graph.clone()
        subgraph.edge_index = edge_index
        return subgraph
    
    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index, rel_embeds):
        """Initialize input embeddings with head and tail"""
        input_embeds = torch.zeros(
            graph.num_nodes, 
            rel_embeds.shape[1], 
            device=rel_embeds.device, 
            dtype=rel_embeds.dtype
        )
        
        input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        input_embeds[head_index] = head_embeds
        
        # Initialize scores
        score = torch.zeros(graph.num_nodes, device=rel_embeds.device, dtype=rel_embeds.dtype)
        
        # --- FIX: Cast the calculation result to the storage dtype ---
        # This handles the case where autocast produces BFloat16 but score is Float32
        calculated_score = self.score(head_embeds, rel_embeds)
        score[head_index] = calculated_score.to(dtype=score.dtype)
        
        return input_embeds, score
    
    def score(self, hidden, rel_embeds):
        """Compute score from hidden and relation embeddings"""
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score
    
    def select_edges(self, graph, score):
        """Select important edges based on scores"""
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        num_nodes = graph.num_nodes
        ks = max(1, int(node_ratio * num_nodes))
        es = max(1, int(degree_ratio * ks * graph.num_edges / num_nodes))
        
        # Get nodes with highest scores
        top_scores, top_nodes = torch.topk(score, min(ks, len(score)))
        
        # Get neighbors of top nodes
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=score.device)
        node_mask[top_nodes] = True
        
        edge_mask = node_mask[graph.edge_index[0]]
        selected_edges = graph.edge_index[:, edge_mask]
        
        # Further select top edges by score
        if selected_edges.shape[1] > es:
            edge_scores = score[selected_edges[1]]
            _, top_edge_idx = torch.topk(edge_scores, min(es, edge_scores.shape[0]))
            selected_edges = selected_edges[:, top_edge_idx]
        
        return selected_edges
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        """Remove edges that are too easy (ground truth edges)"""
        if self.remove_one_hop:
            # Remove edges between h and t in both directions
            h_ext = torch.cat([h_index[:, 0], t_index[:, 0]])
            t_ext = torch.cat([t_index[:, 0], h_index[:, 0]])
        else:
            h_ext = h_index[:, 0]
            t_ext = t_index[:, 0]
            r_ext = r_index[:, 0]
        
        # Create mask for edges to remove
        edge_mask = torch.ones(graph.num_edges, dtype=torch.bool, device=graph.edge_index.device)
        
        for h, t in zip(h_ext, t_ext):
            # Find matching edges
            matching = (graph.edge_index[0] == h) & (graph.edge_index[1] == t)
            edge_mask[matching] = False
        
        # Filter graph
        new_edge_index = graph.edge_index[:, edge_mask]
        new_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
        
        graph.edge_index = new_edge_index
        graph.edge_attr = new_edge_attr
        
        return graph
    
    def negative_sample_to_tail(self, h_index, t_index, r_index):
        """Resample negatives to tail"""
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index