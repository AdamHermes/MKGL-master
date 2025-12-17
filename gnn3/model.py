import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add
import copy

from .layer import PNALayer
from .util import (
    VirtualTensor, RepeatGraph, bincount, variadic_topks
)


class PNA(nn.Module):
    """
    Basic PNA model for knowledge graphs
    """
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()
        
        self.num_relation = base_layer.num_relation
        self.remove_one_hop = remove_one_hop
        self.short_cut = getattr(base_layer, 'short_cut', False)
        
        # Clone base layer for each layer
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(copy.deepcopy(base_layer))
        
        # MLP for final scoring
        feature_dim = base_layer.output_dim + base_layer.input_dim
        mlp_layers = []
        for i in range(num_mlp_layer - 1):
            mlp_layers.extend([
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU()
            ])
        mlp_layers.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def make_undirected(self, data):
        """Convert graph to undirected by adding inverse edges"""
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None
        
        # Add inverse edges
        edge_index_inv = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_inv], dim=1)
        
        if edge_type is not None:
            # Inverse relations have offset of num_relation
            edge_type_inv = edge_type + self.num_relation
            edge_type = torch.cat([edge_type, edge_type_inv], dim=0)
            data.edge_type = edge_type
        
        data.edge_index = edge_index
        data.num_relation = self.num_relation * 2
        return data
    
    def init_input_embeds(self, graph, input_embeds, input_index):
        """Initialize node embeddings with input embeddings at specified indices"""
        node_embeds = torch.zeros(
            graph.num_nodes, input_embeds.shape[-1], 
            device=input_embeds.device, dtype=input_embeds.dtype
        )
        node_embeds[input_index] = input_embeds
        return node_embeds
    
    def aggregate(self, graph, input_embeds, query):
        """
        Run message passing layers
        
        Args:
            graph: PyG Data object
            input_embeds: Initial node embeddings [num_nodes, input_dim]
            query: Query embeddings [batch_size, query_input_dim]
        """
        # Compute degree statistics
        degree_out = torch.bincount(
            graph.edge_index[0], 
            minlength=graph.num_nodes
        ).float().unsqueeze(-1) + 1
        degree_mean = degree_out.log().mean()
        
        layer_input = input_embeds
        for layer in self.layers:
            hidden = layer(
                x=layer_input,
                edge_index=graph.edge_index,
                edge_type=graph.edge_type,
                boundary=input_embeds,
                query=query,
                degree_out=degree_out,
                degree_mean=degree_mean
            )
            if self.short_cut:
                hidden = hidden + layer_input
            layer_input = hidden
        
        return hidden
    
    def forward(self, graph, input_embeds, input_index, query):
        """
        Forward pass
        
        Args:
            graph: PyG Data object
            input_embeds: Input embeddings [num_inputs, input_dim]
            input_index: Indices where to place input embeddings [num_inputs]
            query: Query embeddings [batch_size, query_input_dim]
        """
        graph = self.make_undirected(graph)
        node_embeds = self.init_input_embeds(graph, input_embeds, input_index)
        output = self.aggregate(graph, node_embeds, query)
        return output


class ConditionedPNA(nn.Module):
    """
    Conditioned PNA with selective edge sampling for knowledge graph reasoning
    """
    def __init__(self, base_layer, num_layer, num_mlp_layer=2,
                 node_ratio=0.1, degree_ratio=1.0,
                 test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, remove_one_hop=False):
        super(ConditionedPNA, self).__init__()
        
        self.num_relation = base_layer.num_relation
        self.num_layer = num_layer
        self.remove_one_hop = remove_one_hop
        
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie
        
        # Relation embeddings for queries
        self.rel_embedding = nn.Embedding(base_layer.num_relation * 2, base_layer.input_dim)
        
        # Clone base layer for each layer
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(copy.deepcopy(base_layer))
        
        # Scoring network
        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        
        mlp_layers = []
        for i in range(num_mlp_layer - 1):
            mlp_layers.extend([
                nn.Linear(base_layer.output_dim, feature_dim),
                nn.ReLU()
            ])
        mlp_layers.append(nn.Linear(feature_dim if num_mlp_layer > 1 else base_layer.output_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def make_undirected(self, data):
        """Convert graph to undirected by adding inverse edges"""
        edge_index = data.edge_index
        edge_type = data.edge_type if hasattr(data, 'edge_type') else None
        
        # Add inverse edges
        edge_index_inv = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_inv], dim=1)
        
        if edge_type is not None:
            edge_type_inv = edge_type + self.num_relation
            edge_type = torch.cat([edge_type, edge_type_inv], dim=0)
            data.edge_type = edge_type
        
        data.edge_index = edge_index
        return data
    
    def forward(self, h_index, r_index, t_index, hidden_states, 
                rel_hidden_states, graph, score_text_embs, all_index):
        """
        Forward pass for link prediction
        
        Args:
            h_index: Head entity indices [batch_size, num_negative + 1]
            r_index: Relation indices [batch_size, num_negative + 1]
            t_index: Tail entity indices [batch_size, num_negative + 1]
            hidden_states: Entity embeddings [num_entities, hidden_dim]
            rel_hidden_states: Relation embeddings (unused in this version)
            graph: PyG Data object representing the KG
            score_text_embs: Text embeddings for scoring [num_entities, text_dim]
            all_index: All entity indices
        """
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, r_index, t_index)
        
        graph = self.make_undirected(graph)
        
        # Convert negative samples to tail prediction format
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index
        )
        
        batch_size = len(h_index)
        
        # Repeat graph for batch processing
        graph = RepeatGraph(graph, batch_size)
        
        # Adjust indices for repeated graph
        offset = torch.arange(
            batch_size, device=h_index.device
        ) * graph.num_nodes_per_graph
        h_index = h_index + offset.unsqueeze(-1)
        t_index = t_index + offset.unsqueeze(-1)
        
        # Verify all heads and relations are the same within each sample
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        
        # Get relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0])
        rel_embeds = rel_embeds.type(hidden_states.dtype)
        
        # Initialize input embeddings
        input_embeds, init_score = self.init_input_embeds(
            graph, hidden_states, h_index[:, 0], 
            score_text_embs, all_index, rel_embeds
        )
        
        # Run conditioned aggregation
        score = self.aggregate(
            graph, h_index[:, 0], r_index[:, 0], 
            input_embeds, rel_embeds, init_score
        )
        
        # Get scores for tail entities
        score = score[t_index]
        return score
    
    def init_input_embeds(self, graph, head_embeds, head_index, 
                         tail_embeds, tail_index, rel_embeds):
        """
        Initialize node embeddings and scores
        
        Args:
            graph: RepeatGraph object
            head_embeds: Head entity embeddings [num_entities, dim]
            head_index: Head indices in repeated graph [batch_size]
            tail_embeds: Tail entity embeddings [num_entities, dim]
            tail_index: Tail indices [num_entities]
            rel_embeds: Relation embeddings [batch_size, dim]
        """
        # Create virtual tensor for node embeddings
        input_embeds = VirtualTensor.zeros(
            graph.num_nodes, rel_embeds.shape[1],
            device=rel_embeds.device, dtype=rel_embeds.dtype
        )
        
        # Set tail embeddings
        input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        # Set head embeddings
        input_embeds[head_index] = head_embeds
        
        # Initialize scores (zero for all, high for heads)
        zero_score = self.score(
            torch.zeros_like(rel_embeds), rel_embeds
        )
        score = VirtualTensor.gather(zero_score, graph.batch)
        score[head_index] = self.score(head_embeds, rel_embeds)
        
        return input_embeds, score
    
    def score(self, hidden, rel_embeds):
        """
        Compute relevance score for nodes given query relation
        
        Args:
            hidden: Node features [num_nodes, hidden_dim]
            rel_embeds: Relation embeddings [num_nodes, rel_dim]
        """
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score
    
    def aggregate(self, graph, h_index, r_index, input_embeds, 
                 rel_embeds, init_score):
        """
        Conditioned message passing with edge selection
        
        Args:
            graph: RepeatGraph object
            h_index: Head indices [batch_size]
            r_index: Relation indices [batch_size]
            input_embeds: Initial node embeddings (VirtualTensor)
            rel_embeds: Relation embeddings [batch_size, dim]
            init_score: Initial node scores (VirtualTensor)
        """
        # Initialize node states
        boundary = input_embeds
        hidden = boundary.clone()
        score = init_score
        
        # Compute degree statistics
        degree_out = graph.degree_out().float().unsqueeze(-1) + 1
        degree_mean = degree_out.log().mean()
        
        # Message passing layers
        for layer in self.layers:
            # Select top-k nodes and their top-k' edges
            edge_indices = self.select_edges(graph, score)
            
            # Extract subgraph
            subgraph, node_map, node_indices = graph.subgraph(edge_indices)
            
            # Get features for subgraph nodes
            layer_input = F.sigmoid(score[node_indices]).unsqueeze(-1) * hidden[node_indices]
            boundary_sub = boundary[node_indices]
            
            # Get query for each node in subgraph
            batch_idx = graph.batch[node_indices]
            query_sub = rel_embeds[batch_idx].unsqueeze(0)  # Add batch dim for layer
            
            # Compute degrees for subgraph
            degree_out_sub = torch.bincount(
                subgraph.edge_index[0],
                minlength=subgraph.num_nodes
            ).float().unsqueeze(-1) + 1
            
            # Run layer on subgraph
            hidden_update = layer(
                x=layer_input,
                edge_index=subgraph.edge_index,
                edge_type=subgraph.edge_type,
                boundary=boundary_sub,
                query=query_sub,
                degree_out=degree_out_sub,
                degree_mean=degree_mean
            )
            
            # Update only nodes with outgoing edges in subgraph
            out_mask = degree_out_sub[:, 0] > 0
            node_out = node_indices[out_mask]
            
            # Accumulate updates
            hidden[node_out] = (hidden[node_out] + hidden_update[out_mask]).type(
                hidden[node_out].dtype
            )
            
            # Update scores for active nodes
            batch_idx = graph.batch[node_out]
            score[node_out] = self.score(
                hidden[node_out], rel_embeds[batch_idx]
            ).type(score[node_out].dtype)
        
        return score
    
    def select_edges(self, graph, score):
        """
        Select top-k nodes and their top-k' edges based on scores
        
        Args:
            graph: RepeatGraph object
            score: Node scores (VirtualTensor or Tensor)
        """
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        
        batch_size = graph.repeats
        
        # Compute k values
        ks = (node_ratio * graph.num_nodes_per_graph * torch.ones(
            batch_size, device=graph.device
        )).long()
        es = (degree_ratio * ks * graph.num_edges_per_graph / 
              graph.num_nodes_per_graph).long()
        
        # Get nodes with non-default scores
        if isinstance(score, VirtualTensor):
            node_in = score.keys
        else:
            node_in = torch.arange(len(score), device=score.device)
        
        # Count nodes per batch
        batch = graph.batch[node_in]
        num_nodes = bincount(batch, minlength=batch_size)
        ks = torch.min(ks, num_nodes)
        
        # Select top-k nodes per batch
        score_in = score[node_in]
        _, index = variadic_topks(
            score_in, num_nodes, ks=ks, 
            largest=True, break_tie=self.break_tie
        )
        node_in = node_in[index]
        num_nodes = ks
        
        # Get neighbors of selected nodes
        num_neighbors = graph.num_neighbors(node_in)
        num_edges = scatter_add(
            num_neighbors, 
            graph.batch[node_in], 
            dim_size=batch_size
        )
        es = torch.min(es, num_edges)
        
        # Process in chunks to avoid memory issues
        num_edge_mean = num_edges.float().mean().clamp(min=1)
        chunk_size = max(int(1e7 / num_edge_mean), 1)
        
        # Split into chunks
        num_nodes_chunks = num_nodes.split(chunk_size)
        num_edges_chunks = num_edges.split(chunk_size)
        es_chunks = es.split(chunk_size)
        
        num_chunk_nodes = [nn.sum().item() for nn in num_nodes_chunks]
        node_in_chunks = node_in.split(num_chunk_nodes)
        
        edge_indexes = []
        for node_chunk, num_node, num_edge, e in zip(
            node_in_chunks, num_nodes_chunks, num_edges_chunks, es_chunks
        ):
            # Get edges and neighbors
            edge_index, node_out = graph.neighbors(node_chunk)
            score_edge = score[node_out]
            
            # Select top-k' edges per batch in chunk
            _, index = variadic_topks(
                score_edge, num_edge, ks=e,
                largest=True, break_tie=self.break_tie
            )
            edge_index = edge_index[index]
            edge_indexes.append(edge_index)
        
        edge_index = torch.cat(edge_indexes)
        return edge_index
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        """
        Remove easy edges during training (edges that directly connect h and t)
        
        Args:
            graph: PyG Data object
            h_index: Head indices [batch_size, num_negative + 1]
            t_index: Tail indices [batch_size, num_negative + 1]
            r_index: Relation indices [batch_size, num_negative + 1]
        """
        if self.remove_one_hop:
            # Remove all edges between h and t (any relation)
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
        else:
            # Remove only edges with specific relation
            h_index_ext = h_index
            t_index_ext = t_index
        
        # Flatten for pattern matching
        h_flat = h_index_ext.flatten()
        t_flat = t_index_ext.flatten()
        
        # Find edges to remove
        edge_index = graph.edge_index
        edge_type = graph.edge_type if hasattr(graph, 'edge_type') else None
        
        # Create mask for edges to keep
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        
        for h, t in zip(h_flat, t_flat):
            mask = (edge_index[0] == h) & (edge_index[1] == t)
            if not self.remove_one_hop and edge_type is not None:
                # Also match relation
                r = r_index.flatten()[0]  # Assuming same relation for batch
                mask = mask & (edge_type == r)
            edge_mask = edge_mask & ~mask
        
        # Create new graph with filtered edges
        new_graph = Data(
            edge_index=edge_index[:, edge_mask],
            num_nodes=graph.num_nodes
        )
        if edge_type is not None:
            new_graph.edge_type = edge_type[edge_mask]
        if hasattr(graph, 'num_relation'):
            new_graph.num_relation = graph.num_relation
        
        return new_graph
    
    def negative_sample_to_tail(self, h_index, t_index, r_index):
        """
        Convert negative samples to tail prediction format
        Handles both head and tail negative samples
        
        Args:
            h_index: Head indices [batch_size, num_negative + 1]
            t_index: Tail indices [batch_size, num_negative + 1]
            r_index: Relation indices [batch_size, num_negative + 1]
        """
        # Detect if negatives are tails (heads are all same) or heads (tails are same)
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        
        # Swap h and t for head negatives, and use inverse relation
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        
        return new_h_index, new_t_index, new_r_index