import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from typing import Sequence

class MLP(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no batch normalization, activation or dropout in the last layer.
    """

    def __init__(self, input_dim, hidden_dims, short_cut=False, batch_norm=False, activation="relu", dropout=0):
        super(MLP, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short_cut = short_cut

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):
        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden
    





class PNALayer(MessagePassing):
    """
    PNA Layer converted from TorchDrug to PyTorch Geometric.
    Maintains the exact same logic and aggregation functions.
    """
    
    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, 
                 message_func="distmult", aggregate_func="pna", layer_norm=False, 
                 activation="relu", dependent=True):
        super(PNALayer, self).__init__(aggr=None)  # We'll handle aggregation manually
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
            
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # Output projection based on aggregation type
        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
            
        # Relation embeddings
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        else:
            self.relation = nn.Embedding(num_relation * 2, input_dim)

    def forward(self, graph, input):
        """
        Args:
            graph: PyG Data or Batch object with:
                - edge_index: [2, num_edges]
                - edge_attr: [num_edges] (relation types)
                - query: [batch_size, query_input_dim] or [num_nodes, query_input_dim]
                - boundary: [num_nodes, input_dim] (initial node features)
                - batch: [num_nodes] (batch assignment for each node)
                - pna_degree_out: [num_nodes] (optional, precomputed out-degrees)
            input: [num_nodes, input_dim] - current node features
        """
        batch_size = graph.query.size(0)
        
        # Flatten input if needed
        if input.dim() > 2:
            input = input.flatten(1)
        if graph.boundary.dim() > 2:
            boundary = graph.boundary.flatten(1)
        else:
            boundary = graph.boundary
            
        # Get edge information
        edge_index = graph.edge_index  # [2, num_edges]
        edge_type = graph.edge_attr if hasattr(graph, 'edge_attr') else graph.edge_type  # [num_edges]
        
        # Compute degrees
        node_out = edge_index[0]
        degree_out = getattr(graph, "pna_degree_out", None)
        if degree_out is None:
            degree_out = scatter(torch.ones_like(node_out, dtype=torch.float), 
                               node_out, dim=0, dim_size=graph.num_nodes, reduce='sum')
        degree_out = degree_out.unsqueeze(-1) + 1
        
        # Get relation embeddings
        if self.dependent:
            # Query-dependent relation embeddings
            relation_input = self.relation_linear(graph.query).view(
                batch_size, self.num_relation * 2, self.input_dim
            )
            # Expand relation embeddings for each edge based on batch assignment
            if hasattr(graph, 'batch'):
                # For batched graphs, adjust edge types by batch offset
                edge_batch = graph.batch[edge_index[0]]
                edge_type_offset = edge_type + self.num_relation * 2 * edge_batch
                relation_input = relation_input.flatten(0, 1)  # [batch_size * num_relation * 2, input_dim]
                edge_relation = relation_input[edge_type_offset]
            else:
                # Single graph case
                edge_relation = relation_input[0, edge_type]
        else:
            # Fixed relation embeddings
            if hasattr(graph, 'batch'):
                edge_batch = graph.batch[edge_index[0]]
                edge_type_offset = edge_type + self.num_relation * 2 * edge_batch
                relation_input = self.relation.weight.expand(batch_size, -1, -1).flatten(0, 1)
                edge_relation = relation_input[edge_type_offset]
            else:
                edge_relation = self.relation.weight[edge_type]
        
        # Perform message passing and aggregation
        update = self.message_and_aggregate(
            edge_index, edge_relation, input, boundary, degree_out, graph
        )
        
        # Combine with input
        output = self.combine(input, update)
        
        return output

    def message_and_aggregate(self, edge_index, edge_relation, input, boundary, degree_out, graph):
        """
        Compute messages and aggregate them using the specified aggregation function.
        This replicates the rspmm operations from TorchDrug.
        """
        node_in = edge_index[0]  # source nodes
        node_out = edge_index[1]  # target nodes
        
        # Get edge weights if available
        edge_weight = getattr(graph, 'edge_weight', None)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Type conversion to match input dtype
        edge_relation = edge_relation.type(input.dtype)
        
        # Compute messages: relation_embedding * source_node_features * edge_weight
        # This replicates: generalized_rspmm(adjacency, relation_input, input, sum="add", mul="mul")
        messages = edge_relation * input[node_in] * edge_weight.unsqueeze(-1)
        
        if self.aggregate_func == "sum":
            # Sum aggregation
            update = scatter(messages, node_out, dim=0, dim_size=graph.num_nodes, reduce='sum')
            update = update + boundary
            
        elif self.aggregate_func == "mean":
            # Mean aggregation
            update = scatter(messages, node_out, dim=0, dim_size=graph.num_nodes, reduce='sum')
            update = (update + boundary) / degree_out
            
        elif self.aggregate_func == "max":
            # Max aggregation
            update = scatter(messages, node_out, dim=0, dim_size=graph.num_nodes, reduce='max')
            # Handle nodes with no incoming edges (scatter max returns -inf)
            update = torch.where(torch.isinf(update), boundary, update)
            update = torch.max(update, boundary)
            
        elif self.aggregate_func == "pna":
            # PNA aggregation: multiple aggregators and scalers
            
            # 1. Sum aggregation
            sum_agg = scatter(messages, node_out, dim=0, dim_size=graph.num_nodes, reduce='sum')
            
            # 2. Squared sum for variance computation
            messages_sq = (edge_relation ** 2) * (input[node_in] ** 2) * (edge_weight.unsqueeze(-1) ** 2)
            sq_sum = scatter(messages_sq, node_out, dim=0, dim_size=graph.num_nodes, reduce='sum')
            
            # 3. Max aggregation
            max_agg = scatter(messages, node_out, dim=0, dim_size=graph.num_nodes, reduce='max')
            max_agg = torch.where(torch.isinf(max_agg), boundary, max_agg)
            
            # 4. Min aggregation
            min_agg = scatter(messages, node_out, dim=0, dim_size=graph.num_nodes, reduce='min')
            min_agg = torch.where(torch.isinf(min_agg), boundary, min_agg)
            
            # Compute statistics
            mean = (sum_agg + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max_val = torch.max(max_agg, boundary)
            min_val = torch.min(min_agg, boundary)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            
            # Stack features: [mean, max, min, std]
            features = torch.cat([
                mean.unsqueeze(-1), 
                max_val.unsqueeze(-1), 
                min_val.unsqueeze(-1), 
                std.unsqueeze(-1)
            ], dim=-1)
            features = features.flatten(-2)  # [num_nodes, input_dim * 4]
            
            # Degree scalers: [1, log(degree), 1/log(degree)]
            scale = degree_out.log()
            degree_mean = getattr(graph, "pna_degree_mean", scale.mean())
            scale = scale / degree_mean
            scales = torch.cat([
                torch.ones_like(scale), 
                scale, 
                1 / scale.clamp(min=1e-2)
            ], dim=-1)  # [num_nodes, 3]
            
            # Apply scalers: features [num_nodes, input_dim * 4, 1] * scales [num_nodes, 1, 3]
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
            # Result: [num_nodes, input_dim * 4 * 3] = [num_nodes, input_dim * 12]
            
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        
        return update

    def combine(self, input, update):
        """Combine input features with aggregated updates."""
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output