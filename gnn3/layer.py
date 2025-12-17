import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter


class PNALayer(MessagePassing):
    """
    PNA Layer reimplemented for PyTorch Geometric
    """
    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, 
                 message_func="distmult", aggregate_func="pna", layer_norm=False, 
                 activation="relu", dependent=True):
        super(PNALayer, self).__init__(aggr=None, node_dim=0)
        
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
    
    def forward(self, x, edge_index, edge_type, boundary, query, degree_out, degree_mean):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge relation types [num_edges]
            boundary: Boundary node features [num_nodes, input_dim]
            query: Query embeddings [batch_size, query_input_dim]
            degree_out: Out-degree of nodes [num_nodes, 1]
            degree_mean: Mean log degree for normalization
        """
        batch_size = query.shape[0]
        
        # Get relation embeddings
        if self.dependent:
            # Shape: [batch_size, num_relation * 2, input_dim]
            relation_input = self.relation_linear(query).view(
                batch_size, self.num_relation * 2, self.input_dim
            )
        else:
            # Shape: [num_relation * 2, input_dim] -> [batch_size, num_relation * 2, input_dim]
            relation_input = self.relation.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Perform message passing and aggregation
        out = self.propagate(
            edge_index, 
            x=x, 
            edge_type=edge_type,
            boundary=boundary,
            relation_input=relation_input,
            degree_out=degree_out,
            degree_mean=degree_mean
        )
        
        return out
    
    def message(self, x_j, edge_type, relation_input):
        """
        Compute messages from source nodes
        
        Args:
            x_j: Source node features [num_edges, input_dim]
            edge_type: Edge relation types [num_edges]
            relation_input: Relation embeddings [batch_size, num_relation * 2, input_dim]
        """
        # For batched graphs, we need to handle relation lookup properly
        # Assuming edge_type already accounts for batch offset if needed
        batch_size = relation_input.shape[0]
        
        if batch_size == 1:
            # Single graph case
            rel_emb = relation_input[0, edge_type]  # [num_edges, input_dim]
        else:
            # Multiple graphs - relation_input is flattened
            # edge_type should include batch offset: edge_type + batch_idx * num_relation
            relation_input_flat = relation_input.view(-1, self.input_dim)
            rel_emb = relation_input_flat[edge_type]
        
        # Apply message function (e.g., distmult-style multiplication)
        message = x_j * rel_emb
        return message
    
    def aggregate(self, inputs, index, boundary, degree_out, degree_mean, dim_size=None):
        """
        Aggregate messages using PNA aggregation
        
        Args:
            inputs: Messages [num_edges, input_dim]
            index: Target node indices [num_edges]
            boundary: Boundary features [num_nodes, input_dim]
            degree_out: Out-degrees [num_nodes, 1]
            degree_mean: Mean log degree
        """
        if self.aggregate_func == "sum":
            update = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
            update = update + boundary
            
        elif self.aggregate_func == "mean":
            update = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
            update = (update + boundary) / degree_out
            
        elif self.aggregate_func == "max":
            update = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
            update = torch.max(update, boundary)
            
        elif self.aggregate_func == "pna":
            # PNA uses multiple aggregators and scalers
            sum_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
            max_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
            min_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='min')
            
            # Compute mean and std
            mean = (sum_agg + boundary) / degree_out
            
            # For std, we need sum of squares
            sq_sum = scatter(inputs ** 2, index, dim=0, dim_size=dim_size, reduce='sum')
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            
            # Apply boundary to max/min
            max_agg = torch.max(max_agg, boundary)
            min_agg = torch.min(min_agg, boundary)
            
            # Stack aggregators: [num_nodes, input_dim, 4]
            features = torch.stack([mean, max_agg, min_agg, std], dim=-1)
            features = features.flatten(-2)  # [num_nodes, input_dim * 4]
            
            # Apply degree scalers
            scale = degree_out.log()
            scale = scale / degree_mean
            scales = torch.cat([
                torch.ones_like(scale), 
                scale, 
                1 / scale.clamp(min=1e-2)
            ], dim=-1)  # [num_nodes, 3]
            
            # Combine: [num_nodes, input_dim * 4, 3] -> [num_nodes, input_dim * 12]
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
            # Add original boundary as 13th feature
            update = torch.cat([update, boundary], dim=-1)  # [num_nodes, input_dim * 13]
        else:
            raise ValueError(f"Unknown aggregation function `{self.aggregate_func}`")
        
        return update
    
    def update(self, aggr_out, x):
        """
        Update node features after aggregation
        
        Args:
            aggr_out: Aggregated messages [num_nodes, feature_dim]
            x: Original node features [num_nodes, input_dim]
        """
        # Combine input and aggregated features
        output = self.linear(torch.cat([x, aggr_out], dim=-1))
        
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
            
        return output