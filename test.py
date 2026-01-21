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
    PNA Layer for PyTorch Geometric (converted from TorchDrug)
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
    
    def forward(self, graph, input):
        """
        Forward pass compatible with TorchDrug-style graph objects
        """
        print(f"\n=== PNALayer.forward START ===")
        print(f"Input shape: {input.shape}, dtype: {input.dtype}")
        print(f"Graph num_nodes: {graph.num_nodes}, num_edges: {graph.edge_index.size(1)}")
        
        batch_size = len(graph.query)
        print(f"Batch size: {batch_size}, query shape: {graph.query.shape}")
        
        # POTENTIAL ERROR #1: flatten(1) assumes input is at least 2D
        if input.dim() < 2:
            print(f"ERROR: Input has unexpected dimensions: {input.shape}")
            raise ValueError(f"Input must be at least 2D, got shape {input.shape}")
        
        input = input.flatten(1)
        print(f"After flatten: input shape = {input.shape}")
        
        # POTENTIAL ERROR #2: boundary might not be 2D
        boundary = graph.boundary
        print(f"Boundary shape before flatten: {boundary.shape}, dtype: {boundary.dtype}")
        if boundary.dim() < 2:
            print(f"ERROR: Boundary has unexpected dimensions: {boundary.shape}")
            raise ValueError(f"Boundary must be at least 2D, got shape {boundary.shape}")
        
        boundary = boundary.flatten(1)
        print(f"After flatten: boundary shape = {boundary.shape}")
        
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        print(f"Edge type: min={edge_attr.min()}, max={edge_attr.max()}, shape={edge_attr.shape}")
        
        degree_out = graph.pna_degree_out
        if degree_out.dim() == 1:
            degree_out = degree_out.unsqueeze(-1)
        degree_out = degree_out + 1
        print(f"Degree out shape: {degree_out.shape}")
        
        # Get relation embeddings
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(
                batch_size, self.num_relation * 2, self.input_dim
            )
            print(f"Relation input (dependent) shape: {relation_input.shape}")
        else:
            relation_input = self.relation.weight.unsqueeze(0).expand(batch_size, -1, -1)
            print(f"Relation input (independent) shape: {relation_input.shape}")
        
        # POTENTIAL ERROR #3: Check if relation_input dimensions match expectations
        expected_rel_dim = batch_size * self.num_relation * 2
        print(f"Expected total relation embeddings: {expected_rel_dim}")
        
        # Perform message passing and aggregation
        print(f"Calling propagate with:")
        print(f"  x shape: {input.shape}")
        print(f"  edge_index shape: {edge_index.shape}")
        print(f"  edge_attr shape: {edge_attr.shape}")
        print(f"  boundary shape: {boundary.shape}")
        print(f"  relation_input shape: {relation_input.shape}")
        
        update = self.propagate(
            edge_index, 
            x=input,
            edge_attr=edge_attr,
            boundary=boundary,
            relation_input=relation_input,
            degree_out=degree_out,
            degree_mean=graph.pna_degree_mean,
            node2graph=graph.node2graph if hasattr(graph, 'node2graph') else None,
            size=(input.size(0), input.size(0))
        )
        
        print(f"After propagate: update shape = {update.shape}")
        
        # POTENTIAL ERROR #4: This reshaping might be incorrect
        # Handle batched graphs
        if hasattr(graph, 'node2graph') and batch_size > 1:
            print(f"Reshaping for batched graphs: {len(update)} nodes, {batch_size} batches")
            update = update.view(len(update), batch_size, -1)
            print(f"After view: update shape = {update.shape}")
        
        print(f"=== PNALayer.forward END ===\n")
        return update
    
    def message(self, x_j, edge_attr, relation_input, node2graph_j=None):
        """
        Compute messages from source nodes
        """
        print(f"\n--- PNALayer.message ---")
        print(f"x_j shape: {x_j.shape}")
        print(f"edge_attr shape: {edge_attr.shape}, min: {edge_attr.min()}, max: {edge_attr.max()}")
        print(f"relation_input shape: {relation_input.shape}")
        
        batch_size = relation_input.shape[0]
        
        if batch_size == 1:
            # Single graph case
            print(f"Single graph case")
            # POTENTIAL ERROR #5: edge_attr might be out of bounds
            if edge_attr.max() >= relation_input.size(1):
                print(f"ERROR: edge_attr max ({edge_attr.max()}) >= relation_input size ({relation_input.size(1)})")
            rel_emb = relation_input[0, edge_attr]
        else:
            # Multiple graphs case
            print(f"Multiple graphs case, batch_size={batch_size}")
            if node2graph_j is not None:
                print(f"Using node2graph_j: shape={node2graph_j.shape}, min={node2graph_j.min()}, max={node2graph_j.max()}")
                # POTENTIAL ERROR #6: This indexing might be wrong
                batch_idx = node2graph_j
                # Check bounds
                max_batch_idx = batch_idx.max()
                if max_batch_idx >= batch_size:
                    print(f"ERROR: batch_idx max ({max_batch_idx}) >= batch_size ({batch_size})")
                
                max_edge_attr = edge_attr.max()
                if max_edge_attr >= relation_input.size(1):
                    print(f"ERROR: edge_attr max ({max_edge_attr}) >= num_relations ({relation_input.size(1)})")
                
                # This is the correct indexing for batched graphs
                rel_emb = relation_input[batch_idx, edge_attr]
                print(f"rel_emb shape after indexing: {rel_emb.shape}")
            else:
                print(f"No node2graph_j, using flat indexing")
                relation_input_flat = relation_input.view(-1, self.input_dim)
                rel_emb = relation_input_flat[edge_attr]
        
        print(f"rel_emb shape: {rel_emb.shape}")
        
        # Apply message function
        rel_emb = rel_emb.type(x_j.dtype)
        message = x_j * rel_emb
        print(f"message shape: {message.shape}")
        print(f"---")
        
        return message
    
    def aggregate(self, inputs, index, boundary, degree_out, degree_mean, 
                  dim_size=None, node2graph=None):
        """
        Aggregate messages using specified aggregation function
        """
        print(f"\n--- PNALayer.aggregate ---")
        print(f"aggregate_func: {self.aggregate_func}")
        print(f"inputs shape: {inputs.shape}")
        print(f"index shape: {index.shape}, min: {index.min()}, max: {index.max()}")
        print(f"boundary shape: {boundary.shape}")
        print(f"degree_out shape: {degree_out.shape}")
        print(f"dim_size: {dim_size}")
        
        # POTENTIAL ERROR #7: dim_size might not match boundary size
        if dim_size is not None and dim_size != boundary.size(0):
            print(f"WARNING: dim_size ({dim_size}) != boundary size ({boundary.size(0)})")
        
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
            print(f"PNA aggregation...")
            # PNA uses multiple aggregators and scalers
            sum_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
            sq_sum = scatter(inputs ** 2, index, dim=0, dim_size=dim_size, reduce='sum')
            max_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
            min_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='min')
            
            print(f"Aggregators computed: sum={sum_agg.shape}, max={max_agg.shape}")
            
            # Compute mean and std
            mean = (sum_agg + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            
            # Apply boundary to max/min
            max_agg = torch.max(max_agg, boundary)
            min_agg = torch.min(min_agg, boundary)
            
            # Stack aggregators and flatten
            features = torch.cat([
                mean.unsqueeze(-1), 
                max_agg.unsqueeze(-1), 
                min_agg.unsqueeze(-1), 
                std.unsqueeze(-1)
            ], dim=-1)
            print(f"Features after stacking: {features.shape}")
            
            features = features.flatten(-2)  # [num_nodes, input_dim * 4]
            print(f"Features after flatten: {features.shape}")
            
            # POTENTIAL ERROR #8: Check dimensions
            if features.size(1) != self.input_dim * 4:
                print(f"ERROR: Features dim ({features.size(1)}) != input_dim * 4 ({self.input_dim * 4})")
            
            # Apply degree scalers
            scale = degree_out.log()
            scale = scale / degree_mean
            scales = torch.cat([
                torch.ones_like(scale), 
                scale, 
                1 / scale.clamp(min=1e-2)
            ], dim=-1)  # [num_nodes, 3]
            
            print(f"Scales shape: {scales.shape}")
            
            # Combine: [num_nodes, input_dim * 4, 3] -> [num_nodes, input_dim * 12]
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
            print(f"Update after scaling: {update.shape}")
            
            # POTENTIAL ERROR #9: Check final dimension
            if update.size(1) != self.input_dim * 12:
                print(f"ERROR: Update dim ({update.size(1)}) != input_dim * 12 ({self.input_dim * 12})")
        else:
            raise ValueError(f"Unknown aggregation function `{self.aggregate_func}`")
        
        print(f"Final update shape: {update.shape}")
        print(f"---")
        
        return update
    
    def update(self, aggr_out, x):
        """
        Combine input and aggregated features (analogous to TorchDrug's combine)
        """
        print(f"\n--- PNALayer.update (combine) ---")
        print(f"aggr_out shape: {aggr_out.shape}")
        print(f"x shape: {x.shape}")
        
        # POTENTIAL ERROR #10: Dimension mismatch in concatenation
        concat_input = torch.cat([x, aggr_out], dim=-1)
        print(f"After concat: {concat_input.shape}")
        
        expected_dim = self.input_dim * 13 if self.aggregate_func == "pna" else self.input_dim * 2
        if concat_input.size(-1) != expected_dim:
            print(f"ERROR: Concat dim ({concat_input.size(-1)}) != expected ({expected_dim})")
            print(f"  x contributes: {x.size(-1)}")
            print(f"  aggr_out contributes: {aggr_out.size(-1)}")
        
        output = self.linear(concat_input)
        print(f"After linear: {output.shape}")
        
        if self.layer_norm:
            output = self.layer_norm(output)
            print(f"After layer_norm: {output.shape}")
        if self.activation:
            output = self.activation(output)
            print(f"After activation: {output.shape}")
        
        print(f"Final output shape: {output.shape}")
        print(f"---\n")
            
        return output