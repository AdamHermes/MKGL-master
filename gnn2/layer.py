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
    def __init__(self, input_dim, output_dim, num_relation, query_input_dim,
                 message_func="distmult", aggregate_func="pna",
                 layer_norm=False, activation="relu", dependent=True, **kwargs):
        
        # Set aggr=None because we handle the specific PNA aggregation (mean/max/min/std) manually
        super().__init__(aggr=None, node_dim=0)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        # 1. Normalization & Activation
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # 2. Aggregation Projection (Input * 13 explained: 1 self + 4 aggr * 3 scalers)
        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        # 3. Relation Projection
        if dependent:
            # Projects query embedding to relation weights
            self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        else:
            self.relation = nn.Embedding(num_relation * 2, input_dim)

    def forward(self, graph, input):
        # Setup inputs
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr 
        
        # 1. Calculate Relation Input (The "Message" Weight)
        if self.dependent:
            # Shape: [Batch_Size, Num_Rel*2, Input_Dim]
            rel_weights = self.relation_linear(graph.query).view(-1, self.num_relation * 2, self.input_dim)
            
            # Map edges to the correct graph in the batch
            if hasattr(graph, 'batch') and graph.batch is not None:
                edge_batch_idx = graph.batch[edge_index[0]]
            else:
                edge_batch_idx = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            
            # Gather specific relation weights for each edge
            # shape: [Num_Edges, Input_Dim]
            relation_input = rel_weights[edge_batch_idx, edge_attr]
        else:
            # Independent: Simple embedding lookup
            relation_input = self.relation(edge_attr)

        # 2. Propagate
        # We pass 'input' as 'boundary' so it is available in aggregate() for the self-loop logic
        out = self.propagate(edge_index, x=input, relation_input=relation_input, 
                             boundary=input, 
                             pna_degree_out=graph.pna_degree_out,
                             pna_degree_mean=getattr(graph, "pna_degree_mean", None))
                             
        # 3. Combine (Final Projection)
        out = self.combine(input, out)
        return out

    def message(self, x_j, relation_input):
        # TorchDrug: mul="mul" -> Element-wise multiplication
        return x_j * relation_input

    def aggregate(self, inputs, index, boundary, pna_degree_out, pna_degree_mean=None, dim_size=None):
        # inputs: Messages [Num_Edges, Input_Dim]
        # boundary: Target Node Features [Num_Nodes, Input_Dim]

        # Keep scatter outputs aligned with boundary rows
        if dim_size is None:
            dim_size = boundary.size(0)

        # Compute degree directly from edge index (safest approach)
        degree_from_edges = scatter(torch.ones(index.size(0), device=index.device), 
                                    index, dim=0, dim_size=dim_size, reduce='sum')
        
        # --- A. Aggregators ---
        # 1. Sum
        sum_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
        
        # 2. Sq Sum (for Std)
        sq_sum_agg = scatter(inputs ** 2, index, dim=0, dim_size=dim_size, reduce='sum')
        
        # 3. Max & Min
        max_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
        min_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='min')
        
        # --- B. Combine with Boundary (Self-Loop Logic) ---
        # degree = degree_out + 1 (includes self)
        degree = degree_from_edges.unsqueeze(-1) + 1  # [N, 1]
        
        mean = (sum_agg + boundary) / degree
        sq_mean = (sq_sum_agg + boundary ** 2) / degree
        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
        
        max_feat = torch.max(max_agg, boundary)
        min_feat = torch.min(min_agg, boundary)
        
        # features: list of [N, D] tensors, 4 total
        features = [mean, max_feat, min_feat, std]  # Each is [N, D]
        
        # --- C. Scaling ---
        scale = degree.log()
        
        if pna_degree_mean is None:
            pna_degree_mean = scale.mean()
            
        scale = scale / pna_degree_mean
        
        # scales: [N, 3] - identity, amplification, attenuation
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        
        # --- D. Apply Scaling ---
        # For each feature [N, D], multiply by each scale [N, 1] -> get 3 scaled versions
        # Result: 4 features * 3 scales = 12 feature sets, each [N, D]
        # Final update: [N, D * 12]
        
        scaled_features = []
        for feat in features:  # feat: [N, D]
            for s in range(3):  # 3 scalers
                scaled_feat = feat * scales[:, s:s+1]  # [N, D] * [N, 1] -> [N, D]
                scaled_features.append(scaled_feat)
        
        # Concatenate all 12 scaled features: [N, D * 12]
        update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        
        return update

    def update(self, aggr_out, x):
        """
        aggr_out: [num_nodes, input_dim * 12] for PNA
        x: [num_nodes, input_dim] (original features which include boundary)
        """
        # This now correctly produces input_dim * 13 total
        output = self.linear(torch.cat([x, aggr_out], dim=-1))
        
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
            
        return output