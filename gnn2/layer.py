import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu', dropout=0.0):
        super(MLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # If hidden_dims is a single int, convert to list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Apply activation and dropout to all layers except the last one
            if i < len(hidden_dims) - 1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    




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
        
        # --- A. Aggregators ---
        # 1. Sum
        sum_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
        
        # 2. Sq Sum (for Std)
        sq_sum_agg = scatter(inputs ** 2, index, dim=0, dim_size=dim_size, reduce='sum')
        
        # 3. Max & Min
        # Note: We rely on 'boundary' logic below to handle empty-neighbor cases naturally
        max_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
        min_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='min')
        
        # --- B. Combine with Boundary (Self-Loop Logic) ---
        # TorchDrug logic: degree = degree_out + 1 (includes self)
        degree = pna_degree_out.unsqueeze(-1) + 1 
        
        mean = (sum_agg + boundary) / degree
        sq_mean = (sq_sum_agg + boundary ** 2) / degree
        std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
        
        max_feat = torch.max(max_agg, boundary)
        min_feat = torch.min(min_agg, boundary)
        
        # Stack Features: [N, Dim, 4]
        features = torch.cat([mean.unsqueeze(-1), max_feat.unsqueeze(-1), 
                              min_feat.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        
        # --- C. Scaling ---
        scale = degree.log()
        
        if pna_degree_mean is None:
            pna_degree_mean = scale.mean()
            
        scale = scale / pna_degree_mean
        
        # Scales: [N, 1, 3] -> (1, scale, 1/scale)
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        
        # --- D. Apply Scaling ---
        # [N, D, 4, 1] * [N, 1, 1, 3] -> [N, D, 4, 3] -> Flatten -> [N, D*12]
        update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2).flatten(-1)
        
        return update

    def combine(self, input, update):
        # input: [N, D]
        # update: [N, D*12]
        # cat: [N, D*13]
        output = self.linear(torch.cat([input, update], dim=-1))
        
        if self.layer_norm:
            output = self.layer_norm(output)
            
        if self.activation:
            output = self.activation(output)
            
        return output