import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from typing import Sequence


class MLP(nn.Module):
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
    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        query_input_dim,
        message_func="distmult",
        aggregate_func="pna",
        layer_norm=True,
        activation="relu",
        dependent=True,
    ):
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

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        else:
            self.relation = nn.Embedding(num_relation * 2, input_dim)

    def forward(self, graph, input):
        batch_size = len(graph.query)

        if input.dim() < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input.shape}")

        input = input.flatten(1)

        boundary = graph.boundary
        if boundary.dim() < 2:
            raise ValueError(f"Boundary must be at least 2D, got shape {boundary.shape}")

        boundary = boundary.flatten(1)

        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        degree_out = graph.pna_degree_out
        if degree_out.dim() == 1:
            degree_out = degree_out.unsqueeze(-1)
        degree_out = degree_out + 1

        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(
                batch_size, self.num_relation * 2, self.input_dim
            )
        else:
            relation_input = self.relation.weight.unsqueeze(0).expand(batch_size, -1, -1)

        update = self.propagate(
            edge_index,
            x=input,
            edge_attr=edge_attr,
            boundary=boundary,
            relation_input=relation_input,
            degree_out=degree_out,
            degree_mean=graph.pna_degree_mean,
            node2graph=graph.node2graph if hasattr(graph, "node2graph") else None,
            size=(input.size(0), input.size(0)),
        )

        return update

    def message(self, x_j, edge_attr, relation_input, node2graph_j=None):
        batch_size = relation_input.shape[0]

        if batch_size == 1:
            rel_emb = relation_input[0, edge_attr]
        else:
            if node2graph_j is not None:
                batch_idx = node2graph_j
                rel_emb = relation_input[batch_idx, edge_attr]
            else:
                relation_input_flat = relation_input.view(-1, self.input_dim)
                rel_emb = relation_input_flat[edge_attr]

        rel_emb = rel_emb.type(x_j.dtype)
        message = x_j * rel_emb

        return message

    def aggregate(
        self,
        inputs,
        index,
        boundary,
        degree_out,
        degree_mean,
        dim_size=None,
        node2graph=None,
    ):
        if self.aggregate_func == "sum":
            update = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
            update = update + boundary

        elif self.aggregate_func == "mean":
            update = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
            update = (update + boundary) / degree_out

        elif self.aggregate_func == "max":
            update = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="max")
            update = torch.max(update, boundary)

        elif self.aggregate_func == "pna":
            sum_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="sum")
            sq_sum = scatter(inputs ** 2, index, dim=0, dim_size=dim_size, reduce="sum")
            max_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="max")
            min_agg = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="min")

            mean = (sum_agg + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()

            max_agg = torch.max(max_agg, boundary)
            min_agg = torch.min(min_agg, boundary)

            features = torch.cat(
                [
                    mean.unsqueeze(-1),
                    max_agg.unsqueeze(-1),
                    min_agg.unsqueeze(-1),
                    std.unsqueeze(-1),
                ],
                dim=-1,
            )

            features = features.flatten(-2)

            scale = degree_out.log()
            scale = scale / degree_mean
            scales = torch.cat(
                [
                    torch.ones_like(scale),
                    scale,
                    1 / scale.clamp(min=1e-2),
                ],
                dim=-1,
            )

            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        else:
            raise ValueError(f"Unknown aggregation function `{self.aggregate_func}`")

        return update

    def update(self, aggr_out, x):
        concat_input = torch.cat([x, aggr_out], dim=-1)
        output = self.linear(concat_input)

        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)

        return output
