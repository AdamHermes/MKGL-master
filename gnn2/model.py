import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    """
    GAT layer that mimics PNALayer interface.
    """
    def __init__(self, input_dim, output_dim, query_input_dim,
                 num_heads=4, dropout=0.1, layer_norm=True,
                 dependent=True, aggregate_func='mean'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.query_input_dim = query_input_dim
        self.num_heads = num_heads
        self.dependent = dependent

        # GAT convolution
        self.gat_conv = GATConv(
            input_dim,
            output_dim // num_heads,  # output per head
            heads=num_heads,
            dropout=dropout,
            concat=True,
            add_self_loops=False  # We handle graph structure externally
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

        # Query-dependent gating (if dependent=True)
        if dependent:
            self.query_gate = nn.Sequential(
                nn.Linear(query_input_dim + output_dim, output_dim),
                nn.Sigmoid()
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, node_input):
        """
        Forward pass matching PNALayer interface.

        Args:
            graph: Graph dict with 'edge_index' and optional 'query' and 'node2graph'
            node_input: Node features [num_nodes, input_dim]

        Returns:
            output: Updated node features [num_nodes, output_dim]
        """
        # Ensure edge_index is on same device as node_input
        edge_index = graph['edge_index'].to(node_input.device)

        # Apply GAT (GATConv expects edge_index on same device)
        print("node_input.shape:", node_input.shape)
        print("edge_index.max():", edge_index.max().item())
        print("edge_index.min():", edge_index.min().item())
        assert edge_index.max() < node_input.size(0), "Error: edge_index exceeds node count!"
        assert edge_index.min() >= 0, "Error: edge_index has negative index!"

        output = self.gat_conv(node_input, edge_index)

        # Layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)

        # Query-dependent gating
        if self.dependent and ('query' in graph):
            query = graph['query']
            # Move query to node_input.device and expand to node count
            query = query.to(node_input.device)
            if query.dim() == 2 and output.dim() == 2:
                if 'node2graph' in graph:
                    # node2graph must be on same device
                    node2graph = graph['node2graph'].to(node_input.device)
                    query_expanded = query[node2graph]
                else:
                    query_expanded = query.expand(output.shape[0], -1)
                gate_input = torch.cat([output, query_expanded], dim=-1)
                gate = self.query_gate(gate_input)
                output = output * gate

        output = self.dropout(output)

        return output


class GAT(nn.Module):
    """
    Multi-layer GAT that mimics PNA interface.
    """
    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super().__init__()

        self.num_layer = num_layer
        self.remove_one_hop = remove_one_hop
        self.input_dim = base_layer['input_dim']
        self.output_dim = base_layer['output_dim']

        # Create GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(
                GATLayer(
                    input_dim=base_layer['input_dim'],
                    output_dim=base_layer['output_dim'],
                    query_input_dim=base_layer.get('query_input_dim', base_layer['input_dim']),
                    num_heads=base_layer.get('num_heads', 4),
                    dropout=base_layer.get('dropout', 0.1),
                    layer_norm=base_layer.get('layer_norm', True),
                    dependent=base_layer.get('dependent', True),
                    aggregate_func=base_layer.get('aggregate_func', 'mean')
                )
            )

        # MLP for scoring (if needed)
        feature_dim = base_layer['output_dim'] + base_layer['input_dim']
        mlp_layers = []
        for i in range(num_mlp_layer - 1):
            mlp_layers.extend([
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        mlp_layers.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def aggregate(self, graph, input_embeds):
        """Aggregate through GAT layers."""
        layer_input = input_embeds
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            # Residual connection if shapes match
            if hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        return hidden

    def forward(self, graph, input_embeds):
        """Forward pass."""
        graph = self.make_undirected(graph)
        output = self.aggregate(graph, input_embeds)
        return output

    @staticmethod
    def make_undirected(graph):
        """Make graph undirected by adding inverse edges."""
        edge_index = graph['edge_index']
        edge_index = edge_index.to(edge_index.device)  # keep device
        inverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        graph = dict(graph) if isinstance(graph, dict) else graph
        graph['edge_index'] = torch.cat([edge_index, inverse_edges], dim=1)
        return graph


class ConditionedGAT(nn.Module):
    """
    Conditioned GAT that replaces ConditionedPNA.
    Maintains the same interface for drop-in replacement.
    """
    def __init__(self, base_layer, num_layer, num_mlp_layer=2,
                 node_ratio=0.1, degree_ratio=1, test_node_ratio=None,
                 test_degree_ratio=None, remove_one_hop=False,
                 break_tie=False):
        super().__init__()

        self.num_layer = num_layer
        self.remove_one_hop = remove_one_hop
        self.input_dim = base_layer['input_dim']
        self.output_dim = base_layer['output_dim']

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie

        # Infer num_relation from context (set during init_relation_embedding)
        self.num_relation = None

        # Create GAT layers
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(
                GATLayer(
                    input_dim=base_layer['input_dim'],
                    output_dim=base_layer['output_dim'],
                    query_input_dim=base_layer.get('query_input_dim', base_layer['input_dim']),
                    num_heads=base_layer.get('num_heads', 4),
                    dropout=base_layer.get('dropout', 0.1),
                    layer_norm=base_layer.get('layer_norm', True),
                    dependent=base_layer.get('dependent', True),
                )
            )

        # Relation embedding (will be initialized when num_relation is known)
        self.rel_embedding = None

        # Scoring network
        feature_dim = base_layer['output_dim'] + base_layer['input_dim']
        self.linear = nn.Linear(feature_dim, base_layer['output_dim'])

        # MLP for final scoring
        mlp_layers = []
        for i in range(num_mlp_layer - 1):
            mlp_layers.extend([
                nn.Linear(base_layer['output_dim'], base_layer['output_dim']),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        mlp_layers.append(nn.Linear(base_layer['output_dim'], 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def init_relation_embedding(self, num_relation, device=None, dtype=None):
        """Initialize relation embedding when num_relation is known.

        We create the embedding on the provided device (or module device if None).
        """
        if self.rel_embedding is None:
            self.num_relation = num_relation
            # choose device: if module already has parameters, derive device from them
            if device is None:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device('cpu')
            if dtype is None:
                try:
                    dtype = next(self.parameters()).dtype
                except StopIteration:
                    dtype = torch.float32

            # create embedding directly on correct device
            emb = nn.Embedding(num_relation * 2, self.input_dim)
            emb.to(device=device, dtype=dtype)
            self.rel_embedding = emb

    def forward(self, h_index, r_index, t_index, hidden_states,
                rel_hidden_states, graph, score_text_embs, all_index):
        """
        Main forward pass matching ConditionedPNA interface.
        """
        # Determine device from hidden_states if available, else fall back
        device = hidden_states.device if torch.is_tensor(hidden_states) else torch.device('cpu')

        # Initialize relation embedding if not done
        if self.rel_embedding is None:
            # ensure r_index is on device before reading max
            r_idx_dev = r_index.to(device)
            num_relation = int(r_idx_dev.max().item()) + 1
            self.init_relation_embedding(num_relation, device=device, dtype=hidden_states.dtype)

        # Remove easy edges during training
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        # Make undirected
        graph = self.make_undirected(graph)

        # Convert to tail prediction format
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index)

        batch_size = h_index.shape[0]

        # Create batched graph (edge index and offsets will be created on h_index.device)
        graph_dict, h_index_batch, t_index_batch = self.create_batched_graph(
            graph, h_index, t_index, batch_size)

        # Ensure r_index is on same device as rel_embedding weights (embedding expects device)
        r_index = r_index.to(self.rel_embedding.weight.device)

        rel_embeds = self.rel_embedding(r_index[:, 0]).to(hidden_states.device).type(hidden_states.dtype)

        # Make sure score_text_embs and all_index are on same device as hidden_states
        score_text_embs = score_text_embs.to(hidden_states.device)
        all_index = all_index.to(hidden_states.device)

        # Initialize input embeddings
        input_embeds, init_score = self.init_input_embeds(
            graph_dict, hidden_states, h_index_batch[:, 0],
            score_text_embs, all_index, rel_embeds, batch_size)

        # Aggregate with GAT
        score = self.aggregate(graph_dict, h_index_batch[:, 0],
                               r_index[:, 0], input_embeds,
                               rel_embeds, init_score, batch_size)

        # Get scores for target nodes (t_index_batch located on same device)
        score = score[t_index_batch]
        return score

    def aggregate(self, graph_dict, h_index, r_index, input_embeds,
                  rel_embeds, init_score, batch_size):
        """Aggregate using GAT layers with progressive edge selection."""
        device = input_embeds.device

        # Set up graph attributes (ensure correct devices)
        graph_dict['query'] = rel_embeds.to(device)
        graph_dict['node2graph'] = self.create_node2graph(
            graph_dict['num_nodes'],
            batch_size,
            device=device
        )

        hidden = input_embeds.clone()
        score = init_score.clone().to(device)

        for layer_idx, layer in enumerate(self.layers):
            # Select edges based on current scores (simplified for now)
            selected_edges = graph_dict['edge_index'].to(device)
            print("num_nodes:", graph_dict['num_nodes'])
            print("edge_index max:", graph_dict['edge_index'].max())
            print("edge_index min:", graph_dict['edge_index'].min())

            # Create subgraph dict for layer
            subgraph = {
                'edge_index': selected_edges,
                'num_nodes': graph_dict['num_nodes'],
                'query': graph_dict['query'],
                'node2graph': graph_dict['node2graph']
            }

            # Apply attention weighting
            layer_input = torch.sigmoid(score).unsqueeze(-1) * hidden

            # Apply GAT layer
            # GAT layer expects float inputs
            hidden_update = layer(subgraph, layer_input.type(torch.float32)).to(hidden.dtype)

            # Update hidden states
            hidden = hidden + hidden_update

            # Update scores: expand rel_embeds in a device-safe way
            # compute repeat_interleave count carefully (num_nodes // batch_size)
            repeat_count = graph_dict['num_nodes'] // batch_size
            query_expanded = rel_embeds.to(device).repeat_interleave(repeat_count, dim=0)

            # score_fn expects hidden and rel_embeds aligned on last dim
            score = self.score_fn(hidden, query_expanded).to(score.dtype)

        return score

    def init_input_embeds(self, graph_dict, head_embeds, head_index,
                          tail_embeds, tail_index, rel_embeds, batch_size):
        """Initialize input embeddings."""
        num_nodes = int(graph_dict['num_nodes'])
        device = rel_embeds.device

        # --- Move indices and embeddings to the SAME device ---
        head_index = head_index.to(device)
        tail_index = tail_index.to(device) if tail_index is not None else None

        head_embeds = head_embeds.to(device)
        tail_embeds = tail_embeds.to(device) if tail_embeds is not None else None

        # Init inputs
        input_embeds = torch.zeros(
            num_nodes,
            self.input_dim,
            device=device,
            dtype=rel_embeds.dtype
        )

        # --- Set tail embeddings (if exist) ---
        if tail_embeds is not None and tail_index is not None:
            input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)

        # --- Set head embeddings ---
        input_embeds[head_index] = head_embeds

        # --- Init score ---
        score = torch.zeros(num_nodes, device=device, dtype=head_embeds.dtype)

        # Expand rel embeds to match number of nodes
        repeat_count = num_nodes // batch_size


        rel_embeds_expanded = rel_embeds.to(device).repeat_interleave(repeat_count, dim=0)
        print("rel_embeds.shape:", rel_embeds.shape)
        print("rel_embeds_expanded.shape:", rel_embeds_expanded.shape)
        #print("head_index.max():", head_index.max())
        assert head_index.max() < rel_embeds_expanded.size(0)
        # Compute score only for head nodes (ensure same device)
        score[head_index] = self.score_fn(
            head_embeds,
            rel_embeds_expanded[head_index]
        ).to(score.dtype)

        return input_embeds, score

    def score_fn(self, hidden, rel_embeds):
        """Compute scores for nodes."""
        # ensure both tensors on same device
        device = hidden.device
        rel_embeds = rel_embeds.to(device)
        combined = torch.cat([hidden, rel_embeds], dim=-1)
        combined = combined.to(self.linear.weight.device) if next(self.parameters(), None) is not None else combined
        heuristic = self.linear(combined)
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score

    def create_batched_graph(self, graph, h_index, t_index, batch_size):
        edge_index = graph.edge_index
        num_nodes = edge_index.max().item() + 1  # total number of nodes in the graph
        batched_edges = []

        for i in range(batch_size):
            offset = i * num_nodes
            batched_edges.append(edge_index + offset)
        batched_edge_index = torch.cat(batched_edges, dim=1)

        offsets = torch.arange(batch_size, device=h_index.device) * num_nodes
        h_index_batch = h_index + offsets.unsqueeze(-1)
        t_index_batch = t_index + offsets.unsqueeze(-1)

        graph_dict = {
            'edge_index': batched_edge_index,
            'num_nodes': num_nodes * batch_size
        }
        print("num_nodes per graph:", num_nodes)
        print("batched_edge_index max:", batched_edge_index.max().item())
        print("batched_edge_index min:", batched_edge_index.min().item())
        print("h_index_batch max:", h_index_batch.max().item())
        print("t_index_batch max:", t_index_batch.max().item())
        return graph_dict, h_index_batch, t_index_batch


    def create_node2graph(self, num_nodes, batch_size, device):
        nodes_per_graph = num_nodes // batch_size
        # Use the specific device passed in
        node2graph = torch.arange(batch_size, device=device)
        node2graph = node2graph.repeat_interleave(nodes_per_graph)
        return node2graph

    def make_undirected(self, graph):
        """Make graph undirected."""
        edge_index = graph.edge_index
        inverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        new_graph = graph.clone()
        new_graph.edge_index = torch.cat([edge_index, inverse_edges], dim=1).to(edge_index.device)
        return new_graph

    def remove_easy_edges(self, graph, h_index, r_index, t_index):
        """Remove edges that make training too easy."""
        edge_index = graph.edge_index.to(h_index.device)

        if self.remove_one_hop:
            h_set = set(h_index.flatten().tolist())
            t_set = set(t_index.flatten().tolist())

            mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
            # iterate on CPU-safe indices but operate with tensors on correct device
            for i in range(edge_index.shape[1]):
                src = int(edge_index[0, i].item())
                dst = int(edge_index[1, i].item())
                if (src in h_set and dst in t_set) or (src in t_set and dst in h_set):
                    mask[i] = False

            graph.edge_index = edge_index[:, mask]

        return graph

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        """Convert all samples to tail prediction format."""
        # ensure indices on same device
        device = h_index.device
        h_index = h_index.to(device)
        t_index = t_index.to(device)
        r_index = r_index.to(device)

        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index,
                                  r_index + (self.num_relation if self.num_relation is not None else 0))
        return new_h_index, new_t_index, new_r_index
