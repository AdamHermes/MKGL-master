import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add


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
            graph: Graph object with edge_index and query attributes
            node_input: Node features [num_nodes, input_dim]
            
        Returns:
            output: Updated node features [num_nodes, output_dim]
        """
        edge_index = graph['edge_index']
        
        # Apply GAT
        output = self.gat_conv(node_input, edge_index)
        
        # Layer normalization
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        
        # Query-dependent gating
        if self.dependent and hasattr(graph, 'query'):
            query = graph['query']
            # Expand query to match node dimensions
            if query.dim() == 2 and output.dim() == 2:
                # Map nodes to their graph indices to get corresponding queries
                if hasattr(graph, 'node2graph'):
                    query_expanded = query[graph['node2graph']]
                else:
                    # Assume single graph or queries are already expanded
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
            # Residual connection
            if hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        return hidden
    
    def forward(self, graph, input_embeds):
        """Forward pass."""
        graph = self.make_undirected(graph)
        output = self.aggregate(graph, input_embeds)
        return output
    
    def make_undirected(self, graph):
        """Make graph undirected by adding inverse edges."""
        if hasattr(graph, 'edge_index'):
            edge_index = graph.edge_index
            inverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
            graph.edge_index = torch.cat([edge_index, inverse_edges], dim=1)
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
        
        # Infer num_relation from context (will be set during initialization)
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
    
    def init_relation_embedding(self, num_relation):
        """Initialize relation embedding when num_relation is known."""
        if self.rel_embedding is None:
            self.num_relation = num_relation
            self.rel_embedding = nn.Embedding(num_relation * 2, self.input_dim)
    
    def forward(self, h_index, r_index, t_index, hidden_states, 
                rel_hidden_states, graph, score_text_embs, all_index):
        """
        Main forward pass matching ConditionedPNA interface.
        
        Args:
            h_index: Head entity indices [batch_size, num_neg]
            r_index: Relation indices [batch_size, num_neg]
            t_index: Tail entity indices [batch_size, num_neg]
            hidden_states: Node embeddings [num_nodes, hidden_dim]
            rel_hidden_states: Relation embeddings
            graph: Graph structure
            score_text_embs: Text embeddings for scoring
            all_index: All entity indices
        """
        # Initialize relation embedding if not done
        if self.rel_embedding is None:
            num_relation = r_index.max().item() + 1
            self.init_relation_embedding(num_relation)
        
        # Remove easy edges during training
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        
        # Make undirected
        graph = self.make_undirected(graph)
        
        # Convert to tail prediction format
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index)
        
        batch_size = len(h_index)
        
        # Create batched graph
        graph_dict, h_index_batch, t_index_batch = self.create_batched_graph(
            graph, h_index, t_index, batch_size)
        
        # Get relation embeddings
        rel_embeds = self.rel_embedding(r_index[:, 0])
        rel_embeds = rel_embeds.type(hidden_states.dtype)
        
        # Initialize input embeddings
        input_embeds, init_score = self.init_input_embeds(
            graph_dict, hidden_states, h_index_batch[:, 0],
            score_text_embs, all_index, rel_embeds, batch_size)
        
        # Aggregate with GAT
        score = self.aggregate(graph_dict, h_index_batch[:, 0], 
                              r_index[:, 0], input_embeds, 
                              rel_embeds, init_score, batch_size)
        
        # Get scores for target nodes
        score = score[t_index_batch]
        return score
    
    def aggregate(self, graph_dict, h_index, r_index, input_embeds, 
                  rel_embeds, init_score, batch_size):
        """Aggregate using GAT layers with progressive edge selection."""
        # Set up graph attributes
        graph_dict['query'] = rel_embeds
        graph_dict['node2graph'] = self.create_node2graph(
            graph_dict['num_nodes'], batch_size)
        
        hidden = input_embeds.clone()
        score = init_score.clone()
        
        for layer_idx, layer in enumerate(self.layers):
            # Select edges based on current scores (simplified for now)
            # In full implementation, use variadic_topks like original
            selected_edges = graph_dict['edge_index']
            
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
            hidden_update = layer(subgraph, layer_input.type(torch.float32))
            
            # Update hidden states
            hidden = hidden + hidden_update.type(hidden.dtype)
            
            # Update scores
            query_expanded = rel_embeds.repeat_interleave(
                graph_dict['num_nodes'] // batch_size, dim=0)
            score = self.score_fn(hidden, query_expanded).type(score.dtype)
        
        return score
    
    def init_input_embeds(self, graph_dict, head_embeds, head_index, 
                         tail_embeds, tail_index, rel_embeds, batch_size):
        """Initialize input embeddings."""
        num_nodes = graph_dict['num_nodes']
        input_embeds = torch.zeros(num_nodes, self.input_dim,
                                   device=rel_embeds.device,
                                   dtype=rel_embeds.dtype)
        
        # Set embeddings for known nodes
        if tail_embeds is not None and tail_index is not None:
            input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        input_embeds[head_index] = head_embeds
        
        # Initialize scores
        score = torch.zeros(num_nodes, device=rel_embeds.device)
        
        # Score head nodes
        rel_embeds_expanded = rel_embeds.repeat_interleave(
            num_nodes // batch_size, dim=0)
        score[head_index] = self.score_fn(
            head_embeds, rel_embeds_expanded[head_index])
        
        return input_embeds, score
    
    def score_fn(self, hidden, rel_embeds):
        """Compute scores for nodes."""
        combined = torch.cat([hidden, rel_embeds], dim=-1)
        heuristic = self.linear(combined)
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score
    
    def create_batched_graph(self, graph, h_index, t_index, batch_size):
        """Create batched graph by repeating base graph."""
        edge_index = graph.edge_index
        num_nodes = graph.num_node
        
        # Batch edge indices
        batched_edges = []
        for i in range(batch_size):
            offset = i * num_nodes
            batched_edges.append(edge_index + offset)
        batched_edge_index = torch.cat(batched_edges, dim=1)
        
        # Offset node indices
        offsets = torch.arange(batch_size, device=h_index.device) * num_nodes
        h_index_batch = h_index + offsets.unsqueeze(-1)
        t_index_batch = t_index + offsets.unsqueeze(-1)
        
        graph_dict = {
            'edge_index': batched_edge_index,
            'num_nodes': num_nodes * batch_size
        }
        
        return graph_dict, h_index_batch, t_index_batch
    
    def create_node2graph(self, num_nodes, batch_size):
        """Create node to graph mapping."""
        nodes_per_graph = num_nodes // batch_size
        node2graph = torch.arange(batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')
        node2graph = node2graph.repeat_interleave(nodes_per_graph)
        return node2graph
    
    def make_undirected(self, graph):
        """Make graph undirected."""
        edge_index = graph.edge_index
        inverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        graph.edge_index = torch.cat([edge_index, inverse_edges], dim=1)
        return graph
    
    def remove_easy_edges(self, graph, h_index, r_index, t_index):
        """Remove edges that make training too easy."""
        edge_index = graph.edge_index
        
        if self.remove_one_hop:
            h_set = set(h_index.flatten().tolist())
            t_set = set(t_index.flatten().tolist())
            
            mask = torch.ones(edge_index.shape[1], dtype=torch.bool,
                            device=edge_index.device)
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if (src in h_set and dst in t_set) or (src in t_set and dst in h_set):
                    mask[i] = False
            
            graph.edge_index = edge_index[:, mask]
        
        return graph
    
    def negative_sample_to_tail(self, h_index, t_index, r_index):
        """Convert all samples to tail prediction format."""
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, 
                                  r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index