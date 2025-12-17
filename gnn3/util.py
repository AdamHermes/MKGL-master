import torch
from torch_scatter import scatter_add
from torch_geometric.data import Data, Batch


def multikey_argsort(inputs, descending=False, break_tie=False):
    """Sort by multiple keys"""
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    return order


def bincount(input, minlength=0):
    """Optimized bincount with sorted input detection"""
    if input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)
    
    sorted_input = (input.diff() >= 0).all()
    if sorted_input:
        if minlength == 0:
            minlength = input.max() + 1
        range_tensor = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range_tensor, input)
        return index.diff()
    
    return input.bincount(minlength=minlength)


def variadic_topks(input, size, ks, largest=True, break_tie=False):
    """
    Select top-k elements for each group in variadic-sized batches
    
    Args:
        input: Values to select from [total_elements]
        size: Size of each group [batch_size]
        ks: Number of elements to select per group [batch_size]
        largest: If True, select largest values
        break_tie: If True, break ties randomly
    
    Returns:
        values: Selected values
        indices: Indices of selected values
    """
    index2sample = torch.repeat_interleave(torch.arange(len(size), device=input.device), size)
    if largest:
        index2sample = -index2sample
    order = multikey_argsort((index2sample, input), descending=largest, break_tie=break_tie)
    
    range_tensor = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range_tensor = range_tensor + offset.repeat_interleave(ks)
    index = order[range_tensor]
    
    return input[index], index


class VirtualTensor:
    """
    Sparse tensor that only materializes non-zero entries
    Efficient for cases where only a few entries are set
    """
    def __init__(self, keys=None, values=None, index=None, input=None, 
                 shape=None, dtype=None, device=None):
        if shape is None:
            shape = index.shape + input.shape[1:]
        if index is None:
            index = torch.zeros(*shape[:1], dtype=torch.long, device=device)
        if input is None:
            input = torch.empty(1, *shape[1:], dtype=dtype, device=device)
        if keys is None:
            keys = torch.empty(0, dtype=torch.long, device=device)
        if values is None:
            values = torch.empty(0, *shape[1:], dtype=dtype, device=device)
        
        self.keys = keys
        self.values = values
        self.index = index
        self.input = input
    
    @classmethod
    def zeros(cls, *shape, dtype=None, device=None):
        """Create a virtual tensor filled with zeros"""
        input = torch.zeros(1, *shape[1:], dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)
    
    @classmethod
    def full(cls, shape, value, dtype=None, device=None):
        """Create a virtual tensor filled with a value"""
        input = torch.full((1,) + shape[1:], value, dtype=dtype, device=device)
        return cls(input=input, shape=shape, dtype=dtype, device=device)
    
    @classmethod
    def gather(cls, input, index):
        """Gather values from input using index"""
        return cls(index=index, input=input, dtype=input.dtype, device=input.device)
    
    def clone(self):
        """Clone the virtual tensor"""
        return VirtualTensor(
            self.keys.clone(), 
            self.values.clone(), 
            self.index.clone(), 
            self.input.clone()
        )
    
    @property
    def shape(self):
        return self.index.shape + self.input.shape[1:]
    
    @property
    def dtype(self):
        return self.values.dtype if self.values.numel() > 0 else self.input.dtype
    
    @property
    def device(self):
        return self.values.device if self.values.numel() > 0 else self.input.device
    
    def __getitem__(self, indexes):
        """Get values at specified indices"""
        if not isinstance(indexes, tuple):
            indexes = (indexes,)
        keys = indexes[0]
        
        # Handle empty keys
        if keys.numel() == 0:
            return torch.empty(0, *self.shape[1:], dtype=self.dtype, device=self.device)
        
        # Bounds checking
        if keys.max() >= len(self.index) or keys.min() < 0:
            raise IndexError(
                f"Index out of bounds: keys range [{keys.min()}, {keys.max()}], "
                f"but VirtualTensor has {len(self.index)} elements"
            )
        
        values = self.input[(self.index[keys],) + indexes[1:]]
        
        if len(self.keys) > 0:
            index = torch.bucketize(keys, self.keys)
            index = index.clamp(max=len(self.keys) - 1)
            indexes_new = (index,) + indexes[1:]
            found = keys == self.keys[index]
            indexes_found = tuple(idx[found] for idx in indexes_new)
            values[found] = self.values[indexes_found]
        
        return values
    
    def __setitem__(self, keys, values):
        """Set values at specified indices"""
        new_keys, inverse = torch.cat([self.keys, keys]).unique(return_inverse=True)
        new_values = torch.zeros(
            len(new_keys), *self.shape[1:], dtype=self.dtype, device=self.device
        )
        new_values[inverse[:len(self.keys)]] = self.values
        new_values[inverse[len(self.keys):]] = values
        self.keys = new_keys
        self.values = new_values
    
    def __len__(self):
        return self.shape[0]


class RepeatGraph:
    """
    Efficiently repeat a graph structure multiple times for batch processing
    Uses lazy evaluation to avoid materializing large tensors
    """
    def __init__(self, graph, repeats):
        """
        Args:
            graph: PyG Data object
            repeats: Number of times to repeat the graph
        """
        self.input = graph
        self.repeats = repeats
        
        # Basic properties
        self.num_nodes_per_graph = graph.num_nodes
        self.num_edges_per_graph = graph.num_edges
        self.num_nodes = graph.num_nodes * repeats
        self.num_edges = graph.num_edges * repeats
        self.num_relation = graph.num_relation if hasattr(graph, 'num_relation') else None
        
        self.device = graph.edge_index.device
        
        # Compute offsets for each repeated graph
        self._node_offsets = torch.arange(repeats, device=self.device) * graph.num_nodes
        self._edge_offsets = torch.arange(repeats, device=self.device) * graph.num_edges
    
    @property
    def edge_index(self):
        """Get edge indices for repeated graph"""
        # Repeat edge_index and add offsets
        edge_index = self.input.edge_index.repeat(1, self.repeats)
        offsets = self._node_offsets.repeat_interleave(self.num_edges_per_graph)
        edge_index = edge_index + offsets.unsqueeze(0)
        return edge_index
    
    @property
    def edge_type(self):
        """Get edge types for repeated graph"""
        if hasattr(self.input, 'edge_type'):
            return self.input.edge_type.repeat(self.repeats)
        return None
    
    @property
    def edge_attr(self):
        """Get edge attributes for repeated graph"""
        if hasattr(self.input, 'edge_attr'):
            return self.input.edge_attr.repeat(self.repeats, 1)
        return None
    
    @property
    def batch(self):
        """Get batch assignment for nodes"""
        return torch.arange(self.repeats, device=self.device).repeat_interleave(
            self.num_nodes_per_graph
        )
    
    @property
    def edge_batch(self):
        """Get batch assignment for edges"""
        return torch.arange(self.repeats, device=self.device).repeat_interleave(
            self.num_edges_per_graph
        )
    
    def degree_out(self):
        """Compute out-degree for all nodes"""
        return torch.bincount(
            self.edge_index[0], 
            minlength=self.num_nodes
        )
    
    def subgraph(self, edge_mask):
        """
        Extract subgraph using edge mask
        
        Args:
            edge_mask: Boolean mask or indices [num_selected_edges]
        
        Returns:
            New RepeatGraph with selected edges
        """
        # Handle empty selection
        if isinstance(edge_mask, torch.Tensor):
            if edge_mask.numel() == 0:
                # Return empty graph
                empty_graph = Data(
                    edge_index=torch.empty((2, 0), dtype=torch.long, device=self.device),
                    num_nodes=0
                )
                if self.num_relation is not None:
                    empty_graph.num_relation = self.num_relation
                node_map = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
                node_indices = torch.tensor([], dtype=torch.long, device=self.device)
                return empty_graph, node_map, node_indices
        
        if edge_mask.dtype == torch.bool:
            edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        else:
            edge_indices = edge_mask
        
        # Validate edge indices
        if len(edge_indices) > 0:
            if edge_indices.max() >= self.num_edges or edge_indices.min() < 0:
                raise IndexError(
                    f"Edge indices out of bounds: range [{edge_indices.min()}, {edge_indices.max()}], "
                    f"but graph has {self.num_edges} edges"
                )
        
        # Get selected edges
        edge_index = self.edge_index[:, edge_indices]
        edge_type = self.edge_type[edge_indices] if self.edge_type is not None else None
        edge_attr = self.edge_attr[edge_indices] if self.edge_attr is not None else None
        
        # Find which nodes are involved
        node_indices = torch.unique(edge_index.flatten())
        
        # Remap edge indices to compact node indices
        node_map = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        node_map[node_indices] = torch.arange(len(node_indices), device=self.device)
        edge_index_compact = node_map[edge_index]
        
        # Create new graph
        new_graph = Data(
            edge_index=edge_index_compact,
            num_nodes=len(node_indices)
        )
        if edge_type is not None:
            new_graph.edge_type = edge_type
        if edge_attr is not None:
            new_graph.edge_attr = edge_attr
        if self.num_relation is not None:
            new_graph.num_relation = self.num_relation
        
        # Copy attributes to new graph
        new_graph.node_indices = node_indices  # Track original node indices
        new_graph.edge_indices = edge_indices  # Track original edge indices
        
        return new_graph, node_map, node_indices
    
    def neighbors(self, node_indices):
        """
        Get neighbors of specified nodes
        
        Args:
            node_indices: Node indices to query [num_query_nodes]
        
        Returns:
            edge_indices: Edge indices connecting to these nodes
            neighbor_indices: Target node indices
        """
        edge_index = self.edge_index
        
        # Find edges where source is in node_indices
        # Create a mask for source nodes
        source_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        source_mask[node_indices] = True
        edge_mask = source_mask[edge_index[0]]
        
        edge_indices = edge_mask.nonzero(as_tuple=True)[0]
        neighbor_indices = edge_index[1, edge_indices]
        
        return edge_indices, neighbor_indices
    
    def num_neighbors(self, node_indices):
        """Count number of neighbors for specified nodes"""
        edge_index = self.edge_index
        source_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        source_mask[node_indices] = True
        edge_mask = source_mask[edge_index[0]]
        
        # Get the source nodes of edges that match
        edge_sources = edge_index[0][edge_mask]
        
        # Count edges per node (only for nodes in node_indices)
        # Create ones for each edge
        ones = torch.ones_like(edge_sources)
        num_neighbors = scatter_add(
            ones, 
            edge_sources,
            dim=0,
            dim_size=self.num_nodes
        )
        return num_neighbors[node_indices]
    
    def __getattr__(self, name):
        """Delegate attribute access to input graph"""
        if 'input' in self.__dict__:
            return getattr(self.input, name)
        raise AttributeError(f"RepeatGraph has no attribute '{name}'")