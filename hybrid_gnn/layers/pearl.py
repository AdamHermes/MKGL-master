"""
PEARL_GIN - Positional Encoding via Random Laplacian
=====================================================

PEARL (Positional Encoding via Random Laplacian) uses GIN to learn
structural positional encodings from random noise.

Why PEARL?
----------
- Traditional positional encodings (e.g., Laplacian eigenvectors) are expensive
- PEARL uses random noise + GNN propagation to learn structural information
- Isolated nodes get unique PE (from random noise) instead of being ignored
- Permutation equivariant (desirable property for GNNs)

Architecture:
    Input: Random Gaussian Noise [num_nodes, input_dim]
           + Real Edges only (no semantic/random)
    
    Process: 2-layer GINConv with BatchNorm and ReLU
    
    Output: Structural Positional Encoding [num_nodes, hidden_dim]

Key Properties:
    - Runs ONLY on real edges (structural information only)
    - Noise is unique per node → unique PE even for isolated nodes
    - Lightweight: 2 layers of GIN (fast to compute)


Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class PEARL_GIN(nn.Module):
    """
    Positional Encoding via Random Laplacian using GIN.
    
    This module learns structural positional encodings by propagating
    random noise through the graph structure using GIN layers.
    
    Attributes:
        input_dim (int): Dimension of input random noise
        hidden_dim (int): Dimension of output positional encoding
        num_layers (int): Number of GIN layers (default: 2)
        dropout (float): Dropout probability (default: 0.1)
    
    Example:
        >>> pearl = PEARL_GIN(input_dim=32, hidden_dim=32)
        >>> noise = torch.randn(100, 32)  # 100 nodes
        >>> h_pos = pearl(noise, edge_index)  # [100, 32]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        eps: float = 0.0,
        train_eps: bool = True,
    ):
        """
        Initialize PEARL_GIN.
        
        Args:
            input_dim: Dimension of input random noise
            hidden_dim: Dimension of hidden layers and output
            num_layers: Number of GIN layers (recommended: 2-3)
            dropout: Dropout probability between layers
            eps: Initial epsilon value for GIN (learnable if train_eps=True)
            train_eps: Whether to make epsilon trainable
        """
        super(PEARL_GIN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN layers with MLP
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            
            # MLP for GIN: 2-layer MLP with hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # GINConv with learnable epsilon
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Final projection (optional, for dimension matching)
        self.final_proj = nn.Linear(hidden_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for conv in self.convs:
            for module in conv.nn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_noise: torch.Tensor,
        edge_index: torch.Tensor,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Compute positional encodings from random noise.
        
        Args:
            x_noise: Random noise tensor [num_nodes, input_dim]
                    Should be sampled from standard Gaussian N(0, 1)
            edge_index: Graph connectivity [2, num_edges]
                       Should be REAL edges only (not augmented)
            return_all_layers: If True, return list of all layer outputs
        
        Returns:
            h_pos: Positional encoding [num_nodes, hidden_dim]
                  Or list of encodings if return_all_layers=True
        
        Note:
            - For isolated nodes (no edges), the output will be a transformation
              of the original noise, providing a unique identifier.
            - The random noise should be generated fresh for each forward pass
              during training, but can be fixed during inference for consistency.
        """
        h = x_noise
        layer_outputs = []
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # GIN convolution
            h = conv(h, edge_index)
            
            # Batch normalization
            h = bn(h)
            
            # Activation
            h = F.relu(h)
            
            # Dropout (only during training, not on last layer)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            layer_outputs.append(h)
        
        # Final projection
        h = self.final_proj(h)
        
        if return_all_layers:
            return layer_outputs
        
        return h
    
    def generate_noise(
        self,
        num_nodes: int,
        device: torch.device = None,
        seed: int = None,
    ) -> torch.Tensor:
        """
        Generate random noise for positional encoding.
        
        Args:
            num_nodes: Number of nodes in the graph
            device: Target device for the tensor
            seed: Random seed for reproducibility (optional)
        
        Returns:
            noise: Random Gaussian noise [num_nodes, input_dim]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        noise = torch.randn(num_nodes, self.input_dim, device=device)
        return noise
    
    def __repr__(self) -> str:
        return (
            f"PEARL_GIN("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout})"
        )
