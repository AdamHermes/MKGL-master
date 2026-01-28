"""
SparseGTConv - Sparse Graph Transformer Layer
==============================================

Graph Transformer with sparse attention on augmented edges.

Why Sparse GT instead of Full Attention?
-----------------------------------------
- Full attention is O(N²) → infeasible for large graphs
- Sparse attention only attends to neighbors → O(E) complexity
- Edge-type awareness allows different attention patterns for different edge types

Architecture:
    Multi-Head Attention:
        Q = Linear(x)  # Query
        K = Linear(x)  # Key  
        V = Linear(x)  # Value
        
        Edge Bias = Embedding(edge_type)
        
        Attention = softmax((Q_i · K_j + Q_i · EdgeBias) / sqrt(d))
        Output = Σ (Attention * V_j)

Edge Type Semantics:
-------------------
- Real edges (type 0): Structural neighbors from KG
- Semantic edges (type 1): Similar nodes from embedding space
- Random edges (type 2): Exploration/regularization


Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class SparseGTConv(MessagePassing):
    """
    Sparse Graph Transformer Convolution Layer.
    
    Performs multi-head self-attention over the graph edges with
    edge-type aware attention biases.
    
    Attributes:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        num_heads (int): Number of attention heads
        num_edge_types (int): Number of edge types (default: 3)
        dropout (float): Dropout probability for attention weights
    
    Example:
        >>> layer = SparseGTConv(in_channels=64, out_channels=64, num_heads=4)
        >>> out = layer(x, edge_index, edge_type)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.1,
        bias: bool = True,
        add_self_loops: bool = False,
        **kwargs,
    ):
        """
        Initialize SparseGTConv.
        
        Args:
            in_channels: Dimension of input node features
            out_channels: Dimension of output node features
            num_heads: Number of attention heads (out_channels must be divisible by num_heads)
            num_edge_types: Number of distinct edge types (3 for real/semantic/random)
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear projections
            add_self_loops: Whether to add self-loops (usually False for GT)
        """
        kwargs.setdefault('aggr', 'add')
        super(SparseGTConv, self).__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        assert out_channels % num_heads == 0, \
            f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = out_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.lin_q = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_k = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_v = nn.Linear(in_channels, out_channels, bias=bias)
        
        # Edge type embedding for attention bias
        self.edge_type_emb = nn.Embedding(num_edge_types, out_channels)
        
        # Output projection
        self.lin_out = nn.Linear(out_channels, out_channels, bias=bias)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout),
        )
        
        # Skip connection projection
        self.skip_proj = nn.Linear(in_channels, out_channels, bias=False) \
            if in_channels != out_channels else nn.Identity()
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters for stable training."""
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        nn.init.normal_(self.edge_type_emb.weight, mean=0.0, std=0.02)
        
        if self.lin_q.bias is not None:
            nn.init.zeros_(self.lin_q.bias)
            nn.init.zeros_(self.lin_k.bias)
            nn.init.zeros_(self.lin_v.bias)
            nn.init.zeros_(self.lin_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass of sparse graph transformer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_type: Edge type labels [num_edges] with values in {0, 1, 2}
            return_attention: If True, also return attention weights
        
        Returns:
            out: Updated node features [num_nodes, out_channels]
            attn_weights: (optional) Attention weights [num_edges, num_heads]
        """
        # Pre-normalization
        x_norm = self.norm1(x)
        
        # Compute Q, K, V projections
        q = self.lin_q(x_norm).view(-1, self.num_heads, self.head_dim)
        k = self.lin_k(x_norm).view(-1, self.num_heads, self.head_dim)
        v = self.lin_v(x_norm).view(-1, self.num_heads, self.head_dim)
        
        self._attn_weights = None
        
        # Message passing
        out = self.propagate(
            edge_index,
            q=q, k=k, v=v,
            edge_type=edge_type,
            size=None,
        )
        
        # Reshape
        out = out.view(-1, self.out_channels)
        
        # Output projection
        out = self.lin_out(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        # Residual connection
        out = out + self.skip_proj(x)
        
        # Post-attention normalization and FFN
        out_norm = self.norm2(out)
        out = out + self.ffn(out_norm)
        
        if return_attention:
            return out, self._attn_weights
        
        return out
    
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_type: torch.Tensor,
        index: torch.Tensor,
        ptr,
        size_i,
    ) -> torch.Tensor:
        """Compute attention-weighted messages."""
        # Get edge type embeddings
        edge_emb = self.edge_type_emb(edge_type)
        edge_emb = edge_emb.view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn = (q_i * k_j).sum(dim=-1) * self.scale
        edge_bias = (q_i * edge_emb).sum(dim=-1) * self.scale
        attn = attn + edge_bias
        
        # Softmax over neighbors
        attn = softmax(attn, index, ptr, size_i)
        self._attn_weights = attn.detach()
        
        # Apply dropout
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Weight values by attention
        return attn.unsqueeze(-1) * v_j
    
    def __repr__(self) -> str:
        return (
            f"SparseGTConv("
            f"in={self.in_channels}, "
            f"out={self.out_channels}, "
            f"heads={self.num_heads}, "
            f"edge_types={self.num_edge_types})"
        )
