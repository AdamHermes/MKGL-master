"""
HybridBlock - Combined Local + Global Processing
=================================================

The core building block combining Local PNA and Global Sparse GT.

Architecture:
    ┌─────────────────────────────────┐
    │ LOCAL BRANCH: Lightweight PNA   │
    │   - Aggregators: mean, max      │
    │   - Runs on real_edge_index     │
    └───────────────┬─────────────────┘
                    │
    ┌───────────────┴─────────────────┐
    │ GLOBAL BRANCH: SparseGTConv     │
    │   - Multi-head attention        │
    │   - Runs on aug_edge_index      │
    └───────────────┬─────────────────┘
                    │
    ┌───────────────┴─────────────────┐
    │ FUSION: x + local + global      │
    └─────────────────────────────────┘

Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, scatter

from .sparse_gt import SparseGTConv


class LightweightPNA(MessagePassing):
    """
    Lightweight PNA (Principal Neighbourhood Aggregation) layer.
    
    Simplified version with only mean and max aggregators.
    Query-conditioned for relation-aware message passing.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        kwargs.setdefault('aggr', None)
        super(LightweightPNA, self).__init__(node_dim=0, **kwargs)
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Query projection
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Aggregation combination
        self.agg_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        query: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of lightweight PNA.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph connectivity [2, num_edges]
            query: Relation query [batch_size, hidden_dim] or [num_nodes, hidden_dim]
            batch: Batch assignment for nodes
        
        Returns:
            out: Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        if query is not None:
            if query.size(0) != num_nodes:
                if batch is not None:
                    query = query[batch]
                else:
                    query = query.expand(num_nodes, -1)
            query = self.query_proj(query)
        else:
            query = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        
        # Compute degree for normalization
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        
        # Message passing
        out = self.propagate(
            edge_index,
            x=x,
            query=query,
            deg_inv_sqrt=deg_inv_sqrt,
        )
        
        # Output projection with residual
        out = self.out_proj(out)
        out = out + x
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        query_i: torch.Tensor,
        deg_inv_sqrt_i: torch.Tensor,
        deg_inv_sqrt_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute messages with query conditioning."""
        msg_input = torch.cat([x_j, query_i], dim=-1)
        msg = self.msg_mlp(msg_input)
        norm = deg_inv_sqrt_i.unsqueeze(-1) * deg_inv_sqrt_j.unsqueeze(-1)
        return msg * norm
    
    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        dim_size: int = None,
    ) -> torch.Tensor:
        """Aggregate messages using mean and max."""
        agg_mean = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mean')
        agg_max = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='max')
        combined = torch.cat([agg_mean, agg_max], dim=-1)
        return self.agg_combine(combined)


class HybridBlock(nn.Module):
    """
    Hybrid processing block combining Local PNA and Global Sparse GT.
    
    Attributes:
        hidden_dim (int): Hidden dimension
        num_heads (int): Number of attention heads for GT
        num_edge_types (int): Number of edge types
        use_local (bool): Enable local PNA branch
        use_global (bool): Enable global GT branch
    
    Example:
        >>> block = HybridBlock(hidden_dim=64, num_heads=4)
        >>> out = block(x, real_edges, aug_edges, aug_types, query)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_edge_types: int = 3,
        dropout: float = 0.1,
        use_local: bool = True,
        use_global: bool = True,
    ):
        super(HybridBlock, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_local = use_local
        self.use_global = use_global
        
        # Local branch: Lightweight PNA
        if use_local:
            self.local_pna = LightweightPNA(
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        
        # Global branch: Sparse GT
        if use_global:
            self.global_gt = SparseGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
        
        # Fusion layer
        num_branches = int(use_local) + int(use_global)
        if num_branches > 1:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_branches, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.fusion = nn.Identity()
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        real_edge_index: torch.Tensor,
        aug_edge_index: torch.Tensor,
        aug_edge_type: torch.Tensor,
        query: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of hybrid block.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            real_edge_index: Real KG edges [2, num_real_edges]
            aug_edge_index: Augmented edges [2, num_aug_edges]
            aug_edge_type: Edge types [num_aug_edges]
            query: Relation query
            batch: Batch assignment
        
        Returns:
            out: Updated node features [num_nodes, hidden_dim]
        """
        outputs = []
        
        # Local branch
        if self.use_local:
            h_local = self.local_pna(x, real_edge_index, query=query, batch=batch)
            outputs.append(h_local)
        
        # Global branch
        if self.use_global:
            h_global = self.global_gt(x, aug_edge_index, aug_edge_type)
            outputs.append(h_global)
        
        # Fusion
        if len(outputs) > 1:
            h_concat = torch.cat(outputs, dim=-1)
            h_fused = self.fusion(h_concat)
        else:
            h_fused = outputs[0]
        
        # Residual and final norm
        out = self.final_norm(x + h_fused)
        
        return out
    
    def __repr__(self) -> str:
        return (
            f"HybridBlock("
            f"hidden_dim={self.hidden_dim}, "
            f"heads={self.num_heads}, "
            f"local={self.use_local}, "
            f"global={self.use_global})"
        )
