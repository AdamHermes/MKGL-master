"""
Hybrid GNN Layers
=================

This submodule contains all neural network layers for the hybrid architecture.

Layers:
-------
    - PEARL_GIN: Positional Encoding via Random Laplacian (using GIN)
    - SparseGTConv: Sparse Graph Transformer Convolution
    - PNALight: Lightweight Principal Neighbourhood Aggregation
    - HybridBlock: Combined Local (PNA) + Global (GT) processing

Import Example:
---------------
    from hybrid_gnn.layers import PEARL_GIN, SparseGTConv, HybridBlock
"""

from .pearl import PEARL_GIN
from .sparse_gt import SparseGTConv
from .hybrid_block import HybridBlock

__all__ = [
    "PEARL_GIN",
    "SparseGTConv", 
    "HybridBlock",
]
