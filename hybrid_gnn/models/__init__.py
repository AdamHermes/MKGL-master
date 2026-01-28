"""
Hybrid GNN Models
=================

This submodule contains high-level model classes.

Models:
-------
    - HybridRetriever: Main model for knowledge graph reasoning
    - Scorer: Scoring module for ranking predictions

Import Example:
---------------
    from hybrid_gnn.models import HybridRetriever, Scorer
"""

from .retriever import HybridRetriever
from .scorer import Scorer

__all__ = [
    "HybridRetriever",
    "Scorer",
]
