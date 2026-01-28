"""
Hybrid GNN Utilities
====================

This submodule contains utility functions for graph manipulation and training.

Modules:
--------
    - graph_utils: Edge selection, graph augmentation, batch processing
    - training_utils: Loss functions, metrics, training utilities

Import Example:
---------------
    from hybrid_gnn.utils import EdgeSelector, compute_ranking_metrics
"""

from .graph_utils import (
    multikey_argsort,
    bincount,
    variadic_topks,
    to_undirected_with_inverse,
    EdgeSelector,
    remove_easy_edges,
    negative_sample_to_tail,
    create_subgraph_from_edges,
)

from .training_utils import (
    ContrastiveLoss,
    HybridLoss,
    BPRLoss,
    compute_ranking_metrics,
    compute_hits_at_k_fast,
    compute_mrr_fast,
    LabelSmoothingLoss,
    FocalLoss,
)

__all__ = [
    # Graph utilities
    'multikey_argsort',
    'bincount',
    'variadic_topks',
    'to_undirected_with_inverse',
    'EdgeSelector',
    'remove_easy_edges',
    'negative_sample_to_tail',
    'create_subgraph_from_edges',
    # Training utilities
    'ContrastiveLoss',
    'HybridLoss',
    'BPRLoss',
    'compute_ranking_metrics',
    'compute_hits_at_k_fast',
    'compute_mrr_fast',
    'LabelSmoothingLoss',
    'FocalLoss',
]
