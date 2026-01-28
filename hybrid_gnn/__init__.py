"""
Hybrid GNN Module for Knowledge Graph Reasoning
================================================

This module implements a hybrid architecture combining:
- Local: Lightweight PNA (message passing on real KG edges)
- Global: Sparse Graph Transformer (attention on augmented edges)
- Positional Encoding: PEARL (structural PE from random noise)

Architecture Overview:
---------------------
    Input: Text Embeddings (from LLM) + Graph Structure (KG)
           ↓
    [HybridSampler] → Augmented Graph (Real + Semantic + Random edges)
           ↓
    [PEARL_GIN] → Structural Positional Encoding
           ↓
    [HybridBlock x N] → Local PNA + Global GT (parallel)
           ↓
    [Scorer] → Final ranking scores
           ↓
    Output: Contextualized embeddings for LLM

Module Structure:
----------------
    hybrid_gnn/
    ├── __init__.py           # This file
    ├── sampler.py            # HybridSampler (FAISS-based edge augmentation)
    ├── layers/
    │   ├── __init__.py
    │   ├── pearl.py          # PEARL_GIN (positional encoding)
    │   ├── sparse_gt.py      # SparseGTConv (transformer layer)
    │   ├── pna_light.py      # Lightweight PNA layer
    │   └── hybrid_block.py   # Combined Local + Global block
    ├── models/
    │   ├── __init__.py
    │   ├── retriever.py      # HybridRetriever (main model)
    │   └── scorer.py         # Scoring logic
    ├── utils/
    │   ├── __init__.py
    │   ├── graph_utils.py    # Edge selection, graph manipulation
    │   └── score_utils.py    # Score normalization
    └── tests/
        ├── __init__.py
        ├── test_sampler.py
        ├── test_pearl.py
        ├── test_sparse_gt.py
        ├── test_hybrid_block.py
        └── test_retriever.py

Usage:
------
    from hybrid_gnn import HybridRetriever, HybridSampler
    
    # 1. Augment graph with semantic edges
    sampler = HybridSampler(k_semantic=10, k_random=5)
    sampler.build_semantic_index(text_embeddings)
    aug_edge_index, aug_edge_type = sampler.sample_edges(graph, text_embeddings)
    
    # 2. Create retriever and run inference
    retriever = HybridRetriever(config)
    scores = retriever(h_index, r_index, t_index, hidden_states, rel_hidden_states, graph)


Date: January 2026
Version: 1.0.0

Changelog:
----------
    v1.0.0 (2026-01): Initial implementation
        - HybridSampler with FAISS semantic search
        - PEARL_GIN for positional encoding
        - SparseGTConv for global attention
        - HybridBlock combining PNA + GT
        - HybridRetriever as main model

Future TODO:
------------
    - [ ] Add FlashAttention support for SparseGTConv
    - [ ] Implement dynamic edge sampling (per-batch)
    - [ ] Add multi-GPU support with DistributedDataParallel
    - [ ] Optimize FAISS index for GPU
"""

from .sampler import HybridSampler
from .layers import PEARL_GIN, SparseGTConv, HybridBlock
from .models import HybridRetriever, Scorer
from .retriever_integration import HybridScoreRetriever, create_hybrid_retriever

__version__ = "1.0.0"
__author__ = "MKGL Team"

__all__ = [
    "HybridSampler",
    "PEARL_GIN", 
    "SparseGTConv",
    "HybridBlock",
    "HybridRetriever",
    "Scorer",
    "HybridScoreRetriever",
    "create_hybrid_retriever",
]
