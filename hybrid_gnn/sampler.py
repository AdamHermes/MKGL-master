"""
HybridSampler - Edge Augmentation for Knowledge Graphs
======================================================

This module creates augmented graphs with three types of edges:
1. Real edges: Original edges from the knowledge graph
2. Semantic edges: Edges between semantically similar nodes (via FAISS)
3. Random edges: Random node pairs for exploration

Why Augment Edges?
------------------
- **Isolated nodes**: KGs often have nodes with few/no edges. Semantic edges
  connect them to similar nodes, enabling message passing.
- **Long-range dependencies**: Real edges are local. Semantic edges can connect
  distant but related nodes (e.g., "Paris" ↔ "Rome" via similarity).
- **Exploration**: Random edges add stochasticity and prevent overfitting.

Implementation Details:
-----------------------
- Uses FAISS HNSW index for efficient approximate nearest neighbor search
- Semantic edges are computed ONCE during dataset initialization (not per-batch)
- Edge types are stored in `aug_edge_type` tensor for type-aware attention

Performance Notes:
------------------
- FAISS HNSW: O(log N) query time, O(N) memory
- Total edges: O(|E_real| + N*k_semantic + N*k_random)
- Pre-computation recommended for large graphs (>100k nodes)

Usage:
------
    sampler = HybridSampler(k_semantic=10, k_random=5)
    
    # Build index once
    sampler.build_semantic_index(text_embeddings)
    
    # Sample edges (can be called multiple times with different graphs)
    aug_edge_index, aug_edge_type = sampler.sample_edges(graph, text_embeddings)
    
    # Store in graph object
    graph.aug_edge_index = aug_edge_index
    graph.aug_edge_type = aug_edge_type

Edge Type Encoding:
-------------------
    0 = Real edge (from original KG)
    1 = Semantic edge (from FAISS similarity)
    2 = Random edge (uniform sampling)


Date: January 2026

Changelog:
----------
    v1.0.0: Initial implementation with FAISS HNSW
    
Future TODO:
------------
    - [ ] GPU-accelerated FAISS (faiss-gpu)
    - [ ] Dynamic semantic edge sampling (per-batch)
    - [ ] Edge weight based on similarity score
    - [ ] Support for heterogeneous graphs (different node types)
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from torch_geometric.data import Data

# Try to import FAISS, fallback to CPU-based similarity if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: FAISS not available. Using torch-based similarity (slower).")


class HybridSampler:
    """
    Sampler for creating augmented knowledge graphs with semantic and random edges.
    
    The sampler operates in two phases:
    1. build_semantic_index(): Build FAISS index from text embeddings (one-time)
    2. sample_edges(): Generate augmented edge_index and edge_type tensors
    
    Attributes:
        k_semantic (int): Number of semantic neighbors per node
        k_random (int): Number of random edges per node
        use_faiss (bool): Whether to use FAISS (True) or torch fallback (False)
        index: FAISS index object (None until build_semantic_index is called)
        
    Example:
        >>> sampler = HybridSampler(k_semantic=10, k_random=5)
        >>> sampler.build_semantic_index(embeddings)  # [1000, 128]
        >>> aug_edges, aug_types = sampler.sample_edges(graph, embeddings)
        >>> print(aug_edges.shape)  # [2, ~25000] (1000*10 sem + 1000*5 rnd + real)
    """
    
    # Edge type constants for clarity
    EDGE_TYPE_REAL = 0
    EDGE_TYPE_SEMANTIC = 1
    EDGE_TYPE_RANDOM = 2
    
    def __init__(
        self,
        k_semantic: int = 10,
        k_random: int = 5,
        use_faiss: bool = True,
        faiss_index_type: str = "HNSW",
        faiss_hnsw_m: int = 32,
        seed: Optional[int] = None,
    ):
        """
        Initialize the HybridSampler.
        
        Args:
            k_semantic: Number of nearest neighbors for semantic edges.
                       Higher values = more connectivity, more computation.
                       Recommended: 5-20 depending on graph density.
            k_random: Number of random edges per node.
                     Higher values = more exploration, potential noise.
                     Recommended: 3-10.
            use_faiss: If True, use FAISS for similarity search.
                      If False, use torch-based cosine similarity (slower but no dependencies).
            faiss_index_type: Type of FAISS index. Options:
                             - "HNSW": Hierarchical Navigable Small World (recommended)
                             - "Flat": Exact search (slow for large graphs)
                             - "IVF": Inverted file index (good for very large graphs)
            faiss_hnsw_m: HNSW graph connectivity parameter. Higher = more accurate, more memory.
            seed: Random seed for reproducibility of random edges.
        
        Raises:
            ValueError: If k_semantic or k_random is negative.
        """
        if k_semantic < 0 or k_random < 0:
            raise ValueError("k_semantic and k_random must be non-negative")
        
        self.k_semantic = k_semantic
        self.k_random = k_random
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_index_type = faiss_index_type
        self.faiss_hnsw_m = faiss_hnsw_m
        self.seed = seed
        
        # Will be set by build_semantic_index()
        self.index = None
        self.embeddings_cache = None  # Fallback for non-FAISS mode
        self._num_nodes = None
        self._embedding_dim = None
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def build_semantic_index(
        self,
        text_embeddings: torch.Tensor,
        normalize: bool = True,
    ) -> None:
        """
        Build FAISS index from text embeddings for semantic neighbor search.
        
        This method should be called ONCE before calling sample_edges().
        For large graphs, this can take several minutes.
        
        Args:
            text_embeddings: Node embeddings tensor of shape [num_nodes, embedding_dim].
                            These should be text embeddings from LLM or pre-trained encoder.
            normalize: If True, L2-normalize embeddings before indexing.
                      Recommended True for cosine similarity behavior.
        
        Note:
            - Embeddings are converted to float32 (FAISS requirement)
            - Index is built on CPU. For GPU, modify to use faiss.index_cpu_to_gpu()
        
        Example:
            >>> sampler.build_semantic_index(llm_embeddings)  # [5000, 768]
        """
        # Store dimensions for validation
        self._num_nodes = text_embeddings.size(0)
        self._embedding_dim = text_embeddings.size(1)
        
        # Convert to numpy float32 (FAISS requirement)
        embeddings_np = text_embeddings.detach().cpu().numpy().astype('float32')
        
        # Normalize for cosine similarity
        if normalize:
            # L2 normalize each row
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            embeddings_np = embeddings_np / norms
        
        if self.use_faiss:
            # Build FAISS index
            dim = embeddings_np.shape[1]
            
            if self.faiss_index_type == "HNSW":
                # HNSW: Good balance of speed and accuracy
                # M = number of connections per node in the graph
                self.index = faiss.IndexHNSWFlat(dim, self.faiss_hnsw_m)
                self.index.hnsw.efConstruction = 200  # Higher = better quality, slower build
                self.index.hnsw.efSearch = 50  # Higher = better recall, slower search
                
            elif self.faiss_index_type == "Flat":
                # Flat: Exact search, O(N) per query
                self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
                
            elif self.faiss_index_type == "IVF":
                # IVF: Approximate search, good for very large graphs
                nlist = min(100, self._num_nodes // 10)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(embeddings_np)
                self.index.nprobe = 10  # Number of clusters to search
            else:
                raise ValueError(f"Unknown FAISS index type: {self.faiss_index_type}")
            
            # Add embeddings to index
            self.index.add(embeddings_np)
            
        else:
            # Fallback: Store normalized embeddings for torch-based similarity
            self.embeddings_cache = torch.from_numpy(embeddings_np)
    
    def _find_semantic_neighbors_faiss(
        self,
        text_embeddings: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors using FAISS.
        
        Args:
            text_embeddings: Query embeddings [num_nodes, dim]
        
        Returns:
            src_indices: Source node indices [num_nodes * k_semantic]
            dst_indices: Destination node indices [num_nodes * k_semantic]
        """
        # Prepare query embeddings
        query_np = text_embeddings.detach().cpu().numpy().astype('float32')
        
        # Normalize queries
        norms = np.linalg.norm(query_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        query_np = query_np / norms
        
        # Search for k+1 neighbors (first one is self)
        k_search = min(self.k_semantic + 1, self._num_nodes)
        distances, indices = self.index.search(query_np, k_search)
        
        # Remove self-loops (first column is usually the node itself)
        # But verify this by checking if first neighbor is self
        num_nodes = text_embeddings.size(0)
        src_list = []
        dst_list = []
        
        for i in range(num_nodes):
            neighbors = indices[i]
            # Filter out self-loop and invalid indices
            valid_neighbors = [n for n in neighbors if n != i and n >= 0 and n < num_nodes]
            # Take up to k_semantic neighbors
            valid_neighbors = valid_neighbors[:self.k_semantic]
            
            for neighbor in valid_neighbors:
                src_list.append(i)
                dst_list.append(neighbor)
        
        return np.array(src_list), np.array(dst_list)
    
    def _find_semantic_neighbors_torch(
        self,
        text_embeddings: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback: Find k nearest neighbors using torch cosine similarity.
        
        Warning: This is O(N^2) and slow for large graphs. Use FAISS when possible.
        
        Args:
            text_embeddings: Query embeddings [num_nodes, dim]
        
        Returns:
            src_indices: Source node indices
            dst_indices: Destination node indices
        """
        # Normalize embeddings
        embeddings_norm = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute cosine similarity matrix [num_nodes, num_nodes]
        # WARNING: This can be very large for big graphs!
        similarity = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Set diagonal to -inf to exclude self-loops
        similarity.fill_diagonal_(-float('inf'))
        
        # Get top-k neighbors for each node
        k = min(self.k_semantic, similarity.size(1) - 1)
        _, top_k_indices = similarity.topk(k, dim=1)  # [num_nodes, k]
        
        # Convert to edge format
        num_nodes = text_embeddings.size(0)
        src_indices = np.repeat(np.arange(num_nodes), k)
        dst_indices = top_k_indices.cpu().numpy().flatten()
        
        return src_indices, dst_indices
    
    def _sample_random_edges(
        self,
        num_nodes: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random edges uniformly.
        
        Args:
            num_nodes: Total number of nodes
            device: Target device for tensors
        
        Returns:
            src_random: Source indices [num_random_edges]
            dst_random: Destination indices [num_random_edges]
        """
        if self.k_random <= 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )
        
        # Total random edges to sample
        num_random = num_nodes * self.k_random
        
        # Sample random pairs
        src_random = torch.randint(0, num_nodes, (num_random,), device=device)
        dst_random = torch.randint(0, num_nodes, (num_random,), device=device)
        
        # Remove self-loops
        mask = src_random != dst_random
        src_random = src_random[mask]
        dst_random = dst_random[mask]
        
        return src_random, dst_random
    
    def sample_edges(
        self,
        graph: Data,
        text_embeddings: torch.Tensor,
        add_reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate augmented edge_index and edge_type tensors.
        
        This combines:
        1. Real edges from the input graph
        2. Semantic edges from FAISS similarity search
        3. Random edges for exploration
        
        Args:
            graph: PyTorch Geometric Data object with edge_index attribute.
            text_embeddings: Node embeddings [num_nodes, dim] for semantic search.
            add_reverse: If True, add reverse edges for undirected behavior.
        
        Returns:
            aug_edge_index: Combined edge index [2, num_total_edges]
            aug_edge_type: Edge type labels [num_total_edges]
                          0 = real, 1 = semantic, 2 = random
        
        Raises:
            RuntimeError: If build_semantic_index() was not called first.
        
        Example:
            >>> aug_edges, aug_types = sampler.sample_edges(graph, embeddings)
            >>> graph.aug_edge_index = aug_edges
            >>> graph.aug_edge_type = aug_types
        """
        if self.index is None and self.embeddings_cache is None:
            raise RuntimeError(
                "Must call build_semantic_index() before sample_edges(). "
                "This builds the FAISS index for semantic neighbor search."
            )
        
        device = graph.edge_index.device
        num_nodes = graph.num_nodes
        
        # ============================================================
        # 1. Real edges (from original KG)
        # ============================================================
        real_edges = graph.edge_index  # [2, num_real_edges]
        num_real = real_edges.size(1)
        real_types = torch.full(
            (num_real,), 
            self.EDGE_TYPE_REAL, 
            dtype=torch.long, 
            device=device
        )
        
        # ============================================================
        # 2. Semantic edges (from FAISS similarity)
        # ============================================================
        if self.k_semantic > 0:
            if self.use_faiss and self.index is not None:
                src_sem, dst_sem = self._find_semantic_neighbors_faiss(text_embeddings)
            else:
                src_sem, dst_sem = self._find_semantic_neighbors_torch(text_embeddings)
            
            # Convert to tensors
            semantic_edges = torch.stack([
                torch.from_numpy(src_sem).to(device),
                torch.from_numpy(dst_sem).to(device),
            ])
            num_semantic = semantic_edges.size(1)
            semantic_types = torch.full(
                (num_semantic,),
                self.EDGE_TYPE_SEMANTIC,
                dtype=torch.long,
                device=device
            )
        else:
            semantic_edges = torch.empty(2, 0, dtype=torch.long, device=device)
            semantic_types = torch.empty(0, dtype=torch.long, device=device)
        
        # ============================================================
        # 3. Random edges (exploration)
        # ============================================================
        src_random, dst_random = self._sample_random_edges(num_nodes, device)
        if src_random.size(0) > 0:
            random_edges = torch.stack([src_random, dst_random])
            num_random = random_edges.size(1)
            random_types = torch.full(
                (num_random,),
                self.EDGE_TYPE_RANDOM,
                dtype=torch.long,
                device=device
            )
        else:
            random_edges = torch.empty(2, 0, dtype=torch.long, device=device)
            random_types = torch.empty(0, dtype=torch.long, device=device)
        
        # ============================================================
        # 4. Combine all edges
        # ============================================================
        aug_edge_index = torch.cat([real_edges, semantic_edges, random_edges], dim=1)
        aug_edge_type = torch.cat([real_types, semantic_types, random_types])
        
        # Optionally add reverse edges
        if add_reverse:
            reverse_edge_index = aug_edge_index.flip(0)  # Swap src and dst
            aug_edge_index = torch.cat([aug_edge_index, reverse_edge_index], dim=1)
            aug_edge_type = torch.cat([aug_edge_type, aug_edge_type])
        
        return aug_edge_index, aug_edge_type
    
    def get_edge_statistics(
        self,
        aug_edge_type: torch.Tensor,
    ) -> dict:
        """
        Get statistics about the augmented edges.
        
        Useful for debugging and logging.
        
        Args:
            aug_edge_type: Edge type tensor from sample_edges()
        
        Returns:
            Dictionary with edge counts and percentages.
        """
        total = aug_edge_type.size(0)
        num_real = (aug_edge_type == self.EDGE_TYPE_REAL).sum().item()
        num_semantic = (aug_edge_type == self.EDGE_TYPE_SEMANTIC).sum().item()
        num_random = (aug_edge_type == self.EDGE_TYPE_RANDOM).sum().item()
        
        return {
            "total_edges": total,
            "real_edges": num_real,
            "semantic_edges": num_semantic,
            "random_edges": num_random,
            "real_pct": num_real / total * 100 if total > 0 else 0,
            "semantic_pct": num_semantic / total * 100 if total > 0 else 0,
            "random_pct": num_random / total * 100 if total > 0 else 0,
        }
    
    def __repr__(self) -> str:
        return (
            f"HybridSampler("
            f"k_semantic={self.k_semantic}, "
            f"k_random={self.k_random}, "
            f"use_faiss={self.use_faiss}, "
            f"index_type={self.faiss_index_type})"
        )
