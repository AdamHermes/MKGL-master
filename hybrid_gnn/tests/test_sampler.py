"""
Tests for HybridSampler
=======================

Run tests:
    pytest hybrid_gnn/tests/test_sampler.py -v
    
Run with coverage:
    pytest hybrid_gnn/tests/test_sampler.py --cov=hybrid_gnn.sampler -v
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hybrid_gnn.sampler import HybridSampler, FAISS_AVAILABLE


# =============================================================================
# Fixtures - Reusable test data
# =============================================================================

@pytest.fixture
def simple_graph():
    """
    Create a simple test graph with 10 nodes and some edges.
    
    Graph structure:
        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5
        |    |    |
        6 -- 7 -- 8
             |
             9 (isolated except one edge)
    """
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 1, 4, 2, 5, 3, 4, 4, 5, 3, 6, 4, 7, 5, 8, 7, 9],
        [1, 0, 2, 1, 3, 0, 4, 1, 5, 2, 4, 3, 5, 4, 6, 3, 7, 4, 8, 5, 9, 7],
    ], dtype=torch.long)
    
    return Data(edge_index=edge_index, num_nodes=10)


@pytest.fixture
def dummy_embeddings():
    """
    Create dummy text embeddings for 10 nodes.
    
    Embeddings are designed so that:
    - Nodes 0, 1, 2 are similar (first cluster)
    - Nodes 3, 4, 5 are similar (second cluster)
    - Nodes 6, 7, 8 are similar (third cluster)
    - Node 9 is somewhat isolated
    """
    torch.manual_seed(42)
    
    # Base embeddings for each cluster
    dim = 32
    embeddings = torch.zeros(10, dim)
    
    # Cluster 1: nodes 0, 1, 2
    cluster1_base = torch.randn(dim)
    embeddings[0] = cluster1_base + torch.randn(dim) * 0.1
    embeddings[1] = cluster1_base + torch.randn(dim) * 0.1
    embeddings[2] = cluster1_base + torch.randn(dim) * 0.1
    
    # Cluster 2: nodes 3, 4, 5
    cluster2_base = torch.randn(dim)
    embeddings[3] = cluster2_base + torch.randn(dim) * 0.1
    embeddings[4] = cluster2_base + torch.randn(dim) * 0.1
    embeddings[5] = cluster2_base + torch.randn(dim) * 0.1
    
    # Cluster 3: nodes 6, 7, 8
    cluster3_base = torch.randn(dim)
    embeddings[6] = cluster3_base + torch.randn(dim) * 0.1
    embeddings[7] = cluster3_base + torch.randn(dim) * 0.1
    embeddings[8] = cluster3_base + torch.randn(dim) * 0.1
    
    # Node 9: random (isolated)
    embeddings[9] = torch.randn(dim)
    
    return embeddings


@pytest.fixture
def large_graph():
    """Create a larger graph for performance testing (100 nodes)."""
    torch.manual_seed(42)
    num_nodes = 100
    num_edges = 300
    
    # Random edges
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    
    # Remove self-loops
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]])
    
    return Data(edge_index=edge_index, num_nodes=num_nodes)


@pytest.fixture
def large_embeddings():
    """Create embeddings for 100 nodes."""
    torch.manual_seed(42)
    return torch.randn(100, 32)


# =============================================================================
# Test: Initialization
# =============================================================================

class TestSamplerInit:
    """Tests for HybridSampler initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        sampler = HybridSampler()
        assert sampler.k_semantic == 10
        assert sampler.k_random == 5
        assert sampler.index is None
    
    def test_custom_init(self):
        """Test initialization with custom parameters."""
        sampler = HybridSampler(k_semantic=20, k_random=10, seed=42)
        assert sampler.k_semantic == 20
        assert sampler.k_random == 10
        assert sampler.seed == 42
    
    def test_invalid_k_values(self):
        """Test that negative k values raise error."""
        with pytest.raises(ValueError):
            HybridSampler(k_semantic=-1)
        
        with pytest.raises(ValueError):
            HybridSampler(k_random=-1)
    
    def test_zero_k_values(self):
        """Test that zero k values are valid."""
        sampler = HybridSampler(k_semantic=0, k_random=0)
        assert sampler.k_semantic == 0
        assert sampler.k_random == 0
    
    def test_repr(self):
        """Test string representation."""
        sampler = HybridSampler(k_semantic=10, k_random=5)
        repr_str = repr(sampler)
        assert "HybridSampler" in repr_str
        assert "k_semantic=10" in repr_str
        assert "k_random=5" in repr_str


# =============================================================================
# Test: Build Semantic Index
# =============================================================================

class TestBuildSemanticIndex:
    """Tests for build_semantic_index method."""
    
    def test_build_index_shape(self, dummy_embeddings):
        """Test that index is built with correct dimensions."""
        sampler = HybridSampler()
        sampler.build_semantic_index(dummy_embeddings)
        
        assert sampler._num_nodes == 10
        assert sampler._embedding_dim == 32
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
    def test_faiss_index_created(self, dummy_embeddings):
        """Test that FAISS index is created when available."""
        sampler = HybridSampler(use_faiss=True)
        sampler.build_semantic_index(dummy_embeddings)
        
        assert sampler.index is not None
    
    def test_torch_fallback(self, dummy_embeddings):
        """Test torch-based fallback when FAISS is disabled."""
        sampler = HybridSampler(use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        assert sampler.embeddings_cache is not None
        assert sampler.embeddings_cache.shape == dummy_embeddings.shape
    
    def test_normalized_embeddings(self, dummy_embeddings):
        """Test that embeddings are normalized."""
        sampler = HybridSampler(use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings, normalize=True)
        
        # Check that cached embeddings are normalized
        norms = torch.norm(sampler.embeddings_cache, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# =============================================================================
# Test: Sample Edges
# =============================================================================

class TestSampleEdges:
    """Tests for sample_edges method."""
    
    def test_sample_edges_without_index(self, simple_graph, dummy_embeddings):
        """Test that error is raised if index not built."""
        sampler = HybridSampler()
        
        with pytest.raises(RuntimeError, match="Must call build_semantic_index"):
            sampler.sample_edges(simple_graph, dummy_embeddings)
    
    def test_sample_edges_output_shape(self, simple_graph, dummy_embeddings):
        """Test output tensor shapes."""
        sampler = HybridSampler(k_semantic=3, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # Check shapes
        assert aug_edge_index.dim() == 2
        assert aug_edge_index.size(0) == 2
        assert aug_edge_type.dim() == 1
        assert aug_edge_index.size(1) == aug_edge_type.size(0)
    
    def test_edge_types_correct(self, simple_graph, dummy_embeddings):
        """Test that edge types are 0, 1, or 2."""
        sampler = HybridSampler(k_semantic=3, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        _, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # Check all edge types are valid
        assert (aug_edge_type >= 0).all()
        assert (aug_edge_type <= 2).all()
    
    def test_real_edges_preserved(self, simple_graph, dummy_embeddings):
        """Test that original edges are preserved."""
        sampler = HybridSampler(k_semantic=3, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # Count real edges
        num_real = (aug_edge_type == 0).sum().item()
        assert num_real == simple_graph.edge_index.size(1)
        
        # Check that original edges are present
        real_mask = aug_edge_type == 0
        real_edges = aug_edge_index[:, real_mask]
        
        # All original edges should be in real_edges
        for i in range(simple_graph.edge_index.size(1)):
            edge = simple_graph.edge_index[:, i]
            found = ((real_edges[0] == edge[0]) & (real_edges[1] == edge[1])).any()
            assert found, f"Original edge {edge.tolist()} not found in augmented graph"
    
    def test_no_self_loops_semantic(self, simple_graph, dummy_embeddings):
        """Test that semantic edges don't have self-loops."""
        sampler = HybridSampler(k_semantic=5, k_random=0, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # Get semantic edges
        semantic_mask = aug_edge_type == 1
        semantic_edges = aug_edge_index[:, semantic_mask]
        
        # Check no self-loops
        self_loops = (semantic_edges[0] == semantic_edges[1]).sum().item()
        assert self_loops == 0, f"Found {self_loops} self-loops in semantic edges"
    
    def test_no_self_loops_random(self, simple_graph, dummy_embeddings):
        """Test that random edges don't have self-loops."""
        sampler = HybridSampler(k_semantic=0, k_random=5, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # Get random edges
        random_mask = aug_edge_type == 2
        random_edges = aug_edge_index[:, random_mask]
        
        # Check no self-loops
        self_loops = (random_edges[0] == random_edges[1]).sum().item()
        assert self_loops == 0, f"Found {self_loops} self-loops in random edges"
    
    def test_valid_node_indices(self, simple_graph, dummy_embeddings):
        """Test that all node indices are valid."""
        sampler = HybridSampler(k_semantic=5, k_random=5, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, _ = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # All indices should be in valid range
        assert aug_edge_index.min() >= 0
        assert aug_edge_index.max() < simple_graph.num_nodes
    
    def test_semantic_edges_cluster_correctly(self, simple_graph, dummy_embeddings):
        """Test that semantic edges connect similar nodes."""
        sampler = HybridSampler(k_semantic=2, k_random=0, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        # Get semantic edges
        semantic_mask = aug_edge_type == 1
        semantic_edges = aug_edge_index[:, semantic_mask]
        
        # Check that nodes in same cluster are connected
        # Cluster 1: 0, 1, 2 should have edges among themselves
        cluster1_nodes = {0, 1, 2}
        cluster1_edges = 0
        
        for i in range(semantic_edges.size(1)):
            src, dst = semantic_edges[0, i].item(), semantic_edges[1, i].item()
            if src in cluster1_nodes and dst in cluster1_nodes:
                cluster1_edges += 1
        
        # At least some intra-cluster edges should exist
        # (with k_semantic=2, each node in cluster should connect to 2 others in cluster)
        assert cluster1_edges > 0, "No semantic edges within cluster 1"
    
    def test_zero_semantic_edges(self, simple_graph, dummy_embeddings):
        """Test with k_semantic=0."""
        sampler = HybridSampler(k_semantic=0, k_random=5, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        num_semantic = (aug_edge_type == 1).sum().item()
        assert num_semantic == 0
    
    def test_zero_random_edges(self, simple_graph, dummy_embeddings):
        """Test with k_random=0."""
        sampler = HybridSampler(k_semantic=5, k_random=0, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        num_random = (aug_edge_type == 2).sum().item()
        assert num_random == 0
    
    def test_add_reverse_edges(self, simple_graph, dummy_embeddings):
        """Test that reverse edges are added correctly."""
        sampler = HybridSampler(k_semantic=2, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index_no_rev, aug_edge_type_no_rev = sampler.sample_edges(
            simple_graph, dummy_embeddings, add_reverse=False
        )
        aug_edge_index_rev, aug_edge_type_rev = sampler.sample_edges(
            simple_graph, dummy_embeddings, add_reverse=True
        )
        
        # With reverse, should have roughly double edges
        assert aug_edge_index_rev.size(1) == 2 * aug_edge_index_no_rev.size(1)
        assert aug_edge_type_rev.size(0) == 2 * aug_edge_type_no_rev.size(0)


# =============================================================================
# Test: Edge Statistics
# =============================================================================

class TestEdgeStatistics:
    """Tests for get_edge_statistics method."""
    
    def test_statistics_counts(self, simple_graph, dummy_embeddings):
        """Test that statistics are computed correctly."""
        sampler = HybridSampler(k_semantic=3, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        stats = sampler.get_edge_statistics(aug_edge_type)
        
        # Check total
        assert stats["total_edges"] == aug_edge_index.size(1)
        
        # Check sum
        total_from_parts = stats["real_edges"] + stats["semantic_edges"] + stats["random_edges"]
        assert total_from_parts == stats["total_edges"]
        
        # Check percentages sum to 100
        pct_sum = stats["real_pct"] + stats["semantic_pct"] + stats["random_pct"]
        assert abs(pct_sum - 100.0) < 1e-5


# =============================================================================
# Test: Performance / Larger Graphs
# =============================================================================

class TestPerformance:
    """Performance tests with larger graphs."""
    
    def test_large_graph_sampling(self, large_graph, large_embeddings):
        """Test sampling on a larger graph (100 nodes)."""
        sampler = HybridSampler(k_semantic=10, k_random=5, use_faiss=False)
        sampler.build_semantic_index(large_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(large_graph, large_embeddings)
        
        # Basic sanity checks
        assert aug_edge_index.size(1) > large_graph.edge_index.size(1)
        assert aug_edge_type.size(0) == aug_edge_index.size(1)
        
        # All indices valid
        assert aug_edge_index.max() < large_graph.num_nodes
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
    def test_faiss_sampling(self, large_graph, large_embeddings):
        """Test FAISS-based sampling."""
        sampler = HybridSampler(k_semantic=10, k_random=5, use_faiss=True)
        sampler.build_semantic_index(large_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(large_graph, large_embeddings)
        
        # Basic sanity checks
        assert aug_edge_index.size(1) > large_graph.edge_index.size(1)
        
        # Check semantic edges exist
        num_semantic = (aug_edge_type == 1).sum().item()
        assert num_semantic > 0


# =============================================================================
# Test: Device Compatibility
# =============================================================================

class TestDeviceCompatibility:
    """Tests for GPU/CPU compatibility."""
    
    def test_cpu_graph(self, simple_graph, dummy_embeddings):
        """Test with CPU tensors."""
        sampler = HybridSampler(k_semantic=3, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings)
        
        assert aug_edge_index.device.type == "cpu"
        assert aug_edge_type.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_graph(self, simple_graph, dummy_embeddings):
        """Test with CUDA tensors."""
        simple_graph.edge_index = simple_graph.edge_index.cuda()
        dummy_embeddings_cuda = dummy_embeddings.cuda()
        
        sampler = HybridSampler(k_semantic=3, k_random=2, use_faiss=False)
        sampler.build_semantic_index(dummy_embeddings_cuda)
        
        aug_edge_index, aug_edge_type = sampler.sample_edges(simple_graph, dummy_embeddings_cuda)
        
        assert aug_edge_index.device.type == "cuda"
        assert aug_edge_type.device.type == "cuda"


# =============================================================================
# Test: Reproducibility
# =============================================================================

class TestReproducibility:
    """Tests for reproducible sampling with seeds."""
    
    def test_random_seed_reproducibility(self, simple_graph, dummy_embeddings):
        """Test that same seed gives same random edges."""
        sampler1 = HybridSampler(k_semantic=0, k_random=5, seed=42, use_faiss=False)
        sampler1.build_semantic_index(dummy_embeddings)
        
        sampler2 = HybridSampler(k_semantic=0, k_random=5, seed=42, use_faiss=False)
        sampler2.build_semantic_index(dummy_embeddings)
        
        # Reset seeds before sampling
        torch.manual_seed(42)
        aug1, type1 = sampler1.sample_edges(simple_graph, dummy_embeddings)
        
        torch.manual_seed(42)
        aug2, type2 = sampler2.sample_edges(simple_graph, dummy_embeddings)
        
        # Random edges should be the same
        random_mask1 = type1 == 2
        random_mask2 = type2 == 2
        
        assert random_mask1.sum() == random_mask2.sum()


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
