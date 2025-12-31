"""
Quick test script to verify SemanticAttentionModule works correctly.
Run from the MKGL-master directory:
    python scripts/test_semantic_attention.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import easydict

from retriever import SemanticAttentionModule


def test_semantic_attention_module():
    """Test the SemanticAttentionModule with synthetic data."""
    print("=" * 60)
    print("Testing SemanticAttentionModule")
    print("=" * 60)
    
    # Configuration
    config = easydict.EasyDict({
        'k': 5,
        'threshold': 0.5,
        'temperature': 1.0,
        'gate_init': 0.1,
        'score_weight': 0.5,
        'use_projection': False,
    })
    
    hidden_dim = 256  # Smaller for testing
    num_entities = 100
    max_k = 10
    batch_size = 4
    
    # Create module
    module = SemanticAttentionModule(config, hidden_dim)
    print(f"Module created: {module}")
    
    # Create synthetic semantic neighbors (sparse)
    # Each entity has 5-10 semantic neighbors
    indices = torch.full((num_entities, max_k), -1, dtype=torch.long)
    scores = torch.zeros((num_entities, max_k), dtype=torch.float)
    
    for i in range(num_entities):
        num_neighbors = torch.randint(3, max_k, (1,)).item()
        neighbors = torch.randperm(num_entities)[:num_neighbors]
        neighbors = neighbors[neighbors != i]  # Remove self
        num_neighbors = len(neighbors)
        
        indices[i, :num_neighbors] = neighbors
        scores[i, :num_neighbors] = torch.rand(num_neighbors) * 0.5 + 0.5  # 0.5-1.0
    
    # Set neighbors
    module.set_semantic_neighbors(indices, scores, num_entities)
    print(f"Loaded {num_entities} entities with max_k={max_k}")
    
    # Test forward pass
    entity_ids = torch.randint(0, num_entities, (batch_size,))
    hidden_states = torch.randn(batch_size, hidden_dim)
    all_entity_embeddings = torch.randn(num_entities, hidden_dim)
    
    print(f"\nInput shapes:")
    print(f"  entity_ids: {entity_ids.shape}")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  all_entity_embeddings: {all_entity_embeddings.shape}")
    
    # Forward pass
    enriched, attention = module(
        entity_ids=entity_ids,
        hidden_states=hidden_states,
        all_entity_embeddings=all_entity_embeddings,
        return_attention=True
    )
    
    print(f"\nOutput shapes:")
    print(f"  enriched_hidden: {enriched.shape}")
    print(f"  attention_weights: {attention.shape if attention is not None else 'None'}")
    
    # Verify output
    assert enriched.shape == hidden_states.shape, "Output shape mismatch!"
    if attention is not None:
        assert attention.shape == (batch_size, config.k), "Attention shape mismatch!"
        # Attention should sum to ~1 (or 0 if all invalid)
        attn_sums = attention.sum(dim=-1)
        print(f"  attention sums: {attn_sums}")
    
    # Test with k=0 (should return original)
    config_k0 = easydict.EasyDict({**config, 'k': 0})
    module_k0 = SemanticAttentionModule(config_k0, hidden_dim)
    module_k0.set_semantic_neighbors(indices, scores, num_entities)
    
    enriched_k0 = module_k0(entity_ids, hidden_states, all_entity_embeddings)
    assert torch.allclose(enriched_k0, hidden_states), "k=0 should return original!"
    print("\n✓ k=0 correctly returns original hidden states")
    
    # Test gate behavior
    print(f"\nGate value: {torch.sigmoid(module.gate).item():.4f}")
    print("(Higher gate = more original, lower = more semantic)")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_neighbor_retrieval():
    """Test the get_semantic_neighbors method."""
    print("\n" + "=" * 60)
    print("Testing neighbor retrieval")
    print("=" * 60)
    
    config = easydict.EasyDict({'k': 3, 'temperature': 1.0, 'gate_init': 0.1, 'score_weight': 0.5})
    module = SemanticAttentionModule(config, 64)
    
    # Simple neighbors
    num_entities = 5
    max_k = 4
    indices = torch.tensor([
        [1, 2, -1, -1],  # Entity 0: neighbors 1, 2
        [0, 2, 3, -1],   # Entity 1: neighbors 0, 2, 3
        [0, 1, 3, 4],    # Entity 2: neighbors 0, 1, 3, 4
        [-1, -1, -1, -1], # Entity 3: no neighbors
        [0, 1, -1, -1],  # Entity 4: neighbors 0, 1
    ])
    scores = torch.tensor([
        [0.9, 0.8, 0.0, 0.0],
        [0.85, 0.7, 0.6, 0.0],
        [0.95, 0.9, 0.85, 0.75],
        [0.0, 0.0, 0.0, 0.0],
        [0.8, 0.7, 0.0, 0.0],
    ])
    
    module.set_semantic_neighbors(indices, scores, num_entities)
    
    # Test retrieval
    entity_ids = torch.tensor([0, 2, 3])
    neighbor_ids, neighbor_scores, valid_mask = module.get_semantic_neighbors(entity_ids, k=3)
    
    print(f"Entity IDs: {entity_ids.tolist()}")
    print(f"Neighbor IDs:\n{neighbor_ids}")
    print(f"Neighbor scores:\n{neighbor_scores}")
    print(f"Valid mask:\n{valid_mask}")
    
    # Verify
    assert neighbor_ids[0, 0].item() == 1, "Entity 0's first neighbor should be 1"
    assert valid_mask[2].sum().item() == 0, "Entity 3 should have no valid neighbors"
    
    print("\n✓ Neighbor retrieval tests passed!")


if __name__ == '__main__':
    test_semantic_attention_module()
    test_neighbor_retrieval()
