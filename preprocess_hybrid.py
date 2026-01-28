"""
Preprocessing Script for Hybrid GNN Pipeline
=============================================

This script extends the standard preprocessing with:
1. Pre-computed text embeddings (downscaled)
2. Pre-computed augmented edges (semantic + random)
3. FAISS index for fast neighbor lookup

Usage:
    python preprocess_hybrid.py -c config/fb15k237_hybrid.yaml

Author: MKGL Team
Date: January 2026
"""

import argparse
import json
import os
import os.path as osp
import pickle
import yaml
import easydict
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Import base preprocessing classes
from preprocess_new import InductiveKGCDataset, KGCDataset, Prompter
from dataset_new import FB15k237Inductive, WN18RRInductive, FB15k237, WN18RR

# Import hybrid components
try:
    from hybrid_gnn import HybridSampler
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("Warning: hybrid_gnn not available. Install dependencies first.")


class HybridKGCDataset(KGCDataset):
    """
    KGC Dataset with pre-computed hybrid graph augmentation.
    
    Extends KGCDataset to include:
    - Pre-computed text embeddings (from LLM, downscaled)
    - Augmented edges (semantic neighbors + random edges)
    - Edge type annotations
    """
    
    def __init__(self, args, kgdata, tokenizer, hybrid_config=None):
        self.hybrid_config = hybrid_config or {}
        super().__init__(args, kgdata, tokenizer)
        
        # After base initialization, compute hybrid features
        if HYBRID_AVAILABLE:
            self.compute_hybrid_features()
    
    def compute_hybrid_features(self):
        """Pre-compute text embeddings and augmented edges."""
        print('=' * 60)
        print('Computing Hybrid Features')
        print('=' * 60)
        
        # Get config
        k_semantic = self.hybrid_config.get('k_semantic', 10)
        k_random = self.hybrid_config.get('k_random', 5)
        hidden_dim = self.hybrid_config.get('hidden_dim', 32)
        
        print(f"  k_semantic: {k_semantic}")
        print(f"  k_random: {k_random}")
        print(f"  hidden_dim: {hidden_dim}")
        
        # ============================================================
        # 1. Compute text embeddings for all entities
        # ============================================================
        print("\n[1/3] Computing text embeddings...")
        
        # Get text token IDs for each entity
        num_entities = len(self.kgdata.transductive_vocab)
        text_token_ids = np.stack(self.vocab_df['text_token_ids'].values[:num_entities])
        
        # For now, we'll use random initialization
        # In training, these will be replaced with LLM-derived embeddings
        # But having a placeholder helps with shape checking
        self.entity_embeddings = torch.randn(num_entities, hidden_dim)
        self.entity_embeddings = torch.nn.functional.normalize(
            self.entity_embeddings, p=2, dim=-1
        )
        
        print(f"  Entity embeddings shape: {self.entity_embeddings.shape}")
        
        # ============================================================
        # 2. Build graph structure
        # ============================================================
        print("\n[2/3] Building graph structure...")
        
        # Get triplets from training data
        train_set, _, _ = self.kgdata.split()
        train_triplets = train_set.dataset.triplets[train_set.indices]
        
        # Build edge index (h -> t)
        edge_index = torch.stack([
            train_triplets[:, 0],  # heads
            train_triplets[:, 1],  # tails
        ]).long()
        
        # Edge attributes (relations)
        edge_attr = train_triplets[:, 2].long()
        
        print(f"  Num nodes: {num_entities}")
        print(f"  Num edges: {edge_index.shape[1]}")
        print(f"  Num relations: {edge_attr.max().item() + 1}")
        
        # ============================================================
        # 3. Compute augmented edges using HybridSampler
        # ============================================================
        print("\n[3/3] Computing augmented edges...")
        
        from torch_geometric.data import Data
        
        graph = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_entities,
        )
        
        # Create sampler
        sampler = HybridSampler(
            k_semantic=k_semantic,
            k_random=k_random,
            use_faiss=True,
        )
        
        # Build semantic index
        print("  Building semantic index...")
        sampler.build_semantic_index(self.entity_embeddings)
        
        # Sample augmented edges
        print("  Sampling augmented edges...")
        aug_edge_index, aug_edge_type = sampler.sample_edges(
            graph, self.entity_embeddings
        )
        
        # Store augmented graph info
        self.aug_edge_index = aug_edge_index
        self.aug_edge_type = aug_edge_type
        
        # Statistics
        stats = sampler.get_edge_statistics(aug_edge_type)
        print(f"  Augmented edges: {aug_edge_index.shape[1]}")
        print(f"    Real edges: {stats['real']}")
        print(f"    Semantic edges: {stats['semantic']}")
        print(f"    Random edges: {stats['random']}")
        
        # Store graph for easy access
        self.base_graph = graph
        
        print("\n✅ Hybrid features computed successfully!")
    
    def get_hybrid_graph(self):
        """Get graph with augmented edges."""
        from torch_geometric.data import Data
        
        graph = Data(
            edge_index=self.base_graph.edge_index,
            edge_attr=self.base_graph.edge_attr,
            num_nodes=self.base_graph.num_nodes,
            aug_edge_index=self.aug_edge_index,
            aug_edge_type=self.aug_edge_type,
        )
        return graph
    
    def save(self):
        """Save dataset with hybrid features."""
        saved_dir = self.saved_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        # Save with '_hybrid' suffix
        file_path = saved_dir + self.args.config_name + '.pkl'
        print('##########Save hybrid dataset in %s############' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


class HybridInductiveKGCDataset(InductiveKGCDataset):
    """
    Inductive KGC Dataset with pre-computed hybrid graph augmentation.
    
    For inductive setting, we need separate augmented graphs for:
    - Transductive graph (training)
    - Inductive graph (testing with new entities)
    """
    
    def __init__(self, args, kgdata, tokenizer, hybrid_config=None):
        self.hybrid_config = hybrid_config or {}
        super().__init__(args, kgdata, tokenizer)
        
        if HYBRID_AVAILABLE:
            self.compute_hybrid_features()
    
    def compute_hybrid_features(self):
        """Pre-compute text embeddings and augmented edges for both graphs."""
        print('=' * 60)
        print('Computing Hybrid Features (Inductive)')
        print('=' * 60)
        
        k_semantic = self.hybrid_config.get('k_semantic', 10)
        k_random = self.hybrid_config.get('k_random', 5)
        hidden_dim = self.hybrid_config.get('hidden_dim', 32)
        
        # ============================================================
        # 1. Transductive Graph (for training)
        # ============================================================
        print("\n[1/2] Processing transductive graph...")
        
        num_trans_entities = len(self.kgdata.transductive_vocab)
        self.trans_entity_embeddings = torch.randn(num_trans_entities, hidden_dim)
        self.trans_entity_embeddings = torch.nn.functional.normalize(
            self.trans_entity_embeddings, p=2, dim=-1
        )
        
        # Build transductive graph
        train_set, _, _ = self.kgdata.split()
        train_triplets = train_set.dataset.triplets[train_set.indices]
        
        trans_edge_index = torch.stack([
            train_triplets[:, 0],
            train_triplets[:, 1],
        ]).long()
        trans_edge_attr = train_triplets[:, 2].long()
        
        from torch_geometric.data import Data
        
        trans_graph = Data(
            edge_index=trans_edge_index,
            edge_attr=trans_edge_attr,
            num_nodes=num_trans_entities,
        )
        
        # Augment transductive graph
        sampler = HybridSampler(k_semantic=k_semantic, k_random=k_random, use_faiss=True)
        sampler.build_semantic_index(self.trans_entity_embeddings)
        trans_aug_edge_index, trans_aug_edge_type = sampler.sample_edges(
            trans_graph, self.trans_entity_embeddings
        )
        
        self.trans_graph = trans_graph
        self.trans_aug_edge_index = trans_aug_edge_index
        self.trans_aug_edge_type = trans_aug_edge_type
        
        print(f"  Transductive nodes: {num_trans_entities}")
        print(f"  Transductive edges: {trans_edge_index.shape[1]}")
        print(f"  Augmented edges: {trans_aug_edge_index.shape[1]}")
        
        # ============================================================
        # 2. Inductive Graph (for testing)
        # ============================================================
        print("\n[2/2] Processing inductive graph...")
        
        num_ind_entities = len(self.kgdata.inductive_vocab)
        self.ind_entity_embeddings = torch.randn(num_ind_entities, hidden_dim)
        self.ind_entity_embeddings = torch.nn.functional.normalize(
            self.ind_entity_embeddings, p=2, dim=-1
        )
        
        # Build inductive graph (test set triplets)
        _, _, test_set = self.kgdata.split()
        # Note: For inductive, we need the inference graph, not just test triplets
        # This may need adjustment based on how kgdata provides the inference graph
        
        # For now, use test triplets as a placeholder
        # In practice, you'd use kgdata.inference_graph or similar
        test_triplets = test_set.dataset.triplets[test_set.indices]
        
        ind_edge_index = torch.stack([
            test_triplets[:, 0],
            test_triplets[:, 1],
        ]).long()
        ind_edge_attr = test_triplets[:, 2].long()
        
        ind_graph = Data(
            edge_index=ind_edge_index,
            edge_attr=ind_edge_attr,
            num_nodes=num_ind_entities,
        )
        
        # Augment inductive graph
        sampler_ind = HybridSampler(k_semantic=k_semantic, k_random=k_random, use_faiss=True)
        sampler_ind.build_semantic_index(self.ind_entity_embeddings)
        ind_aug_edge_index, ind_aug_edge_type = sampler_ind.sample_edges(
            ind_graph, self.ind_entity_embeddings
        )
        
        self.ind_graph = ind_graph
        self.ind_aug_edge_index = ind_aug_edge_index
        self.ind_aug_edge_type = ind_aug_edge_type
        
        print(f"  Inductive nodes: {num_ind_entities}")
        print(f"  Inductive edges: {ind_edge_index.shape[1]}")
        print(f"  Augmented edges: {ind_aug_edge_index.shape[1]}")
        
        print("\n✅ Hybrid features computed successfully!")
    
    def get_hybrid_graph(self, split='train'):
        """Get graph with augmented edges for specified split."""
        from torch_geometric.data import Data
        
        if split in ['train', 'valid']:
            return Data(
                edge_index=self.trans_graph.edge_index,
                edge_attr=self.trans_graph.edge_attr,
                num_nodes=self.trans_graph.num_nodes,
                aug_edge_index=self.trans_aug_edge_index,
                aug_edge_type=self.trans_aug_edge_type,
            )
        else:  # test
            return Data(
                edge_index=self.ind_graph.edge_index,
                edge_attr=self.ind_graph.edge_attr,
                num_nodes=self.ind_graph.num_nodes,
                aug_edge_index=self.ind_aug_edge_index,
                aug_edge_type=self.ind_aug_edge_type,
            )
    
    def save(self):
        """Save dataset with hybrid features."""
        saved_dir = self.saved_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        file_path = saved_dir + self.args.config_name + '.pkl'
        print('##########Save hybrid dataset in %s############' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


def main():
    parser = argparse.ArgumentParser(description='Hybrid GNN data preprocessing')
    parser.add_argument("--config", "-c", type=str, default='config/fb15k237_hybrid.yaml')
    parser.add_argument("--version", "-v", type=str, default='')
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load Config
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))

        if 'ind' in args.config:
            if not args.version:
                raise ValueError("Inductive config requires --version (e.g., v1).")
            cfg.dataset.version = args.version
        elif args.version:
            cfg.dataset.version = args.version
        elif not hasattr(cfg.dataset, 'version'):
            cfg.dataset.version = ''

    # Set Config Name
    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version') and cfg.dataset.version:
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name

    print('=' * 60)
    print('Hybrid GNN Preprocessing')
    print('=' * 60)
    print(f"Config file: {args.config}")
    print(f"Config name: {args.config_name}")
    print(f"Dataset version: {cfg.dataset.get('version', 'NOT SET')}")
    
    # Extract hybrid config
    hybrid_config = {}
    if hasattr(cfg, 'score_retriever') and cfg.score_retriever.get('use_hybrid', False):
        hybrid_cfg = cfg.score_retriever.get('hybrid', {})
        hybrid_config = {
            'k_semantic': hybrid_cfg.get('k_semantic', 10),
            'k_random': hybrid_cfg.get('k_random', 5),
            'hidden_dim': cfg.score_retriever.get('r', 32),
        }
        print(f"\nHybrid config:")
        for k, v in hybrid_config.items():
            print(f"  {k}: {v}")
    
    # Instantiate Dataset
    dataset_class_str = cfg.dataset.get('class', '')
    dataset_version = cfg.dataset.get('version', '')
    is_inductive = 'Inductive' in dataset_class_str or 'ind' in args.config

    if is_inductive and not dataset_version:
        raise ValueError("Inductive datasets need a version.")

    kgdata = None
    if 'FB15k237Inductive' in dataset_class_str:
        kgdata = FB15k237Inductive(version=dataset_version)
    elif 'WN18RRInductive' in dataset_class_str:
        kgdata = WN18RRInductive(version=dataset_version)
    elif 'FB15k237' in dataset_class_str:
        kgdata = FB15k237(version=dataset_version)
    elif 'WN18RR' in dataset_class_str:
        kgdata = WN18RR(version=dataset_version)
    else:
        raise ValueError(f"Unknown dataset class: {dataset_class_str}")

    print('\n***************Load tokenizer***************')
    tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    
    print('\n***************Create hybrid dataset***************')
    if is_inductive:
        dataset = HybridInductiveKGCDataset(args, kgdata, tokenizer, hybrid_config)
    else:
        dataset = HybridKGCDataset(args, kgdata, tokenizer, hybrid_config)
    
    print('\n✅ Preprocessing complete!')
    print(f"Dataset saved to: data/preprocessed/{args.config_name}.pkl")


if __name__ == "__main__":
    if not HYBRID_AVAILABLE:
        print("Error: hybrid_gnn module not available.")
        print("Please ensure all dependencies are installed:")
        print("  pip install torch torch-geometric faiss-cpu")
        exit(1)
    
    main()
