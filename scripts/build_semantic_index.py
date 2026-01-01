"""
Offline Semantic Neighbor Index Builder for Dynamic Semantic Attention Module.

This script:
1. Loads the preprocessed dataset with entity embeddings
2. Extracts PNA-aggregated text embeddings for all entities
3. Builds a FAISS index for efficient similarity search
4. Queries top-K neighbors per entity, filtering by threshold and structural neighbors
5. Saves sparse semantic adjacency matrix (COO format)

Usage:
    python scripts/build_semantic_index.py --config config/fb15k237.yaml --top_k 100 --threshold 0.5
"""

import argparse
import os
import sys

# Add parent directory to path to import preprocess_new
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import dataset classes at module level (BEFORE pickle.load)
# These need to be added to __main__ namespace because pickle saved them with __main__ module
try:
    from preprocess_new import InductiveKGCDataset, KGCDataset, Prompter
    from dataset_new import InductiveKnowledgeGraphDataset, StandardKGCDataset, FB15k237Inductive, WN18RRInductive, FB15k237, WN18RR
    
    # Add to __main__ namespace so pickle can find them
    import __main__
    __main__.InductiveKGCDataset = InductiveKGCDataset
    __main__.KGCDataset = KGCDataset
    __main__.Prompter = Prompter
    __main__.InductiveKnowledgeGraphDataset = InductiveKnowledgeGraphDataset
    __main__.StandardKGCDataset = StandardKGCDataset
    __main__.FB15k237Inductive = FB15k237Inductive
    __main__.WN18RRInductive = WN18RRInductive
    __main__.FB15k237 = FB15k237
    __main__.WN18RR = WN18RR
except ImportError as e:
    print(f"Error importing dataset classes: {e}")
    print(f"Make sure preprocess_new.py and dataset_new.py are in: {parent_dir}")
    sys.exit(1)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import easydict
import pickle
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not installed. Will use brute-force search (slower).")


def aggregate_text_pna(token_ids, text_embeddings):
    """
    PNA-style aggregation for multi-token entity names.
    Mirrors BasePNARetriever.aggregate_text() in retriever.py
    
    Args:
        token_ids: [num_entities, max_tokens] - token IDs per entity
        text_embeddings: [vocab_size, hidden_dim] - LLM token embeddings
    
    Returns:
        [num_entities, hidden_dim * 12] - PNA aggregated features
    """
    device = text_embeddings.device
    token_ids = token_ids.to(device)
    
    # Mask for valid tokens (non-padding)
    token_mask = (token_ids > 0).unsqueeze(-1).float()  # [N, L, 1]
    token_lengths = token_mask.sum(dim=1).clamp(min=1)  # [N, 1]
    
    # Get token embeddings
    token_embs = text_embeddings[token_ids]  # [N, L, H]
    
    # PNA aggregations
    mean = (token_embs * token_mask).sum(dim=1) / token_lengths
    sq_mean = (token_embs**2 * token_mask).sum(dim=1) / token_lengths
    max_val, _ = (token_embs * token_mask + (1 - token_mask) * -1e9).max(dim=1)
    min_val, _ = (token_embs * token_mask + (1 - token_mask) * 1e9).min(dim=1)
    std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
    
    features = torch.cat([mean, max_val, min_val, std], dim=-1)  # [N, H*4]
    
    # Degree scaling
    scale = token_lengths.log().clamp(min=1e-6)
    scale = scale / scale.mean()
    ones = torch.ones_like(scale)
    scales = torch.cat([ones, scale, 1 / scale.clamp(min=1e-2)], dim=-1)  # [N, 3]
    
    # Combine features with scales
    result = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)  # [N, H*12]
    
    return result


def get_structural_neighbors(graph, num_entities):
    """
    Build set of 1-hop structural neighbors for each entity.
    
    Args:
        graph: PyG Data object with edge_index
        num_entities: Number of entities
    
    Returns:
        dict: {entity_id: set of neighbor_ids}
    """
    edge_index = graph.edge_index.cpu().numpy()
    neighbors = {i: set() for i in range(num_entities)}
    
    for src, tgt in zip(edge_index[0], edge_index[1]):
        if src < num_entities and tgt < num_entities:
            neighbors[src].add(tgt)
            neighbors[tgt].add(src)  # Undirected
    
    return neighbors


def build_faiss_index(embeddings):
    """
    Build FAISS index for cosine similarity search.
    
    Args:
        embeddings: [num_entities, dim] normalized embeddings
    
    Returns:
        FAISS index
    """
    dim = embeddings.shape[1]
    
    if FAISS_AVAILABLE:
        # Use Inner Product index (cosine similarity for normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
    else:
        # Fallback: store embeddings for brute-force search
        index = embeddings
    
    return index


def search_neighbors(index, query_embeddings, top_k):
    """
    Search for top-K nearest neighbors.
    
    Args:
        index: FAISS index or numpy array
        query_embeddings: [num_queries, dim]
        top_k: Number of neighbors to retrieve
    
    Returns:
        scores: [num_queries, top_k] - similarity scores
        indices: [num_queries, top_k] - neighbor indices
    """
    if FAISS_AVAILABLE:
        scores, indices = index.search(query_embeddings.astype(np.float32), top_k)
    else:
        # Brute-force cosine similarity
        scores = query_embeddings @ index.T  # [N, N]
        indices = np.argsort(-scores, axis=1)[:, :top_k]
        scores = np.take_along_axis(scores, indices, axis=1)
    
    return scores, indices


def filter_neighbors(entity_id, neighbor_indices, neighbor_scores, 
                     structural_neighbors, threshold):
    """
    Filter neighbors: remove self, structural neighbors, and below-threshold.
    
    Args:
        entity_id: Current entity
        neighbor_indices: Array of neighbor IDs
        neighbor_scores: Array of similarity scores
        structural_neighbors: Set of 1-hop neighbors
        threshold: Minimum similarity threshold
    
    Returns:
        filtered_indices, filtered_scores
    """
    filtered_idx = []
    filtered_scores = []
    
    for idx, score in zip(neighbor_indices, neighbor_scores):
        # Skip self
        if idx == entity_id:
            continue
        # Skip structural neighbors
        if idx in structural_neighbors:
            continue
        # Skip below threshold
        if score < threshold:
            continue
        
        filtered_idx.append(idx)
        filtered_scores.append(score)
    
    return filtered_idx, filtered_scores


def main():
    parser = argparse.ArgumentParser(description='Build semantic neighbor index')
    parser.add_argument('--config', '-c', type=str, default='config/fb15k237.yaml',
                        help='Config file path')
    parser.add_argument('--version', '-v', type=str, default='',
                        help='Dataset version (e.g., v1 for inductive datasets)')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of top neighbors to retrieve before filtering')
    parser.add_argument('--max_neighbors', type=int, default=100,
                        help='Maximum neighbors to keep after filtering')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum cosine similarity threshold')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: data/semantic_neighbors_{config_name}.pt)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for embedding extraction')
    parser.add_argument('--inductive', action='store_true',
                        help='Build index for inductive test entities (instead of transductive)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
    
    # Handle version from command line or config
    if args.version:
        cfg.dataset.version = args.version
    
    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version') and cfg.dataset.version:
        config_name += '_' + cfg.dataset.version
    
    # Add suffix for inductive index
    index_suffix = '_ind' if args.inductive else ''
    
    print(f"Building semantic index for: {config_name}{index_suffix}")
    print(f"Mode: {'INDUCTIVE (test entities)' if args.inductive else 'TRANSDUCTIVE (train/valid entities)'}")
    print(f"Parameters: top_k={args.top_k}, threshold={args.threshold}, max_neighbors={args.max_neighbors}")
    
    # Load preprocessed dataset
    saved_dir = 'data/preprocessed/'
    file_path = os.path.join(saved_dir, config_name + '.pkl')
    
    if not os.path.exists(file_path):
        print(f"Error: Preprocessed dataset not found at {file_path}")
        print("Please run preprocessing first:")
        print(f"  python preprocess_new.py --config {args.config}" + (f" --version {args.version}" if args.version else ""))
        sys.exit(1)
    
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Get entity token IDs (kgl2token format)
    vocab_df = dataset.vocab_df
    
    # For inductive mode, filter to only inductive entities
    # For transductive mode, filter to only transductive entities
    if args.inductive:
        # Inductive entities have transductive=0 and entity=1
        entity_mask = (vocab_df['entity'] == 1) & (vocab_df['transductive'] == 0)
        print("Using INDUCTIVE entities (for test split)")
    else:
        # Transductive entities have transductive=1 and entity=1
        entity_mask = (vocab_df['entity'] == 1) & (vocab_df['transductive'] == 1)
        print("Using TRANSDUCTIVE entities (for train/valid split)")
    
    entity_df = vocab_df[entity_mask].copy()
    
    # Reset index to get contiguous entity IDs (0 to num_entities-1)
    entity_df = entity_df.reset_index(drop=False)  # Keep original token_index as column
    entity_df['local_id'] = range(len(entity_df))  # New local IDs
    
    num_entities = len(entity_df)
    print(f"Number of entities: {num_entities}")
    
    # Extract text token IDs for entities
    kgl_token_length = cfg.get('kgl_token_length', 10)
    text_token_ids = np.stack(entity_df['text_token_ids'].values)[:, :kgl_token_length]
    text_token_ids = torch.tensor(text_token_ids, dtype=torch.long)
    
    print(f"Text token IDs shape: {text_token_ids.shape}")
    
    # Load LLM embeddings
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = cfg.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    print(f"Loading LLM embeddings from {model_name}...")
    
    # Just load the embedding layer, not the full model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    text_embeddings = model.get_input_embeddings().weight.data.float()
    
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Compute PNA-aggregated entity embeddings
    print("Computing PNA-aggregated entity embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_embeddings = text_embeddings.to(device)
    
    # Process in batches to avoid OOM
    all_embeddings = []
    for i in tqdm(range(0, num_entities, args.batch_size)):
        batch_ids = text_token_ids[i:i+args.batch_size].to(device)
        with torch.no_grad():
            batch_embs = aggregate_text_pna(batch_ids, text_embeddings)
        all_embeddings.append(batch_embs.cpu())
    
    entity_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Entity embeddings shape: {entity_embeddings.shape}")
    
    # Normalize for cosine similarity
    entity_embeddings = F.normalize(entity_embeddings, p=2, dim=1)
    entity_embeddings_np = entity_embeddings.numpy()
    
    # Build structural neighbor sets
    print("Building structural neighbor sets...")
    # Use appropriate graph based on mode
    if args.inductive:
        # For inductive, use inductive_fact_graph if available
        if hasattr(dataset.kgdata, 'inductive_fact_graph'):
            graph = dataset.kgdata.inductive_fact_graph
            print("Using inductive_fact_graph for structural neighbors")
        else:
            graph = dataset.kgdata.fact_graph
            print("Warning: inductive_fact_graph not found, using fact_graph")
    else:
        graph = dataset.kgdata.fact_graph
        print("Using fact_graph for structural neighbors")
    
    structural_neighbors = get_structural_neighbors(graph, num_entities)
    
    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(entity_embeddings_np)
    
    # Search and filter neighbors
    print("Searching for semantic neighbors...")
    # Retrieve more than needed to account for filtering
    search_k = min(args.top_k + 50, num_entities)  # Extra buffer for filtering
    scores, indices = search_neighbors(index, entity_embeddings_np, search_k)
    
    # Filter and build sparse adjacency
    print("Filtering and building sparse adjacency...")
    all_src = []
    all_dst = []
    all_scores = []
    
    neighbor_counts = []
    
    for entity_id in tqdm(range(num_entities)):
        filtered_idx, filtered_scores = filter_neighbors(
            entity_id,
            indices[entity_id],
            scores[entity_id],
            structural_neighbors[entity_id],
            args.threshold
        )
        
        # Keep only max_neighbors
        filtered_idx = filtered_idx[:args.max_neighbors]
        filtered_scores = filtered_scores[:args.max_neighbors]
        
        neighbor_counts.append(len(filtered_idx))
        
        for neighbor_id, score in zip(filtered_idx, filtered_scores):
            all_src.append(entity_id)
            all_dst.append(neighbor_id)
            all_scores.append(score)
    
    # Statistics
    neighbor_counts = np.array(neighbor_counts)
    print(f"\nNeighbor statistics:")
    print(f"  Mean neighbors per entity: {neighbor_counts.mean():.2f}")
    print(f"  Median neighbors per entity: {np.median(neighbor_counts):.2f}")
    print(f"  Min neighbors: {neighbor_counts.min()}")
    print(f"  Max neighbors: {neighbor_counts.max()}")
    print(f"  Entities with 0 neighbors: {(neighbor_counts == 0).sum()}")
    print(f"  Total semantic edges: {len(all_src)}")
    
    # Save as sparse COO format
    semantic_adj = {
        'src': torch.tensor(all_src, dtype=torch.long),
        'dst': torch.tensor(all_dst, dtype=torch.long),
        'scores': torch.tensor(all_scores, dtype=torch.float),
        'num_entities': num_entities,
        'threshold': args.threshold,
        'max_neighbors': args.max_neighbors,
    }
    
    # Also save dense format for easy lookup (padded)
    # Shape: [num_entities, max_k] for indices, [num_entities, max_k] for scores
    max_k = args.max_neighbors
    dense_indices = torch.full((num_entities, max_k), -1, dtype=torch.long)
    dense_scores = torch.zeros((num_entities, max_k), dtype=torch.float)
    
    current_idx = 0
    for entity_id in range(num_entities):
        count = neighbor_counts[entity_id]
        if count > 0:
            dense_indices[entity_id, :count] = torch.tensor(all_dst[current_idx:current_idx+count])
            dense_scores[entity_id, :count] = torch.tensor(all_scores[current_idx:current_idx+count])
        current_idx += count
    
    semantic_adj['dense_indices'] = dense_indices
    semantic_adj['dense_scores'] = dense_scores
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        os.makedirs('data', exist_ok=True)
        output_path = f'data/semantic_neighbors_{config_name}{index_suffix}.pt'
    
    print(f"\nSaving to {output_path}...")
    torch.save(semantic_adj, output_path)
    print("Done!")


if __name__ == '__main__':
    main()
