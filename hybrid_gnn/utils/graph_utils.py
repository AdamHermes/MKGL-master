"""
Graph Utilities for Hybrid GNN
==============================

Contains utility functions for graph processing:
- select_edges: Variadic top-k edge selection
- remove_easy_edges: Remove trivial edges during training
- to_undirected_with_inverse: Create bidirectional graphs


Date: January 2026
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def multikey_argsort(
    inputs: list,
    descending: bool = False,
    break_tie: bool = False
) -> torch.Tensor:
    """
    Sort by multiple keys with optional tie-breaking.
    
    Args:
        inputs: List of tensors to sort by (priority from last to first)
        descending: Sort in descending order
        break_tie: Add random perturbation to break ties
        
    Returns:
        Sorted indices
    """
    if break_tie:
        order = torch.randperm(len(inputs[0]), device=inputs[0].device)
    else:
        order = torch.arange(len(inputs[0]), device=inputs[0].device)
    
    for key in inputs[::-1]:
        index = key[order].argsort(stable=True, descending=descending)
        order = order[index]
    
    return order


def bincount(input: torch.Tensor, minlength: int = 0) -> torch.Tensor:
    """
    Efficient bincount with sorted input optimization.
    
    Args:
        input: Input tensor of indices
        minlength: Minimum length of output
        
    Returns:
        Counts for each index
    """
    if input.numel() == 0:
        return torch.zeros(minlength, dtype=torch.long, device=input.device)

    sorted_check = (input.diff() >= 0).all() if input.numel() > 1 else True
    if sorted_check:
        if minlength == 0:
            minlength = input.max() + 1
        range_tensor = torch.arange(minlength + 1, device=input.device)
        index = torch.bucketize(range_tensor, input)
        return index.diff()

    return input.bincount(minlength=minlength)


def variadic_topks(
    input: torch.Tensor,
    size: torch.Tensor,
    ks: torch.Tensor,
    largest: bool = True,
    break_tie: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variadic top-k selection for batched graphs.
    
    Args:
        input: Input scores
        size: Number of elements per graph
        ks: Number of elements to select per graph
        largest: Select largest values
        break_tie: Randomly break ties
        
    Returns:
        Selected values and their indices
    """
    index2sample = torch.repeat_interleave(size)
    if largest:
        index2sample = -index2sample
    
    order = multikey_argsort(
        (index2sample, input),
        descending=largest,
        break_tie=break_tie
    )

    range_tensor = torch.arange(ks.sum(), device=input.device)
    offset = (size - ks).cumsum(0) - size + ks
    range_tensor = range_tensor + offset.repeat_interleave(ks)
    index = order[range_tensor]

    return input[index], index


def to_undirected_with_inverse(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_relations: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert directed graph to undirected with inverse relations.
    
    Mimics TorchDrug's graph.undirected(add_inverse=True):
    1. Flips edges (h, t) -> (t, h)
    2. Creates inverse relations: r -> r + num_relations
    
    Args:
        edge_index: [2, E] edge indices
        edge_attr: [E] relation IDs
        num_relations: Total number of relation types
        
    Returns:
        new_edge_index: [2, 2E] with original + inverse edges
        new_edge_attr: [2E] with original + inverse relations
    """
    edge_index_inv = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_attr_inv = edge_attr + num_relations
    
    new_edge_index = torch.cat([edge_index, edge_index_inv], dim=1)
    new_edge_attr = torch.cat([edge_attr, edge_attr_inv], dim=0)
    
    return new_edge_index, new_edge_attr


class EdgeSelector:
    """
    Edge selection module for graph pruning.
    
    Selects top-k nodes by score, then top-k edges from those nodes.
    """
    
    def __init__(
        self,
        node_ratio: float = 0.3,
        degree_ratio: float = 1.0,
        test_node_ratio: float = 0.3,
        test_degree_ratio: float = 1.0,
        break_tie: bool = True,
    ):
        """
        Args:
            node_ratio: Fraction of nodes to keep (training)
            degree_ratio: Multiplier for edge count (training)
            test_node_ratio: Fraction of nodes to keep (testing)
            test_degree_ratio: Multiplier for edge count (testing)
            break_tie: Randomly break score ties
        """
        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio
        self.test_degree_ratio = test_degree_ratio
        self.break_tie = break_tie
        self._training = True
    
    def train(self, mode: bool = True):
        self._training = mode
        return self
    
    def eval(self):
        return self.train(False)
    
    def select_edges(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        score: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Select edges based on node scores.
        
        Process:
        1. Select top-k nodes per graph by score
        2. Keep edges where source is a selected node
        3. Select top-k of those edges by target score
        
        Args:
            edge_index: [2, E] edge tensor
            edge_attr: [E] edge attributes
            batch: [N] node-to-graph assignment
            score: [N] node scores
            num_graphs: Number of graphs in batch
            
        Returns:
            Edge indices to keep
        """
        node_ratio = self.node_ratio if self._training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self._training else self.test_degree_ratio
        
        num_nodes = score.shape[0]
        num_edges = edge_index.shape[1]
        
        num_nodes_per_graph = bincount(batch, minlength=num_graphs)
        
        ks = (num_nodes_per_graph.float() * node_ratio).long()
        ks = torch.clamp(ks, min=1)
        ks = torch.min(ks, num_nodes_per_graph)
        
        index = variadic_topks(score, num_nodes_per_graph, ks=ks, break_tie=self.break_tie)[1]
        node_in = index
        
        src_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        src_mask[node_in] = True
        
        edge_mask_in = src_mask[edge_index[0]]
        
        edge_batch = batch[edge_index[0][edge_mask_in]]
        num_edges_per_graph = bincount(edge_batch, minlength=num_graphs)
        
        es = (degree_ratio * ks.float() * (num_edges / num_nodes)).long()
        es = torch.clamp(es, min=1)
        
        if es.size(0) != num_edges_per_graph.size(0):
            es = (degree_ratio * ks.float() * (num_edges_per_graph.float() / num_nodes_per_graph.float().clamp(min=1))).long()
            es = torch.clamp(es, min=1)

        es = torch.min(es, num_edges_per_graph)

        valid_edge_indices = torch.nonzero(edge_mask_in).squeeze(-1)
        
        if valid_edge_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=edge_index.device)
        
        node_out = edge_index[1][valid_edge_indices]
        score_edge = score[node_out]
        
        if score_edge.numel() == 0:
            return valid_edge_indices
        
        final_edge_indices = variadic_topks(
            score_edge, 
            num_edges_per_graph, 
            ks=es, 
            break_tie=self.break_tie
        )[1]
        
        return valid_edge_indices[final_edge_indices]


def remove_easy_edges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    h_index: torch.Tensor,
    t_index: torch.Tensor,
    r_index: torch.Tensor,
    num_nodes: int,
    remove_one_hop: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove trivial/easy edges during training.
    
    Removes edges that directly connect query (h, r, t) triples
    to prevent information leakage.
    
    Args:
        edge_index: [2, E] edge tensor
        edge_attr: [E] relation IDs
        h_index: Head entity indices
        t_index: Tail entity indices  
        r_index: Relation indices
        num_nodes: Total number of nodes
        remove_one_hop: If True, remove all (h, t) edges regardless of relation
                       If False, only remove edges with matching relation
        
    Returns:
        Filtered edge_index and edge_attr
    """
    edge_rels = edge_attr.squeeze() if edge_attr.dim() > 1 else edge_attr
    num_rels = max(edge_rels.max().item() if edge_rels.numel() > 0 else 0, 
                   r_index.max().item() if r_index.numel() > 0 else 0) + 1

    if remove_one_hop:
        h_ext = torch.cat([h_index, t_index], dim=0)
        t_ext = torch.cat([t_index, h_index], dim=0)
        
        graph_hashes = edge_index[0].long() * num_nodes + edge_index[1].long()
        batch_hashes = h_ext.long() * num_nodes + t_ext.long()
    else:
        graph_hashes = (
            edge_index[0].long() * num_nodes + edge_index[1].long()
        ) * num_rels + edge_rels.long()
        batch_hashes = (
            h_index.long() * num_nodes + t_index.long()
        ) * num_rels + r_index.long()

    mask_to_remove = torch.isin(graph_hashes, batch_hashes)
    final_mask = ~mask_to_remove
    
    new_edge_index = edge_index[:, final_mask]
    new_edge_attr = edge_attr[final_mask]
    
    return new_edge_index, new_edge_attr


def negative_sample_to_tail(
    h_index: torch.Tensor,
    t_index: torch.Tensor,
    r_index: torch.Tensor,
    num_relations: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize negative samples to always predict tail.
    
    For head prediction samples, swap h and t and use inverse relation.
    
    Args:
        h_index: [B, K] head indices
        t_index: [B, K] tail indices
        r_index: [B, K] relation indices
        num_relations: Number of base relations (for computing inverse)
        
    Returns:
        Normalized (h, t, r) tuples
    """
    is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
    new_h_index = torch.where(is_t_neg, h_index, t_index)
    new_t_index = torch.where(is_t_neg, t_index, h_index)
    new_r_index = torch.where(is_t_neg, r_index, r_index + num_relations)
    return new_h_index, new_t_index, new_r_index


def create_subgraph_from_edges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    selected_edges: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a subgraph from selected edge indices.
    
    Args:
        edge_index: [2, E] original edge tensor
        edge_attr: [E] original edge attributes
        selected_edges: Indices of edges to keep
        num_nodes: Original number of nodes
        
    Returns:
        sub_edge_index: Subgraph edge index (remapped)
        sub_edge_attr: Subgraph edge attributes
        node_mapping: Mapping from new to old node indices
    """
    sub_edge_index = edge_index[:, selected_edges]
    sub_edge_attr = edge_attr[selected_edges]
    
    unique_nodes = torch.unique(sub_edge_index.flatten())
    node_mapping = unique_nodes
    
    inverse_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    inverse_mapping[unique_nodes] = torch.arange(len(unique_nodes), device=edge_index.device)
    
    remapped_edge_index = inverse_mapping[sub_edge_index]
    
    return remapped_edge_index, sub_edge_attr, node_mapping
