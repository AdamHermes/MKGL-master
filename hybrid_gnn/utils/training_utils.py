"""
Training Utilities for Hybrid GNN
=================================

Contains loss functions and evaluation metrics for training.


Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for knowledge graph embedding.
    
    Uses InfoNCE-style loss with temperature scaling.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.1,
    ):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        scores: torch.Tensor,
        positive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            scores: [B, K] where K includes 1 positive + (K-1) negatives
            positive_mask: Optional mask for positive samples (default: first column)
            
        Returns:
            Loss value
        """
        scores = scores / self.temperature
        
        if positive_mask is None:
            targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores, targets)
        else:
            pos_scores = (scores * positive_mask.float()).sum(dim=-1)
            neg_scores = scores * (~positive_mask).float()
            neg_scores = neg_scores.masked_fill(positive_mask, float('-inf'))
            
            all_scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
            targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(all_scores, targets)
        
        return loss


class HybridLoss(nn.Module):
    """
    Combined loss for hybrid GNN training.
    
    Includes:
    - Main contrastive/ranking loss
    - Regularization terms
    - Optional auxiliary losses
    """
    
    def __init__(
        self,
        main_weight: float = 1.0,
        reg_weight: float = 0.01,
        diversity_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super(HybridLoss, self).__init__()
        self.main_weight = main_weight
        self.reg_weight = reg_weight
        self.diversity_weight = diversity_weight
        self.contrastive = ContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        scores: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            scores: [B, K] prediction scores
            embeddings: Optional embeddings for regularization
            attention_weights: Optional attention weights for diversity loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        main_loss = self.contrastive(scores)
        loss_dict['main'] = main_loss
        
        total_loss = self.main_weight * main_loss
        
        if embeddings is not None and self.reg_weight > 0:
            reg_loss = torch.mean(embeddings ** 2)
            loss_dict['reg'] = reg_loss
            total_loss = total_loss + self.reg_weight * reg_loss
        
        if attention_weights is not None and self.diversity_weight > 0:
            attn_flat = attention_weights.view(attention_weights.shape[0], -1)
            entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10), dim=-1)
            diversity_loss = -entropy.mean()
            loss_dict['diversity'] = diversity_loss
            total_loss = total_loss + self.diversity_weight * diversity_loss
        
        loss_dict['total'] = total_loss
        return total_loss, loss_dict


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss.
    
    Encourages positive samples to score higher than negatives.
    """
    
    def __init__(self, margin: float = 0.0):
        super(BPRLoss, self).__init__()
        self.margin = margin
    
    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BPR loss.
        
        Args:
            pos_scores: [B] or [B, 1] positive scores
            neg_scores: [B, K] negative scores
            
        Returns:
            Loss value
        """
        if pos_scores.dim() == 1:
            pos_scores = pos_scores.unsqueeze(-1)
        
        diff = pos_scores - neg_scores - self.margin
        loss = -F.logsigmoid(diff).mean()
        
        return loss


def compute_ranking_metrics(
    scores: torch.Tensor,
    target_indices: Optional[torch.Tensor] = None,
    ks: Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, float]:
    """
    Compute ranking metrics: Hits@K, MRR, MR.
    
    Args:
        scores: [B, N] prediction scores for all candidates
        target_indices: [B] indices of correct answers (default: 0)
        ks: Tuple of K values for Hits@K
        
    Returns:
        Dictionary of metrics
    """
    batch_size = scores.shape[0]
    
    if target_indices is None:
        target_indices = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
    
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    
    ranks = torch.zeros(batch_size, device=scores.device)
    for i in range(batch_size):
        rank_pos = (sorted_indices[i] == target_indices[i]).nonzero()
        if rank_pos.numel() > 0:
            ranks[i] = rank_pos[0].item() + 1
        else:
            ranks[i] = scores.shape[1]
    
    metrics = {}
    
    for k in ks:
        hits_k = (ranks <= k).float().mean().item()
        metrics[f'hits@{k}'] = hits_k
    
    mrr = (1.0 / ranks).mean().item()
    metrics['mrr'] = mrr
    
    mr = ranks.mean().item()
    metrics['mr'] = mr
    
    return metrics


def compute_hits_at_k_fast(
    scores: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Fast Hits@K computation assuming target is at index 0.
    
    Args:
        scores: [B, N] where column 0 is the positive
        k: K value for Hits@K
        
    Returns:
        Hits@K value
    """
    pos_scores = scores[:, 0:1]
    neg_scores = scores[:, 1:]
    
    ranks = (neg_scores >= pos_scores).sum(dim=-1) + 1
    
    hits_k = (ranks <= k).float().mean().item()
    return hits_k


def compute_mrr_fast(scores: torch.Tensor) -> float:
    """
    Fast MRR computation assuming target is at index 0.
    
    Args:
        scores: [B, N] where column 0 is the positive
        
    Returns:
        MRR value
    """
    pos_scores = scores[:, 0:1]
    neg_scores = scores[:, 1:]
    
    ranks = (neg_scores >= pos_scores).sum(dim=-1) + 1
    
    mrr = (1.0 / ranks.float()).mean().item()
    return mrr


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for classification.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: [B, C] prediction logits
            targets: [B] target class indices
            
        Returns:
            Loss value
        """
        num_classes = logits.shape[-1]
        
        one_hot = F.one_hot(targets, num_classes).float()
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_one_hot * log_probs).sum(dim=-1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: [B, C] prediction logits
            targets: [B] target class indices
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()
