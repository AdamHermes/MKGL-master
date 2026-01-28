"""
Scorer - Scoring Module for KG Reasoning
=========================================

Computes compatibility scores between node embeddings and relation queries.


Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Scorer(nn.Module):
    """
    Scoring module for ranking tail predictions.
    
    Uses MLP with normalization for stable training.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(Scorer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim * 2
        
        self.linear = nn.Linear(self.feature_dim, hidden_dim)
        
        layers = []
        for i in range(num_mlp_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden: torch.Tensor,
        query: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute scores for node embeddings given a query.
        
        Args:
            hidden: Node embeddings
            query: Relation query
            normalize: Whether to normalize embeddings
        
        Returns:
            scores: Compatibility scores
        """
        if normalize:
            hidden = F.normalize(hidden, p=2, dim=-1)
            query = F.normalize(query, p=2, dim=-1)
        
        if hidden.dim() == 2 and query.dim() == 2:
            combined = torch.cat([hidden, query], dim=-1)
        elif hidden.dim() == 3:
            query_expanded = query.unsqueeze(1).expand(-1, hidden.size(1), -1)
            combined = torch.cat([hidden, query_expanded], dim=-1)
        else:
            combined = torch.cat([hidden, query], dim=-1)
        
        heuristic = self.linear(combined)
        heuristic = F.normalize(heuristic, p=2, dim=-1)
        
        score = self.mlp(hidden * heuristic).squeeze(-1)
        score = score * 10.0
        score = torch.clamp(score, min=-15, max=15)
        
        return score
