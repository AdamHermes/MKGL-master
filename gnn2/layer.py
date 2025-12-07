import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='relu', dropout=0.0):
        super(MLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # If hidden_dims is a single int, convert to list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
            
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Apply activation and dropout to all layers except the last one
            if i < len(hidden_dims) - 1:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)