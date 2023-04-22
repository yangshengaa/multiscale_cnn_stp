"""
multilayer perceptron (FCNN)
"""

# load packages
from typing import List

import torch
import torch.nn as nn

class MLP(nn.Module):
    """vanilla multilayer perceptron"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, nl: nn.Module=nn.ReLU()) -> None:
        # tag
        self.input_dim = input_dim 
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.nl = nl

        # layers
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

        # hidden layers
        hidden_layers_list = []
        for i in range(1, len(self.hidden_dims)):
            hidden_layers_list.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            # insert nl
            hidden_layers_list.append(nl)
        self.hidden_layers = nn.Sequential(*hidden_layers_list)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.input_layer(X)
        X = self.nl(X)
        X = self.hidden_layers(X)
        X = self.output_layer(X)
        return X
