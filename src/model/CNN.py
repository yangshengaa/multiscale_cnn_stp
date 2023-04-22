"""
CNN model
"""

# load packages
from typing import List

import torch
import torch.nn as nn

class CNN(nn.Module):
    # TODO: what parameters to pass in? 
    def __init__(self, _) -> None:
        super().__init__()
        raise NotImplementedError()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO:
        # make sure to return raw logits (i.e. results before softmax). 
        # We will let CrossEntropyLoss compute softmax for us
        raise NotImplementedError()
