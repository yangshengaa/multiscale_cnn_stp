"""
PyTorch Dataset Generator
"""

# load packages
import os 
import numpy as np

import torch
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    """time series dataset generator"""
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str='cpu') -> None:
        # store everything in the same device (only parts of the data)
        self.X = torch.Tensor(X).to(device)
        self.y = torch.Tensor(y).to(device).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index) -> torch.Tensor:
        return self.X[index], self.y[index]
