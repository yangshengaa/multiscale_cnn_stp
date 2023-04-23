"""
CNN model
"""

# load packages
from typing import List

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_features=3, output_dim=3, scale=1) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, num_features))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1))
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 1))
        self.relu3 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc1 = nn.Linear(736, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x
