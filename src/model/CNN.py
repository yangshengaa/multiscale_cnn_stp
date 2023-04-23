"""
CNN model
"""

# load packages
from typing import List

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, window_size=100, num_features=3, output_dim=3, scale=1) -> None:
        super(CNN, self).__init__()
        self.scale = scale
        self.window_size = window_size

        conv1_list = []
        conv2_list = []
        conv3_list = []
        for s in range(1, scale+1):
            conv1_list.append(nn.Conv2d(1, 16, kernel_size=(1, num_features), dilation=(s, 1)))
            conv2_list.append(nn.Conv2d(16, 32, kernel_size=(3, 1), padding='same'))
            conv3_list.append(nn.Conv2d(32, 32, kernel_size=(3, 1), padding='same'))

        self.conv1_list = conv1_list
        self.conv2_list = conv2_list
        self.conv3_list = conv3_list
        self.fc = nn.LazyLinear(output_dim)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        feature_list = []
        X = torch.unsqueeze(X, 1)
        for s in range(self.scale):
            indices = [i for i in range(self.window_size) if i % (s+1) == 0]
            x = X[:, :, indices, :]
            x = self.activation(self.conv1_list[s](x))
            x = self.activation(self.conv2_list[s](x))
            x = self.pool(x)
            x = self.activation(self.conv3_list[s](x))
            x = self.pool(x)
            x = x.view(x.shape[0], -1)
            feature_list.append(x)
        x = torch.cat(feature_list, 1)
        x = self.fc(x)
        return x
