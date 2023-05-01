"""
CNN model
"""

# load packages
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, window_size=100, n_features=3, n_filters=32, output_dim=3, scale=1, nl=nn.ReLU()) -> None:
        super(CNN, self).__init__()
        self.scale = scale
        self.window_size = window_size
        self.nl = nl

        conv1_list = []
        conv2_list = []
        conv3_list = []
        for s in range(1, scale+1):
            conv1_list.append(nn.Conv2d(1, n_filters//2, kernel_size=(1, n_features)))
            conv2_list.append(nn.Conv2d(n_filters//2, n_filters, kernel_size=(3, 1), padding='same'))
            conv3_list.append(nn.Conv2d(n_filters, n_filters, kernel_size=(3, 1), padding='same'))

        self.conv1_list = nn.ModuleList(conv1_list)
        self.conv2_list = nn.ModuleList(conv2_list)
        self.conv3_list = nn.ModuleList(conv3_list)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc = nn.Linear(scale*n_filters, output_dim)
        self.gru = nn.GRU(scale*n_filters, scale*n_filters, batch_first=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        feature_list = []
        X = torch.unsqueeze(X, 1)
        for s in range(self.scale):
            indices = [i for i in range(self.window_size) if i % (s+1) == 0]
            x = X[:, :, indices, :]
            x = self.nl(self.conv1_list[s](x))
            x = self.nl(self.conv2_list[s](x))
            x = self.pool(x)
            x = self.nl(self.conv3_list[s](x))
            x = self.pool(x)
            x = x.squeeze()
            if s == 0:
                L = x.shape[-1]
            else:
                x = F.pad(x, (L-x.shape[-1], 0))
            feature_list.append(x)
        x = torch.cat(feature_list, 1)
        x = x.permute(0, 2, 1)
        x = self.gru(x)[1].squeeze()
        x = self.fc(x)
        return x
