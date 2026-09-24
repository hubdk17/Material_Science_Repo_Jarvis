"""Site-local MLP baseline predicting 5D symmetric traceless EFG tensors from local descriptors."""

import torch
import torch.nn as nn
from src.features.tensor_transforms import cartesian_5d_to_matrix


class SiteLocalMLP(nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 128, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor):
        out_5d = self.net(x)
        V_matrix = cartesian_5d_to_matrix(out_5d)
        return out_5d, V_matrix
