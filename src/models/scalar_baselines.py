"""Task A scalar baseline models: A0, A1, A2, A3, A4 (CGCNN), A5 (ALIGNN)."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.data import Batch

from src.features.descriptors import (
    extract_crystal_structural_features,
    extract_magpie_composition_features
)
from src.features.graph import GaussianSmearing


class MedianBaseline:
    """A0: Training-set median predictor."""
    def __init__(self):
        self.median_val: float = 0.0

    def fit(self, y_train: np.ndarray):
        self.median_val = float(np.median(y_train))
        return self

    def predict(self, num_samples: int) -> np.ndarray:
        return np.full(num_samples, self.median_val, dtype=np.float32)


class TabularBaselines:
    """A1, A2, A3: Composition and structural descriptors with RF / XGBoost / GBDT."""
    def __init__(self, model_type: str = "rf", **kwargs):
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 15),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            )
        elif model_type == "xgb":
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_depth=kwargs.get("max_depth", 6),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            )
        elif model_type == "gbdt":
            self.model = HistGradientBoostingRegressor(
                max_iter=kwargs.get("n_estimators", 200),
                learning_rate=kwargs.get("learning_rate", 0.05),
                max_depth=kwargs.get("max_depth", 6),
                random_state=kwargs.get("random_state", 42)
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class CGCNN(nn.Module):
    """A4: Crystal Graph Convolutional Neural Network for crystal-level scalar prediction."""
    def __init__(
        self,
        orig_atom_fea_len: int = 119,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        num_gaussians: int = 50,
        radius_cutoff: float = 5.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(orig_atom_fea_len, atom_fea_len)
        self.distance_expansion = GaussianSmearing(0.0, radius_cutoff, num_gaussians)

        self.convs = nn.ModuleList([
            CGConv(channels=atom_fea_len, dim=num_gaussians, batch_norm=True)
            for _ in range(n_conv)
        ])

        self.fc_pool = nn.Linear(atom_fea_len, h_fea_len)
        self.fc_out = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.SiLU(),
            nn.Linear(h_fea_len // 2, 1)
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        # Node embeddings
        h = self.embedding(batch.x)  # [N_nodes, atom_fea_len]
        edge_attr = self.distance_expansion(batch.edge_dist)  # [E, num_gaussians]

        for conv in self.convs:
            h = conv(h, batch.edge_index, edge_attr)
            h = F.silu(h)

        # Global crystal pooling
        crystal_fea = global_mean_pool(h, batch.batch)  # [B, atom_fea_len]
        crystal_fea = F.silu(self.fc_pool(crystal_fea))
        out = self.fc_out(crystal_fea).squeeze(-1)  # [B]
        return out


class ALIGNNEquivalent(nn.Module):
    """A5: Atomistic line graph neural network architecture incorporating edge and line graph representations."""
    def __init__(
        self,
        atom_dim: int = 64,
        edge_dim: int = 64,
        num_layers: int = 4,
        num_gaussians: int = 50,
        radius_cutoff: float = 5.0
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(119, atom_dim)
        self.dist_expand = GaussianSmearing(0.0, radius_cutoff, num_gaussians)
        self.edge_proj = nn.Linear(num_gaussians, edge_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "node_update": nn.Sequential(
                    nn.Linear(atom_dim + edge_dim, atom_dim),
                    nn.BatchNorm1d(atom_dim),
                    nn.SiLU(),
                    nn.Linear(atom_dim, atom_dim)
                ),
                "edge_update": nn.Sequential(
                    nn.Linear(edge_dim + 2 * atom_dim, edge_dim),
                    nn.BatchNorm1d(edge_dim),
                    nn.SiLU(),
                    nn.Linear(edge_dim, edge_dim)
                )
            }))

        self.fc = nn.Sequential(
            nn.Linear(atom_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        h = self.atom_embed(batch.x)
        e = self.edge_proj(self.dist_expand(batch.edge_dist))

        src, dst = batch.edge_index

        for layer in self.layers:
            # Edge update incorporating connected node states
            e_cat = torch.cat([e, h[src], h[dst]], dim=-1)
            e = e + layer["edge_update"](e_cat)

            # Node aggregation: scatter mean of incoming edge states
            # Vectorized aggregation using index_add
            agg_e = torch.zeros_like(h)
            agg_e.index_add_(0, dst, e)
            h_cat = torch.cat([h, agg_e], dim=-1)
            h = h + layer["node_update"](h_cat)

        crystal_rep = global_mean_pool(h, batch.batch)
        return self.fc(crystal_rep).squeeze(-1)
