"""Invariant crystal GNNs for site-resolved tensor prediction (B3 and B4)."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.features.graph import GaussianSmearing
from src.features.tensor_transforms import (
    cartesian_5d_to_matrix,
    cartesian_6d_to_matrix,
    project_symmetric_traceless
)


class InvariantTensorGNN(nn.Module):
    """Invariant message-passing neural network for sitewise tensor prediction.

    Can operate in two modes:
    - mode="unconstrained_6d" (B3): Predicts 6 Cartesian components directly.
    - mode="symmetric_traceless_5d" (B4): Predicts 5 independent components,
      which are then projected to an exact symmetric-traceless 3x3 matrix.
    """
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_gaussians: int = 50,
        radius_cutoff: float = 5.0,
        mode: str = "symmetric_traceless_5d"
    ):
        super().__init__()
        self.mode = mode
        self.output_dim = 6 if mode == "unconstrained_6d" else 5

        self.atom_embed = nn.Embedding(119, node_dim)
        self.dist_expand = GaussianSmearing(0.0, radius_cutoff, num_gaussians)
        self.edge_proj = nn.Linear(num_gaussians, node_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "msg": nn.Sequential(
                    nn.Linear(2 * node_dim + node_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, node_dim)
                ),
                "node_update": nn.Sequential(
                    nn.Linear(2 * node_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, node_dim)
                )
            })
            for _ in range(num_layers)
        ])

        # Sitewise MLP predicting tensor components
        self.site_head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, self.output_dim)
        )

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
        - raw_output: [N_sites, output_dim] (5D or 6D)
        - full_matrix: [N_sites, 3, 3] reconstructed Cartesian matrix
        """
        h = self.atom_embed(batch.x)
        e = self.edge_proj(self.dist_expand(batch.edge_dist))
        src, dst = batch.edge_index

        for layer in self.layers:
            # Message passing
            msg_input = torch.cat([h[src], h[dst], e], dim=-1)
            msg = layer["msg"](msg_input)

            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)

            h = h + layer["node_update"](torch.cat([h, agg], dim=-1))

        out_comps = self.site_head(h)  # [N_sites, output_dim]

        if self.mode == "unconstrained_6d":
            # Reconstruct 6D into symmetric matrix
            full_matrix = cartesian_6d_to_matrix(out_comps)
        else:
            # Reconstruct 5D into exact symmetric traceless matrix
            full_matrix = cartesian_5d_to_matrix(out_comps)

        return out_comps, full_matrix
