"""E(3)-Equivariant Graph Neural Network (B5) and Site-Symmetry Projection (B6)."""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import e3nn.o3 as o3
from torch_geometric.data import Batch

from src.features.graph import GaussianSmearing
from src.features.tensor_transforms import irrep_2e_to_matrix


class EquivariantMessagePassingLayer(nn.Module):
    """Equivariant interaction layer utilizing spherical harmonics and tensor products."""
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        num_gaussians: int = 50
    ):
        super().__init__()
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            shared_weights=False
        )
        self.fc_radial = nn.Sequential(
            nn.Linear(num_gaussians, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel)
        )
        self.self_conn = o3.Linear(irreps_in, irreps_out)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rad: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        weights = self.fc_radial(edge_rad)  # [E, weight_numel]

        # Tensor product message
        msg = self.tp(h[src], edge_sh, weights)  # [E, irreps_out.dim]

        # Aggregate messages at destination nodes
        out = torch.zeros(h.shape[0], msg.shape[-1], dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, msg)

        # Self-connection
        out = out + self.self_conn(h)
        return out


class EquivariantTensorGNN(nn.Module):
    """B5: E(3)-Equivariant Graph Neural Network predicting l=2 site-resolved EFG tensors."""
    def __init__(
        self,
        node_embed_dim: int = 32,
        irreps_hidden: str = "32x0e + 16x1o + 8x2e",
        num_layers: int = 3,
        num_gaussians: int = 50,
        radius_cutoff: float = 5.0
    ):
        super().__init__()
        self.radius_cutoff = radius_cutoff
        self.atom_embed = nn.Embedding(119, node_embed_dim)

        self.irreps_node_in = o3.Irreps(f"{node_embed_dim}x0e")
        self.irreps_sh = o3.Irreps("0e + 1o + 2e")
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps("1x2e")

        self.dist_expand = GaussianSmearing(0.0, radius_cutoff, num_gaussians)

        # Initial linear projection to hidden irreps
        self.in_proj = o3.Linear(self.irreps_node_in, self.irreps_hidden)

        # Message passing layers
        self.layers = nn.ModuleList([
            EquivariantMessagePassingLayer(
                irreps_in=self.irreps_hidden,
                irreps_sh=self.irreps_sh,
                irreps_out=self.irreps_hidden,
                num_gaussians=num_gaussians
            )
            for _ in range(num_layers)
        ])

        # Output head projecting to 1x2e (5 components)
        self.out_head = o3.Linear(self.irreps_hidden, self.irreps_out)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
        - v2e: [N_sites, 5] l=2 spherical tensor components
        - V_matrix: [N_sites, 3, 3] Cartesian symmetric traceless matrix
        """
        # Node features
        h = self.in_proj(self.atom_embed(batch.x))

        # Spherical harmonics of normalized edge vectors
        edge_vec = batch.edge_vec
        dist = batch.edge_dist.clamp(min=1e-6)
        edge_dir = edge_vec / dist.unsqueeze(-1)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_dir, normalize=True)
        edge_rad = self.dist_expand(batch.edge_dist)

        # Equivariant message passing
        for layer in self.layers:
            h = layer(h, batch.edge_index, edge_sh, edge_rad)

        # Project to l=2 irrep (5 components)
        v2e = self.out_head(h)  # [N_sites, 5]

        # Reconstruct full 3x3 Cartesian tensor
        V_matrix = irrep_2e_to_matrix(v2e)  # [N_sites, 3, 3]

        return v2e, V_matrix


def apply_site_symmetry_projection(
    V: torch.Tensor,
    symmetry_matrices: List[torch.Tensor]
) -> torch.Tensor:
    """B6: Projects EFG tensor onto the invariant subspace of the site point group:
    V_sym = 1/|G| sum_{R in G} R @ V @ R^T
    """
    if not symmetry_matrices:
        return V

    V_proj = torch.zeros_like(V)
    for R in symmetry_matrices:
        R_t = R.to(dtype=V.dtype, device=V.device)
        V_proj = V_proj + R_t @ V @ R_t.transpose(-1, -2)

    return V_proj / len(symmetry_matrices)
