"""Periodic crystal graph construction for invariant and equivariant neural networks."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data

# Atomic number mapping
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm"
]
ELEMENT_TO_Z = {el: i + 1 for i, el in enumerate(ELEMENTS)}


class GaussianSmearing(torch.nn.Module):
    """Expands interatomic distances into Gaussian radial basis functions."""
    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def build_periodic_crystal_graph(
    lattice_mat: np.ndarray,
    fractional_coords: np.ndarray,
    elements: List[str],
    radius_cutoff: float = 5.0,
    max_neighbors: int = 16,
    target_tensor_st: Optional[np.ndarray] = None,
    target_scalar: Optional[float] = None
) -> Data:
    """Constructs a periodic crystal graph using minimum image convention.

    Parameters:
    - lattice_mat: 3x3 array where rows are lattice vectors a, b, c
    - fractional_coords: Nx3 array of atomic positions in fractional coords
    - elements: length-N list of element symbols
    - radius_cutoff: neighbor search cutoff in Angstroms
    - max_neighbors: maximum neighbors per node
    - target_tensor_st: optional Nx5 or Nx3x3 ground truth symmetric traceless tensor
    - target_scalar: optional crystal-level target scalar (e.g. max_efg)

    Returns:
    - PyG Data object with:
      - x: [N] atomic numbers
      - pos: [N, 3] Cartesian positions
      - edge_index: [2, E]
      - edge_dist: [E] distances
      - edge_vec: [E, 3] displacement vectors r_ij = r_j - r_i
      - y_tensor: [N, 5] (optional)
      - y_scalar: float (optional)
    """
    N = len(elements)
    atomic_numbers = torch.tensor([ELEMENT_TO_Z.get(el, 0) for el in elements], dtype=torch.long)

    lat = np.array(lattice_mat, dtype=np.float64)
    frac = np.array(fractional_coords, dtype=np.float64) % 1.0
    cart_pos = frac @ lat  # [N, 3]

    # Calculate bounding box for periodic replicas
    # Inverse lengths of reciprocal lattice vectors determine replica bounds
    recip_lat = np.linalg.inv(lat)
    plane_dist = 1.0 / np.linalg.norm(recip_lat, axis=0)
    max_replicas = np.ceil(radius_cutoff / plane_dist).astype(int) + 1

    # Generate all periodic translation vectors
    rx = np.arange(-max_replicas[0], max_replicas[0] + 1)
    ry = np.arange(-max_replicas[1], max_replicas[1] + 1)
    rz = np.arange(-max_replicas[2], max_replicas[2] + 1)
    grid = np.meshgrid(rx, ry, rz, indexing="ij")
    translation_indices = np.stack(grid, axis=-1).reshape(-1, 3)

    # Replicated coordinates in Cartesian space
    translations = translation_indices @ lat  # [M, 3]
    num_translations = translations.shape[0]

    src_nodes: List[int] = []
    dst_nodes: List[int] = []
    distances: List[float] = []
    disp_vectors: List[np.ndarray] = []

    # Vectorized neighbor search per atom
    for i in range(N):
        pos_i = cart_pos[i]  # [3]
        # Difference vectors to all atoms in all replicas: pos_j + T - pos_i
        # Shape: [N, num_translations, 3]
        diffs = (cart_pos[:, np.newaxis, :] + translations[np.newaxis, :, :]) - pos_i[np.newaxis, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=-1)  # [N, num_translations]

        # Mask out self-interaction at origin (T = 0)
        origin_idx = np.where((translation_indices == [0, 0, 0]).all(axis=1))[0][0]
        dists[i, origin_idx] = np.inf

        # Find neighbors within cutoff
        valid_mask = dists <= radius_cutoff
        j_indices, t_indices = np.where(valid_mask)
        valid_dists = dists[valid_mask]
        valid_vecs = diffs[valid_mask]

        if len(valid_dists) > max_neighbors:
            sort_idx = np.argsort(valid_dists)
            cutoff_dist = valid_dists[sort_idx[max_neighbors - 1]]
            # Keep all neighbors up to cutoff_dist + tolerance to avoid breaking symmetry
            keep_mask = valid_dists <= (cutoff_dist + 1e-4)
            j_indices = j_indices[keep_mask]
            valid_dists = valid_dists[keep_mask]
            valid_vecs = valid_vecs[keep_mask]

        for j, d, vec in zip(j_indices, valid_dists, valid_vecs):
            src_nodes.append(i)
            dst_nodes.append(j)
            distances.append(d)
            disp_vectors.append(vec)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_dist = torch.tensor(distances, dtype=torch.float32)
    edge_vec = torch.tensor(np.array(disp_vectors), dtype=torch.float32) if disp_vectors else torch.zeros((0, 3), dtype=torch.float32)

    data = Data(
        x=atomic_numbers,
        pos=torch.tensor(cart_pos, dtype=torch.float32),
        edge_index=edge_index,
        edge_dist=edge_dist,
        edge_vec=edge_vec,
        num_nodes=N
    )

    if target_tensor_st is not None:
        data.y_tensor = torch.tensor(target_tensor_st, dtype=torch.float32)
    if target_scalar is not None:
        data.y_scalar = torch.tensor([target_scalar], dtype=torch.float32)

    return data
