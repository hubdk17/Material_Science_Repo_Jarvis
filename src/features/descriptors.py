"""Compositional and local structural feature extraction for baseline models."""

import math
from collections import Counter
from typing import Any, Dict, List, Tuple
import numpy as np

# Core elemental properties table (Z: [atomic_mass, electronegativity, covalent_radius, valence_e, group, period])
ELEMENT_PROPERTIES: Dict[str, List[float]] = {
    "H": [1.008, 2.20, 0.31, 1, 1, 1],
    "He": [4.003, 0.00, 0.28, 2, 18, 1],
    "Li": [6.94, 0.98, 1.28, 1, 1, 2],
    "Be": [9.012, 1.57, 0.96, 2, 2, 2],
    "B": [10.81, 2.04, 0.84, 3, 13, 2],
    "C": [12.011, 2.55, 0.76, 4, 14, 2],
    "N": [14.007, 3.04, 0.71, 5, 15, 2],
    "O": [15.999, 3.44, 0.66, 6, 16, 2],
    "F": [18.998, 3.98, 0.57, 7, 17, 2],
    "Ne": [20.18, 0.00, 0.58, 8, 18, 2],
    "Na": [22.99, 0.93, 1.66, 1, 1, 3],
    "Mg": [24.305, 1.31, 1.41, 2, 2, 3],
    "Al": [26.982, 1.61, 1.21, 3, 13, 3],
    "Si": [28.085, 1.90, 1.11, 4, 14, 3],
    "P": [30.974, 2.19, 1.07, 5, 15, 3],
    "S": [32.06, 2.58, 1.05, 6, 16, 3],
    "Cl": [35.45, 3.16, 1.02, 7, 17, 3],
    "Ar": [39.948, 0.00, 1.06, 8, 18, 3],
    "K": [39.098, 0.82, 2.03, 1, 1, 4],
    "Ca": [40.078, 1.00, 1.76, 2, 2, 4],
    "Sc": [44.956, 1.36, 1.70, 3, 3, 4],
    "Ti": [47.867, 1.54, 1.60, 4, 4, 4],
    "V": [50.942, 1.63, 1.53, 5, 5, 4],
    "Cr": [51.996, 1.66, 1.39, 6, 6, 4],
    "Mn": [54.938, 1.55, 1.39, 7, 7, 4],
    "Fe": [55.845, 1.83, 1.32, 8, 8, 4],
    "Co": [58.933, 1.88, 1.26, 9, 9, 4],
    "Ni": [58.693, 1.91, 1.24, 10, 10, 4],
    "Cu": [63.546, 1.90, 1.32, 11, 11, 4],
    "Zn": [65.38, 1.65, 1.22, 12, 12, 4],
    "Ga": [69.723, 1.81, 1.22, 3, 13, 4],
    "Ge": [72.63, 2.01, 1.20, 4, 14, 4],
    "As": [74.922, 2.18, 1.19, 5, 15, 4],
    "Se": [78.971, 2.55, 1.20, 6, 16, 4],
    "Br": [79.904, 2.96, 1.20, 7, 17, 4],
    "Kr": [83.798, 3.00, 1.16, 8, 18, 4],
    "Rb": [85.468, 0.82, 2.20, 1, 1, 5],
    "Sr": [87.62, 0.95, 1.95, 2, 2, 5],
    "Y": [88.906, 1.22, 1.90, 3, 3, 5],
    "Zr": [91.224, 1.33, 1.75, 4, 4, 5],
    "Nb": [92.906, 1.60, 1.64, 5, 5, 5],
    "Mo": [95.95, 2.16, 1.54, 6, 6, 5],
    "Tc": [98.0, 1.90, 1.47, 7, 7, 5],
    "Ru": [101.07, 2.20, 1.46, 8, 8, 5],
    "Rh": [102.91, 2.28, 1.42, 9, 9, 5],
    "Pd": [106.42, 2.20, 1.39, 10, 10, 5],
    "Ag": [107.87, 1.93, 1.45, 11, 11, 5],
    "Cd": [112.41, 1.69, 1.44, 12, 12, 5],
    "In": [114.82, 1.78, 1.42, 3, 13, 5],
    "Sn": [118.71, 1.96, 1.39, 4, 14, 5],
    "Sb": [121.76, 2.05, 1.39, 5, 15, 5],
    "Te": [127.6, 2.10, 1.38, 6, 16, 5],
    "I": [126.9, 2.66, 1.39, 7, 17, 5],
    "Xe": [131.29, 2.60, 1.40, 8, 18, 5],
    "Cs": [132.91, 0.79, 2.44, 1, 1, 6],
    "Ba": [137.33, 0.89, 2.15, 2, 2, 6],
    "La": [138.91, 1.10, 2.07, 3, 3, 6],
    "Ce": [140.12, 1.12, 2.04, 4, 3, 6],
    "Pr": [140.91, 1.13, 2.03, 5, 3, 6],
    "Nd": [144.24, 1.14, 2.01, 6, 3, 6],
    "Pm": [145.0, 1.13, 1.99, 7, 3, 6],
    "Sm": [150.36, 1.17, 1.98, 8, 3, 6],
    "Eu": [151.96, 1.20, 1.98, 9, 3, 6],
    "Gd": [157.25, 1.20, 1.96, 10, 3, 6],
    "Tb": [158.93, 1.22, 1.94, 11, 3, 6],
    "Dy": [162.5, 1.23, 1.92, 12, 3, 6],
    "Ho": [164.93, 1.24, 1.92, 13, 3, 6],
    "Er": [167.26, 1.24, 1.89, 14, 3, 6],
    "Tm": [168.93, 1.25, 1.90, 15, 3, 6],
    "Yb": [173.05, 1.10, 1.87, 16, 3, 6],
    "Lu": [174.97, 1.27, 1.87, 3, 3, 6],
    "Hf": [178.49, 1.30, 1.75, 4, 4, 6],
    "Ta": [180.95, 1.50, 1.70, 5, 5, 6],
    "W": [183.84, 2.36, 1.62, 6, 6, 6],
    "Re": [186.21, 1.90, 1.51, 7, 7, 6],
    "Os": [190.23, 2.20, 1.44, 8, 8, 6],
    "Ir": [192.22, 2.20, 1.41, 9, 9, 6],
    "Pt": [195.08, 2.28, 1.36, 10, 10, 6],
    "Au": [196.97, 2.54, 1.36, 11, 11, 6],
    "Hg": [200.59, 2.00, 1.32, 12, 12, 6],
    "Tl": [204.38, 1.62, 1.45, 3, 13, 6],
    "Pb": [207.2, 2.33, 1.46, 4, 14, 6],
    "Bi": [208.98, 2.02, 1.48, 5, 15, 6],
    "Po": [209.0, 2.00, 1.40, 6, 16, 6],
    "At": [210.0, 2.20, 1.50, 7, 17, 6],
    "Rn": [222.0, 0.00, 1.50, 8, 18, 6],
    "Fr": [223.0, 0.70, 2.60, 1, 1, 7],
    "Ra": [226.0, 0.90, 2.21, 2, 2, 7],
    "Ac": [227.0, 1.10, 2.15, 3, 3, 7],
    "Th": [232.04, 1.30, 2.06, 4, 3, 7],
    "Pa": [231.04, 1.50, 2.00, 5, 3, 7],
    "U": [238.03, 1.38, 1.96, 6, 3, 7],
    "Np": [237.0, 1.36, 1.90, 7, 3, 7],
    "Pu": [244.0, 1.28, 1.87, 8, 3, 7],
    "Am": [243.0, 1.30, 1.80, 9, 3, 7],
    "Cm": [247.0, 1.30, 1.69, 10, 3, 7],
    "Bk": [247.0, 1.30, 1.54, 11, 3, 7],
    "Cf": [251.0, 1.30, 1.83, 12, 3, 7],
    "Es": [252.0, 1.30, 1.50, 13, 3, 7],
    "Fm": [257.0, 1.30, 1.50, 14, 3, 7]
}

DEFAULT_PROP = [50.0, 1.5, 1.5, 4, 7, 4]


def extract_magpie_composition_features(elements: List[str]) -> np.ndarray:
    """Extracts Magpie-style compositional descriptor vector.
    Computes mean, std, min, max, range for each of the 6 fundamental atomic properties.
    Total features: 6 * 5 = 30 features.
    """
    props = np.array([ELEMENT_PROPERTIES.get(el, DEFAULT_PROP) for el in elements], dtype=np.float32)

    mean_feat = np.mean(props, axis=0)
    std_feat = np.std(props, axis=0)
    min_feat = np.min(props, axis=0)
    max_feat = np.max(props, axis=0)
    range_feat = max_feat - min_feat

    return np.concatenate([mean_feat, std_feat, min_feat, max_feat, range_feat], axis=0)


def extract_crystal_structural_features(record: Dict[str, Any]) -> np.ndarray:
    """Extracts global structural descriptors combining composition, volume, density, and packing."""
    atoms = record["atoms"]
    elements = atoms["elements"]
    comp_feats = extract_magpie_composition_features(elements)

    lat = np.array(atoms["lattice_mat"], dtype=float)
    vol = abs(float(np.linalg.det(lat)))
    n_atoms = len(elements)
    atomic_masses = [ELEMENT_PROPERTIES.get(el, DEFAULT_PROP)[0] for el in elements]
    total_mass = sum(atomic_masses)

    # Density in g/cm^3: mass / vol (with conversion: mass in amu / vol in A^3 * 1.66054)
    density = (total_mass / vol) * 1.66054 if vol > 0 else 0.0
    vol_per_atom = vol / n_atoms if n_atoms > 0 else 0.0

    cell_params = np.array(atoms.get("abc", [1.0, 1.0, 1.0]), dtype=float)
    cell_angles = np.array(atoms.get("angles", [90.0, 90.0, 90.0]), dtype=float)

    struct_feats = np.array([
        vol,
        n_atoms,
        total_mass,
        density,
        vol_per_atom,
        *cell_params,
        *cell_angles
    ], dtype=np.float32)

    return np.concatenate([comp_feats, struct_feats], axis=0)


def extract_site_local_descriptors(
    lattice_mat: np.ndarray,
    frac_coords: np.ndarray,
    elements: List[str],
    site_idx: int,
    cutoff: float = 4.0
) -> np.ndarray:
    """Extracts local coordination environment descriptors for a single atomic site (Baseline B2)."""
    el = elements[site_idx]
    center_props = np.array(ELEMENT_PROPERTIES.get(el, DEFAULT_PROP), dtype=np.float32)

    lat = np.array(lattice_mat, dtype=float)
    frac = np.array(frac_coords, dtype=float) % 1.0
    cart_pos = frac @ lat
    center_pos = cart_pos[site_idx]

    # Find neighbor distances within cutoff under PBC
    recip_lat = np.linalg.inv(lat)
    plane_dist = 1.0 / np.linalg.norm(recip_lat, axis=0)
    max_rep = np.ceil(cutoff / plane_dist).astype(int)

    rx = np.arange(-max_rep[0], max_rep[0] + 1)
    ry = np.arange(-max_rep[1], max_rep[1] + 1)
    rz = np.arange(-max_rep[2], max_rep[2] + 1)
    grid = np.meshgrid(rx, ry, rz, indexing="ij")
    translations = np.stack(grid, axis=-1).reshape(-1, 3) @ lat

    diffs = (cart_pos[:, np.newaxis, :] + translations[np.newaxis, :, :]) - center_pos[np.newaxis, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=-1)

    # Exclude self-atom at origin
    origin_idx = np.where((np.stack(grid, axis=-1).reshape(-1, 3) == [0, 0, 0]).all(axis=1))[0][0]
    dists[site_idx, origin_idx] = np.inf

    neighbor_dists = dists[dists <= cutoff]
    coord_num = len(neighbor_dists)
    mean_dist = float(np.mean(neighbor_dists)) if coord_num > 0 else cutoff
    min_dist = float(np.min(neighbor_dists)) if coord_num > 0 else cutoff
    std_dist = float(np.std(neighbor_dists)) if coord_num > 0 else 0.0

    local_stats = np.array([coord_num, mean_dist, min_dist, std_dist], dtype=np.float32)
    return np.concatenate([center_props, local_stats], axis=0)
