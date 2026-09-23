"""Mandatory rotation covariance and invariance test suite for Task A and Task B models."""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import e3nn.o3 as o3
from torch_geometric.data import Batch

from src.features.graph import build_periodic_crystal_graph


def run_rotation_invariance_test_scalar(
    model: torch.nn.Module,
    sample_records: List[Dict[str, Any]],
    num_rotations: int = 5,
    tolerance: float = 1e-3,
    radius_cutoff: float = 5.05
) -> Dict[str, Any]:
    """Tests that a crystal-level scalar model is rotationally invariant under random SO(3) rotations:
    |f(R * crystal) - f(crystal)| < tolerance.
    """
    model.eval()
    errors = []

    for rec in sample_records:
        lat = np.array(rec["atoms"]["lattice_mat"], dtype=float)
        coords = np.array(rec["atoms"]["coords"], dtype=float)
        elems = rec["atoms"]["elements"]

        g_orig = build_periodic_crystal_graph(lat, coords, elems, radius_cutoff=radius_cutoff)
        b_orig = Batch.from_data_list([g_orig])

        with torch.no_grad():
            pred_orig = model(b_orig).item()

        for _ in range(num_rotations):
            R = o3.rand_matrix().numpy()
            lat_rot = lat @ R.T

            g_rot = build_periodic_crystal_graph(lat_rot, coords, elems, radius_cutoff=radius_cutoff)
            b_rot = Batch.from_data_list([g_rot])

            with torch.no_grad():
                pred_rot = model(b_rot).item()

            diff = abs(pred_orig - pred_rot)
            errors.append(diff)

    max_err = float(np.max(errors)) if errors else 0.0
    mean_err = float(np.mean(errors)) if errors else 0.0

    return {
        "max_invariance_error": max_err,
        "mean_invariance_error": mean_err,
        "passed": max_err < tolerance,
        "tolerance": tolerance,
        "num_tests": len(errors)
    }


def run_rotation_covariance_test_tensor(
    model: torch.nn.Module,
    sample_records: List[Dict[str, Any]],
    num_rotations: int = 5,
    tolerance: float = 1e-4,
    radius_cutoff: float = 5.05
) -> Dict[str, Any]:
    """Tests that a site-resolved tensor model is rotationally covariant under random SO(3) rotations:
    ||V'(R * crystal) - R V(crystal) R^T||_F < tolerance.
    """
    model.eval()
    errors = []

    for rec in sample_records:
        lat = np.array(rec["atoms"]["lattice_mat"], dtype=float)
        coords = np.array(rec["atoms"]["coords"], dtype=float)
        elems = rec["atoms"]["elements"]

        g_orig = build_periodic_crystal_graph(lat, coords, elems, radius_cutoff=radius_cutoff)
        b_orig = Batch.from_data_list([g_orig])

        with torch.no_grad():
            _, V_orig = model(b_orig)

        for _ in range(num_rotations):
            R = o3.rand_matrix()
            R_np = R.numpy()
            lat_rot = lat @ R_np.T

            g_rot = build_periodic_crystal_graph(lat_rot, coords, elems, radius_cutoff=radius_cutoff)
            b_rot = Batch.from_data_list([g_rot])

            with torch.no_grad():
                _, V_rot = model(b_rot)

            R_torch = R.to(dtype=V_orig.dtype, device=V_orig.device)
            V_orig_transformed = R_torch @ V_orig @ R_torch.transpose(-1, -2)

            frob_diff = (V_rot - V_orig_transformed).norm(dim=(-2, -1)).max().item()
            errors.append(frob_diff)

    max_err = float(np.max(errors)) if errors else 0.0
    mean_err = float(np.mean(errors)) if errors else 0.0

    return {
        "max_covariance_error": max_err,
        "mean_covariance_error": mean_err,
        "passed": max_err < tolerance,
        "tolerance": tolerance,
        "num_tests": len(errors)
    }
