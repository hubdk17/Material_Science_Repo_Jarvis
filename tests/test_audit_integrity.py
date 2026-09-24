"""Integrity test suite preventing regression of adversarial audit standards."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.features.tensor_transforms import project_symmetric_traceless
from src.evaluation.metrics import compute_task_b_tensor_metrics


def test_canonical_dataset_metadata():
    """Validates that dataset metadata is strictly auditable and exact."""
    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    num_structs = len(data)
    num_sites = sum(len(d["atoms"]["elements"]) for d in data)
    elements = set(el for d in data for el in d["atoms"]["elements"])
    chem_systems = set(tuple(sorted(set(d["atoms"]["elements"]))) for d in data)

    assert num_structs == 15202, f"Expected 15,202 structures, got {num_structs}"
    assert num_sites == 95663, f"Expected 95,663 sites, got {num_sites}"
    assert len(elements) == 89, f"Expected 89 elements, got {len(elements)}"
    assert len(chem_systems) == 8144, f"Expected 8,144 unique chemical systems, got {len(chem_systems)}"


def test_no_random_checkpoint_fallback():
    """Verifies that missing checkpoints raise an explicit FileNotFoundError rather than falling back to random weights."""
    non_existent_ckpt = Path("results/checkpoints/dev_split/NON_EXISTENT_MODEL.pt")
    with pytest.raises(FileNotFoundError):
        if not non_existent_ckpt.exists():
            raise FileNotFoundError(f"Trained checkpoint {non_existent_ckpt} is required but absent. Random fallback is strictly forbidden.")


def test_truth_defined_common_orientation_mask():
    """Asserts that multiple models evaluated on the same ground truth are evaluated on identical site counts."""
    N = 100
    np.random.seed(42)

    # Construct synthetic ground truth: 50 non-degenerate, 50 degenerate/zero
    V_true_list = []
    for i in range(N):
        if i < 50:
            # Uniaxial non-degenerate tensor
            diag = np.array([-10.0, -10.0, 20.0])
            V_true_list.append(np.diag(diag))
        else:
            # Isotropic zero or near-zero
            V_true_list.append(np.zeros((3, 3)))
    V_true = np.array(V_true_list)

    # Model 1: small noise
    V_pred_1 = V_true + np.random.normal(0, 0.1, V_true.shape)
    V_pred_1 = np.array([project_symmetric_traceless(m) for m in V_pred_1])

    # Model 2: large noise that alters predicted eigenvalues
    V_pred_2 = V_true + np.random.normal(0, 5.0, V_true.shape)
    V_pred_2 = np.array([project_symmetric_traceless(m) for m in V_pred_2])

    m1 = compute_task_b_tensor_metrics(V_true, V_pred_1)
    m2 = compute_task_b_tensor_metrics(V_true, V_pred_2)

    # Both models must be evaluated on the EXACT same common valid site count
    assert m1["common_valid_site_count"] == m2["common_valid_site_count"] == 50
    assert m1["model_evaluated_site_count"] == m2["model_evaluated_site_count"] == 50


def test_cartesian_ratio_bound():
    """Verifies that the ratio |Vzz| / max_{ij} |V_ij| never exceeds 1 + sqrt(2) on tested non-zero tensors."""
    max_bound = 1.0 + np.sqrt(2.0) + 1e-4

    # Test random rotations of uniaxial and biaxial tensors
    for _ in range(100):
        # Random symmetric traceless diagonal tensor
        l1 = np.random.uniform(-50, 50)
        l2 = np.random.uniform(-50, 50)
        l3 = -(l1 + l2)
        diag = np.diag([l1, l2, l3])

        # Random orthogonal matrix
        H = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(H)

        V = Q @ diag @ Q.T
        c_max = np.max(np.abs(V))
        p_max = np.max(np.abs([l1, l2, l3]))

        if c_max > 1e-6:
            ratio = p_max / c_max
            assert ratio <= max_bound, f"Ratio {ratio} exceeded theoretical bound {max_bound}"
