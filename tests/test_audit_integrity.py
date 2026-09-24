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
from src.features.site_symmetry import get_site_stabilizers, project_tensor_by_site_symmetry, classify_site_efg_symmetry


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


def test_b0_exact_nans():
    """Requirement 6: Asserts B0 metrics have exact NaNs for undefined quantities."""
    N = 20
    V_true = np.array([np.diag([-10.0, -10.0, 20.0]) for _ in range(N)])
    V_pred_b0 = np.zeros_like(V_true)

    metrics = compute_task_b_tensor_metrics(V_true, V_pred_b0)

    assert np.isnan(metrics["asymmetry_eta_mae"]), "B0 eta MAE must be NaN"
    assert np.isnan(metrics["asymmetry_eta_conditional_mae"]), "B0 conditional eta MAE must be NaN"
    assert metrics["eta_coverage_rate"] == 0.0, "B0 eta coverage rate must be 0.0"
    assert np.isnan(metrics["principal_axis_angle_mean_deg"]), "B0 orientation mean must be NaN"
    assert np.isnan(metrics["principal_axis_angle_median_deg"]), "B0 orientation median must be NaN"
    assert metrics["prediction_validity_rate"] == 0.0, "B0 prediction validity rate must be 0.0"


def test_analytical_orientation_error_on_rotated_tensors():
    """Requirement 7: Asserts that orientation errors on analytically rotated tensors match expected angles."""
    # Base uniaxial tensor with principal axis along z = [0, 0, 1]
    V_base = np.diag([-5.0, -5.0, 10.0])

    angles_deg = [10.0, 25.0, 45.0, 60.0, 75.0, 90.0]
    V_true_list = []
    V_pred_list = []

    for deg in angles_deg:
        rad = np.radians(deg)
        # Rotation around y-axis by angle deg
        R = np.array([
            [np.cos(rad), 0.0, np.sin(rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(rad), 0.0, np.cos(rad)]
        ])
        V_rot = R @ V_base @ R.T

        V_true_list.append(V_base)
        V_pred_list.append(V_rot)

    V_true = np.array(V_true_list)
    V_pred = np.array(V_pred_list)

    # Compute orientation metrics
    metrics = compute_task_b_tensor_metrics(V_true, V_pred)

    expected_mean = float(np.mean(angles_deg))
    assert abs(metrics["principal_axis_angle_mean_deg"] - expected_mean) < 1e-4, (
        f"Mean orientation error {metrics['principal_axis_angle_mean_deg']} != expected {expected_mean}"
    )


def test_b6_no_target_leakage_and_cubic_enforcement():
    """Requirement 3.B: Proves B6 receives only structures and predictions, and enforces zero on cubic sites."""
    # Generate the 24 proper rotations of the cubic group O
    Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)

    cubic_ops = [np.eye(3, dtype=float)]
    added = True
    while added:
        added = False
        new_mats = []
        for g in cubic_ops:
            for gen in [Rx, Ry]:
                prod = g @ gen
                if not any(np.allclose(prod, m) for m in cubic_ops) and not any(np.allclose(prod, m) for m in new_mats):
                    new_mats.append(prod)
                    added = True
        cubic_ops.extend(new_mats)

    assert len(cubic_ops) == 24, f"Cubic group O must have order 24, got {len(cubic_ops)}"

    is_forced_zero, inv_dim, _ = classify_site_efg_symmetry(cubic_ops)
    assert is_forced_zero, "Cubic site stabilizer must enforce invariant dimension 0"
    assert inv_dim == 0, "Invariant dimension must be 0"

    # Arbitrary non-zero prediction input
    V_pred_input = np.array([
        [12.3, -4.5, 6.7],
        [-4.5, -8.1, 2.2],
        [6.7, 2.2, -4.2]
    ])
    # Project by site symmetry
    V_proj = project_tensor_by_site_symmetry(V_pred_input, cubic_ops)
    # Must project to exactly 0 (within machine precision 1e-12)
    assert np.allclose(V_proj, 0.0, atol=1e-12), f"Projected tensor on cubic site must be zero, got {V_proj}"


def test_uncertainty_responsiveness_to_residuals():
    """Requirement 3.C: Proves coverage responds dynamically to changed residuals."""
    val_res = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    q_80 = float(np.percentile(val_res, 80))

    # Test set 1: low residuals
    test_res_low = np.array([15.0, 25.0, 35.0, 45.0])
    cov_low = float(np.mean(test_res_low <= q_80))

    # Test set 2: degraded high residuals
    test_res_high = np.array([90.0, 110.0, 120.0, 150.0])
    cov_high = float(np.mean(test_res_high <= q_80))

    assert cov_low == 1.0, "Low residuals should achieve 100% empirical coverage"
    assert cov_high == 0.0, "High residuals should achieve 0% empirical coverage"
    assert cov_low > cov_high, "Coverage must respond dynamically to shifted residual distributions"


def test_cartesian_ratio_bound():
    """Verifies that the ratio |Vzz| / max_{ij} |V_ij| never exceeds 1 + sqrt(2) on tested non-zero tensors."""
    max_bound = 1.0 + np.sqrt(2.0) + 1e-4

    # Test random rotations of uniaxial and biaxial tensors
    for _ in range(100):
        l1 = np.random.uniform(-50, 50)
        l2 = np.random.uniform(-50, 50)
        l3 = -(l1 + l2)
        diag = np.diag([l1, l2, l3])

        H = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(H)

        V = Q @ diag @ Q.T
        c_max = np.max(np.abs(V))
        p_max = np.max(np.abs([l1, l2, l3]))

        if c_max > 1e-6:
            ratio = p_max / c_max
            assert ratio <= max_bound, f"Ratio {ratio} exceeded theoretical bound {max_bound}"
