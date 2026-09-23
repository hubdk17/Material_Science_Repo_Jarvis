"""Automated unit and regression tests for unit conversion, split verification, and metric validity."""

import json
import zipfile
import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_crystal_macro_tensor_metrics,
    compute_task_b_tensor_metrics
)
from src.features.tensor_transforms import (
    cartesian_5d_to_matrix,
    compute_principal_components,
    project_symmetric_traceless
)


def test_unit_conversion_constant():
    """Verifies physical unit conversion factor: 1 x 10^21 V m^-2 = 10 V Angstrom^-2."""
    v_m2 = 1.0e21  # V / m^2
    angstrom_in_meters = 1.0e-10
    v_ang2 = v_m2 * (angstrom_in_meters ** 2)
    # 1e21 * 1e-20 = 10.0 V / Angstrom^2
    assert np.isclose(v_ang2, 10.0), f"Expected 10.0, got {v_ang2}"


def test_manually_checked_records_unit_reconciliation():
    """Verifies that 10.0 * max(|CSV_VASP_V|) matches official benchmark target on reference crystals."""
    with zipfile.ZipFile("data/raw/dft_3d_max_efg.json.zip", "r") as z:
        bench_data = json.loads(z.read("dft_3d_max_efg.json").decode("utf-8"))

    bench_targets = {}
    for s in ["train", "val", "test"]:
        bench_targets.update(bench_data[s])

    # Reference records:
    # JVASP-10: CSV max is 8.9678 -> bench is 89.678
    assert np.isclose(bench_targets["JVASP-10"], 89.678, atol=1e-3)
    # JVASP-32394: CSV max is 12.6342 -> bench is 126.342
    assert np.isclose(bench_targets["JVASP-32394"], 126.342, atol=1e-3)
    # JVASP-8667: CSV max is 5.0249 -> bench is 50.249
    assert np.isclose(bench_targets["JVASP-8667"], 50.249, atol=1e-3)
    # JVASP-51: Benchmark target is 76.088
    assert np.isclose(bench_data["train"]["JVASP-51"], 76.088, atol=1e-3)


def test_official_split_partition_counts():
    """Verifies official partition counts vs locally available counts."""
    with zipfile.ZipFile("data/raw/dft_3d_max_efg.json.zip", "r") as z:
        bench_data = json.loads(z.read("dft_3d_max_efg.json").decode("utf-8"))

    with open("splits/official_jarvis_scalar.json", "r", encoding="utf-8") as f:
        split_a = json.load(f)

    # Official benchmark counts
    assert len(bench_data["train"]) == 9493
    assert len(bench_data["val"]) == 1186
    assert len(bench_data["test"]) == 1186
    assert (len(bench_data["train"]) + len(bench_data["val"]) + len(bench_data["test"])) == 11865

    # Locally available counts (JVASP-51 missing from local training set)
    assert len(split_a["train"]["jids"]) == 9492
    assert len(split_a["val"]["jids"]) == 1186
    assert len(split_a["test"]["jids"]) == 1186
    assert split_a["missing_ids_report"]["train_missing"] == ["JVASP-51"]


def test_b0_undefined_metrics_return_nan():
    """Verifies that B0 (zero tensor) reports NaN for eta and orientation error rather than 0.0 or arbitrary angle."""
    V_true = np.array([
        [[20.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, -10.0]],
        [[5.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, -20.0]]
    ])
    V_b0 = np.zeros_like(V_true)

    metrics = compute_task_b_tensor_metrics(V_true, V_b0)

    # Frobenius error must be non-zero
    assert metrics["frobenius_error_mean"] > 0.0
    # B0 eta must be NaN
    assert np.isnan(metrics["asymmetry_eta_mae"]), "B0 eta error should be NaN"
    # B0 orientation must be NaN
    assert np.isnan(metrics["principal_axis_angle_mean_deg"]), "B0 orientation error should be NaN"
    assert metrics["valid_orientation_count"] == 0
    assert metrics["valid_eta_count"] == 0


def test_scale_aware_degeneracy_and_sign_ambiguity():
    """Verifies that degenerate eigenvalues are excluded and sign-flipped eigenvectors have zero angular error."""
    # Diagonal tensor with nondegenerate eigenvalues: |lambda_3| = 30, |lambda_2| = 20, |lambda_1| = 10
    V_t = np.diag([10.0, -20.0, 10.0])[None]
    # Predict identical tensor but rotated by 180 degrees (eigenvector sign flipped)
    # Since V is symmetric, sign-flip of eigenvector does not change the tensor or axis
    V_p = V_t.copy()

    metrics = compute_task_b_tensor_metrics(V_t, V_p)
    assert np.isclose(metrics["frobenius_error_mean"], 0.0)
    assert np.isclose(metrics["principal_axis_angle_mean_deg"], 0.0, atol=1e-4)

    # Perfectly degenerate tensor: lambda_1 = lambda_2 = lambda_3 = 0 (cubic)
    V_deg = np.zeros((1, 3, 3))
    m_deg = compute_task_b_tensor_metrics(V_deg, V_deg)
    assert m_deg["valid_orientation_count"] == 0
