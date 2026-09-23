"""Automated unit tests for SO(3) rotational invariance and equivariance."""

import json
import pytest
import torch
import numpy as np

from src.evaluation.rotation_tests import (
    run_rotation_invariance_test_scalar,
    run_rotation_covariance_test_tensor
)
from src.models.scalar_baselines import ALIGNNEquivalent
from src.models.invariant_gnn import InvariantTensorGNN
from src.models.equivariant_gnn import EquivariantTensorGNN


@pytest.fixture(scope="module")
def sample_records():
    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:5]  # Fast unit test on 5 diverse crystals


def test_scalar_rotation_invariance(sample_records):
    """Proves ALIGNN scalar prediction is invariant under 3D spatial rotations."""
    model = ALIGNNEquivalent()
    res = run_rotation_invariance_test_scalar(model, sample_records, num_rotations=3, tolerance=1e-3)
    assert res["passed"] is True
    assert res["max_invariance_error"] < 1e-4


def test_equivariant_gnn_rotation_covariance(sample_records):
    """Proves EquivariantTensorGNN transforms covariantly under 3D spatial rotations: V(RX) = R V(X) R^T."""
    model = EquivariantTensorGNN()
    res = run_rotation_covariance_test_tensor(model, sample_records, num_rotations=3, tolerance=1e-4)
    assert res["passed"] is True
    assert res["max_covariance_error"] < 1e-4


def test_invariant_gnn_fails_covariance_control(sample_records):
    """Proves InvariantTensorGNN fails tensor covariance because Cartesian predictions cannot track physical rotation."""
    model = InvariantTensorGNN(mode="symmetric_traceless_5d")
    res = run_rotation_covariance_test_tensor(model, sample_records, num_rotations=3, tolerance=1e-4)
    # Control test: Must fail covariance check with non-trivial error
    assert res["passed"] is False
    assert res["max_covariance_error"] > 0.01
