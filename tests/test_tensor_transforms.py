"""Unit tests for tensor transformations and rotational covariance."""

import numpy as np
import pytest
import torch
import e3nn.o3 as o3

from src.features.tensor_transforms import (
    project_symmetric_traceless,
    matrix_to_cartesian_6d,
    cartesian_6d_to_matrix,
    matrix_to_cartesian_5d,
    cartesian_5d_to_matrix,
    matrix_to_irrep_2e,
    irrep_2e_to_matrix,
    compute_principal_components
)


def test_symmetric_traceless_projection():
    """Tests that project_symmetric_traceless produces exact symmetry and zero trace."""
    torch.manual_seed(42)
    A = torch.randn(10, 3, 3)
    V_st = project_symmetric_traceless(A)

    # Check symmetry: V = V^T
    sym_err = torch.norm(V_st - V_st.transpose(-1, -2)).item()
    assert sym_err < 1e-6

    # Check trace = 0
    trace = torch.diagonal(V_st, dim1=-2, dim2=-1).sum(-1)
    assert torch.max(torch.abs(trace)).item() < 1e-6


def test_cartesian_6d_roundtrip():
    """Tests bidirectional conversion between 3x3 symmetric matrix and 6D representation."""
    torch.manual_seed(42)
    A = torch.randn(5, 3, 3)
    V_sym = (A + A.transpose(-1, -2)) / 2.0

    v6 = matrix_to_cartesian_6d(V_sym)
    assert v6.shape == (5, 6)

    V_rec = cartesian_6d_to_matrix(v6)
    rec_err = torch.norm(V_sym - V_rec).item()
    assert rec_err < 1e-6


def test_cartesian_5d_roundtrip():
    """Tests bidirectional conversion between 3x3 symmetric traceless matrix and 5D representation."""
    torch.manual_seed(42)
    A = torch.randn(5, 3, 3)
    V_st = project_symmetric_traceless(A)

    v5 = matrix_to_cartesian_5d(V_st)
    assert v5.shape == (5, 5)

    V_rec = cartesian_5d_to_matrix(v5)
    rec_err = torch.norm(V_st - V_rec).item()
    assert rec_err < 1e-6


def test_irrep_2e_roundtrip():
    """Tests bidirectional conversion between 3x3 symmetric traceless matrix and e3nn 1x2e irrep."""
    torch.manual_seed(42)
    A = torch.randn(10, 3, 3)
    V_st = project_symmetric_traceless(A)

    v2e = matrix_to_irrep_2e(V_st)
    assert v2e.shape == (10, 5)

    V_rec = irrep_2e_to_matrix(v2e)
    rec_err = torch.norm(V_st - V_rec).item()
    assert rec_err < 1e-5


def test_rotation_covariance_e3nn():
    """Tests that rotating Cartesian tensor by R corresponds exactly to D_2(R) @ v_2e."""
    torch.manual_seed(123)
    irrep = o3.Irrep("2e")

    for _ in range(5):
        # Generate random rotation matrix R in SO(3)
        R = o3.rand_matrix()

        # Random symmetric traceless tensor V
        A = torch.randn(3, 3)
        V = project_symmetric_traceless(A)

        # 1. Rotate in Cartesian space: V' = R @ V @ R^T
        V_rot = R @ V @ R.T

        # 2. Convert rotated matrix to 2e irrep
        v2e_from_rot_matrix = matrix_to_irrep_2e(V_rot)

        # 3. Transform 2e irrep using D_2(R) Wigner matrix
        v2e_original = matrix_to_irrep_2e(V)
        D_R = irrep.D_from_matrix(R)
        v2e_from_wigner = D_R @ v2e_original

        # Covariance error must be within numerical precision
        covar_err = torch.norm(v2e_from_rot_matrix - v2e_from_wigner).item()
        assert covar_err < 1e-5, f"Rotation covariance failed with error {covar_err}"


def test_principal_components_and_eta():
    """Tests principal components convention |Vzz| >= |Vyy| >= |Vxx| and eta bounds."""
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(3, 3)
        V = project_symmetric_traceless(A)
        vxx, vyy, vzz, eta = compute_principal_components(V)

        assert abs(vzz) >= abs(vyy) - 1e-5
        assert abs(vyy) >= abs(vxx) - 1e-5
        assert 0.0 <= eta <= 1.0 + 1e-5
