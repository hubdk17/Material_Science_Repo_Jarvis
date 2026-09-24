"""Crystallographic site-symmetry analysis and point-group tensor projection using spglib.

Provides:
- Exact extraction of site stabilizer operations from crystal symmetry without target leakage.
- Conversion of fractional-space symmetry matrices to Cartesian orthogonal matrices.
- Projective averaging of rank-2 EFG tensors over the site stabilizer group:
    V_sym = (1 / |G_i|) * sum_{R in G_i} R @ V @ R^T
- Group-theoretic determination of whether site symmetry strictly enforces EFG = 0.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import spglib


def get_site_stabilizers(
    lattice_mat: np.ndarray,
    fractional_coords: np.ndarray,
    elements: List[str],
    symprec: float = 1e-4
) -> List[List[np.ndarray]]:
    """Computes Cartesian orthogonal symmetry matrices for the stabilizer group G_i
    of each atomic site i in the unit cell.

    Parameters:
    - lattice_mat: 3x3 array where rows are lattice vectors a, b, c.
    - fractional_coords: Nx3 array of atomic fractional coordinates.
    - elements: length-N list of element symbols.
    - symprec: spglib symmetry tolerance.

    Returns:
    - List of length N, where each element is a list of 3x3 Cartesian orthogonal matrices
      corresponding to operations in the space group that map site i to itself
      (modulo lattice translations).
    """
    lat = np.array(lattice_mat, dtype=np.float64)
    frac = np.array(fractional_coords, dtype=np.float64) % 1.0

    # Map elements to unique integer types
    unique_elems = {el: idx + 1 for idx, el in enumerate(sorted(set(elements)))}
    atom_types = [unique_elems[el] for el in elements]

    cell = (lat, frac, atom_types)
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    N = len(elements)
    if dataset is None:
        return [[np.eye(3, dtype=np.float64)] for _ in range(N)]

    rotations = getattr(dataset, "rotations", None)
    translations = getattr(dataset, "translations", None)
    if rotations is None:
        rotations = dataset["rotations"]
        translations = dataset["translations"]

    # Metric tensor for converting fractional rotations to Cartesian:
    # A is lattice matrix where rows are a, b, c:
    # r_cart = r_frac @ A
    # In Cartesian space: r'_cart = r_cart @ R_cart^T
    # r'_frac @ A = (r_frac @ R_frac^T) @ A  =>  R_cart^T = A^{-1} R_frac^T A
    # => R_cart = A^T R_frac (A^T)^{-1}
    A = lat
    A_inv = np.linalg.inv(A)

    stabilizers_per_site = []

    for i in range(N):
        x_i = frac[i]
        site_ops = []

        for R_frac, t_frac in zip(rotations, translations):
            # Check if this operation maps site i to itself modulo integer lattice vector:
            # x'_i = R_frac @ x_i + t_frac
            x_mapped = R_frac @ x_i + t_frac
            diff = x_mapped - x_i
            diff_round = np.round(diff)

            if np.allclose(diff, diff_round, atol=symprec):
                # R_frac maps site i to itself modulo lattice translation
                # Convert R_frac to Cartesian orthogonal matrix:
                # R_cart acts on Cartesian column vectors: v_cart' = R_cart @ v_cart
                # Since r_cart = A^T @ r_frac (when vectors are columns):
                # R_cart = A^T @ R_frac @ (A^T)^{-1}
                R_cart = A.T @ R_frac @ np.linalg.inv(A.T)

                # Ensure numerical orthogonality
                U, _, Vt = np.linalg.svd(R_cart)
                R_cart_ortho = U @ Vt

                site_ops.append(R_cart_ortho)

        if not site_ops:
            site_ops = [np.eye(3, dtype=np.float64)]

        stabilizers_per_site.append(site_ops)

    return stabilizers_per_site


def project_tensor_by_site_symmetry(
    V: np.ndarray,
    stabilizer_ops: List[np.ndarray]
) -> np.ndarray:
    """Projects a rank-2 Cartesian tensor V by averaging over the site stabilizer group:
        V_proj = (1 / |G|) * sum_{R in G} R @ V @ R^T

    Then enforces traceless symmetric constraints.
    """
    if not stabilizer_ops:
        return V

    V_avg = np.zeros_like(V, dtype=np.float64)
    for R in stabilizer_ops:
        V_avg += R @ V @ R.T

    V_avg /= len(stabilizer_ops)

    # Enforce symmetric traceless
    V_sym = (V_avg + V_avg.T) / 2.0
    trace = np.trace(V_sym)
    return V_sym - (trace / 3.0) * np.eye(3, dtype=np.float64)


def classify_site_efg_symmetry(
    stabilizer_ops: List[np.ndarray],
    threshold: float = 1e-4
) -> Tuple[bool, int, str]:
    """Determines whether site stabilizer symmetry strictly forces any rank-2 traceless
    symmetric tensor to vanish identically (dimension of invariant subspace = 0).

    Constructs the 5D representation of stabilizer operations on symmetric traceless tensors
    and computes the projection operator:
        P = (1 / |G|) * sum_{R in G} (R (x) R)|_{ST}
    The trace of P gives the dimension of the invariant subspace.

    Returns:
    - is_symmetry_enforced_zero: True if invariant subspace dimension is 0.
    - invariant_dimension: dimension of invariant subspace (0 to 5).
    - site_symmetry_description: e.g. "cubic_symmetry_enforced_zero" or "finite_subspace"
    """
    from src.features.tensor_transforms import matrix_to_cartesian_5d, cartesian_5d_to_matrix

    # Basis tensors for the 5-dimensional symmetric traceless space
    basis_5d = np.eye(5)
    basis_mats = [cartesian_5d_to_matrix(basis_5d[k]) for k in range(5)]

    # Compute 5x5 representation matrix for each stabilizer operation
    P_5x5 = np.zeros((5, 5), dtype=np.float64)
    for R in stabilizer_ops:
        M_R = np.zeros((5, 5), dtype=np.float64)
        for k in range(5):
            V_k = basis_mats[k]
            V_k_rot = R @ V_k @ R.T
            v5_rot = matrix_to_cartesian_5d(V_k_rot)
            M_R[:, k] = v5_rot
        P_5x5 += M_R

    P_5x5 /= len(stabilizer_ops)

    # Invariant subspace dimension is the trace of the projector matrix P
    inv_dim = int(np.round(np.trace(P_5x5)))

    is_forced_zero = (inv_dim == 0)
    desc = "symmetry_enforced_zero" if is_forced_zero else f"invariant_dim_{inv_dim}"
    return is_forced_zero, inv_dim, desc
