"""Tensor transformation utilities for EFG rank-2 tensors.

Provides bidirectional conversions between:
- Full 3x3 Cartesian tensor
- 6-component symmetric Cartesian representation
- 5-component symmetric-traceless Cartesian representation
- 5-component l=2 irreducible representation (e3nn compatible)
- Eigenvalue / principal component representation (|Vzz| >= |Vyy| >= |Vxx|, eta)
"""

from typing import Tuple, Union
import numpy as np
import torch
import e3nn.o3 as o3

# Precompute e3nn change of basis for symmetric rank-2 tensors
# Shape: [6, 3, 3] where index 0 is 1x0e (trace) and indices 1:6 are 1x2e (symmetric traceless)
_RTP = o3.ReducedTensorProducts("ij=ji", i="1o", j="1o")
_Q_SYM = _RTP.change_of_basis.detach()  # [6, 3, 3]
_Q_2E = _Q_SYM[1:].clone()  # [5, 3, 3]
_Q_2E_NP = _Q_2E.numpy()


def project_symmetric_traceless(V: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Projects arbitrary 3x3 tensor to symmetric traceless form:
    V_sym = (V + V^T) / 2
    V_ST = V_sym - Tr(V_sym)/3 * I
    """
    if isinstance(V, list):
        V = np.asarray(V, dtype=float)

    if isinstance(V, np.ndarray):
        V_sym = (V + V.swapaxes(-1, -2)) / 2.0
        trace = np.trace(V_sym, axis1=-2, axis2=-1)
        eye = np.eye(3, dtype=V.dtype)
        if V.ndim > 2:
            eye = np.broadcast_to(eye, V.shape)
            trace = trace[..., np.newaxis, np.newaxis]
        return V_sym - (trace / 3.0) * eye
    elif isinstance(V, torch.Tensor):
        V_sym = (V + V.transpose(-1, -2)) / 2.0
        trace = torch.diagonal(V_sym, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(3, dtype=V.dtype, device=V.device)
        return V_sym - (trace / 3.0) * eye
    else:
        raise TypeError(f"Unsupported type: {type(V)}")


def matrix_to_cartesian_6d(V: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Converts 3x3 symmetric matrix to 6 independent components:
    [Vxx, Vyy, Vzz, Vxy, Vxz, Vyz]
    """
    if isinstance(V, np.ndarray):
        return np.stack([
            V[..., 0, 0],
            V[..., 1, 1],
            V[..., 2, 2],
            V[..., 0, 1],
            V[..., 0, 2],
            V[..., 1, 2]
        ], axis=-1)
    elif isinstance(V, torch.Tensor):
        return torch.stack([
            V[..., 0, 0],
            V[..., 1, 1],
            V[..., 2, 2],
            V[..., 0, 1],
            V[..., 0, 2],
            V[..., 1, 2]
        ], dim=-1)


def cartesian_6d_to_matrix(v6: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Converts 6 independent components back to 3x3 symmetric matrix."""
    if isinstance(v6, np.ndarray):
        vxx, vyy, vzz = v6[..., 0], v6[..., 1], v6[..., 2]
        vxy, vxz, vyz = v6[..., 3], v6[..., 4], v6[..., 5]
        row0 = np.stack([vxx, vxy, vxz], axis=-1)
        row1 = np.stack([vxy, vyy, vyz], axis=-1)
        row2 = np.stack([vxz, vyz, vzz], axis=-1)
        return np.stack([row0, row1, row2], axis=-2)
    elif isinstance(v6, torch.Tensor):
        vxx, vyy, vzz = v6[..., 0], v6[..., 1], v6[..., 2]
        vxy, vxz, vyz = v6[..., 3], v6[..., 4], v6[..., 5]
        row0 = torch.stack([vxx, vxy, vxz], dim=-1)
        row1 = torch.stack([vxy, vyy, vyz], dim=-1)
        row2 = torch.stack([vxz, vyz, vzz], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_cartesian_5d(V: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Converts 3x3 symmetric traceless matrix to 5 independent components:
    [Vxx, Vyy, Vxy, Vxz, Vyz]
    (Vzz is implicitly -(Vxx + Vyy)).
    """
    V_st = project_symmetric_traceless(V)
    if isinstance(V_st, np.ndarray):
        return np.stack([
            V_st[..., 0, 0],
            V_st[..., 1, 1],
            V_st[..., 0, 1],
            V_st[..., 0, 2],
            V_st[..., 1, 2]
        ], axis=-1)
    elif isinstance(V_st, torch.Tensor):
        return torch.stack([
            V_st[..., 0, 0],
            V_st[..., 1, 1],
            V_st[..., 0, 1],
            V_st[..., 0, 2],
            V_st[..., 1, 2]
        ], dim=-1)


def cartesian_5d_to_matrix(v5: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Converts 5 independent components back to 3x3 symmetric traceless matrix:
    Vzz = -(Vxx + Vyy)
    """
    if isinstance(v5, np.ndarray):
        vxx, vyy = v5[..., 0], v5[..., 1]
        vxy, vxz, vyz = v5[..., 2], v5[..., 3], v5[..., 4]
        vzz = -(vxx + vyy)
        row0 = np.stack([vxx, vxy, vxz], axis=-1)
        row1 = np.stack([vxy, vyy, vyz], axis=-1)
        row2 = np.stack([vxz, vyz, vzz], axis=-1)
        return np.stack([row0, row1, row2], axis=-2)
    elif isinstance(v5, torch.Tensor):
        vxx, vyy = v5[..., 0], v5[..., 1]
        vxy, vxz, vyz = v5[..., 2], v5[..., 3], v5[..., 4]
        vzz = -(vxx + vyy)
        row0 = torch.stack([vxx, vxy, vxz], dim=-1)
        row1 = torch.stack([vxy, vyy, vyz], dim=-1)
        row2 = torch.stack([vxz, vyz, vzz], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_irrep_2e(V: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Projects 3x3 tensor to l=2 irreducible representation (shape [..., 5])
    using e3nn reduced tensor product basis.
    """
    V_st = project_symmetric_traceless(V)
    if isinstance(V_st, np.ndarray):
        return np.einsum("kij,...ij->...k", _Q_2E_NP, V_st)
    elif isinstance(V_st, torch.Tensor):
        Q = _Q_2E.to(dtype=V_st.dtype, device=V_st.device)
        return torch.einsum("kij,...ij->...k", Q, V_st)


def irrep_2e_to_matrix(v2e: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Reconstructs 3x3 symmetric traceless matrix from l=2 irreducible representation."""
    if isinstance(v2e, np.ndarray):
        return np.einsum("kij,...k->...ij", _Q_2E_NP, v2e)
    elif isinstance(v2e, torch.Tensor):
        Q = _Q_2E.to(dtype=v2e.dtype, device=v2e.device)
        return torch.einsum("kij,...k->...ij", Q, v2e)


def compute_principal_components(
    V: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates physical principal components and asymmetry parameter following convention:
    |Vzz| >= |Vyy| >= |Vxx|
    eta = (Vxx - Vyy) / Vzz

    Returns:
    (Vxx, Vyy, Vzz, eta) as numpy arrays.
    """
    if isinstance(V, torch.Tensor):
        V_np = V.detach().cpu().numpy()
    else:
        V_np = V

    V_sym = (V_np + V_np.swapaxes(-1, -2)) / 2.0

    # Handle batched or single matrix
    if V_sym.ndim == 2:
        evals = np.linalg.eigvalsh(V_sym)
        # Sort by absolute value: |vxx| <= |vyy| <= |vzz|
        order = np.argsort(np.abs(evals))
        vxx, vyy, vzz = evals[order[0]], evals[order[1]], evals[order[2]]
        eta = float(abs(vxx - vyy) / abs(vzz)) if abs(vzz) > 1e-6 else 0.0
        eta = min(1.0, max(0.0, eta))
        return np.array(vxx), np.array(vyy), np.array(vzz), np.array(eta)
    else:
        evals = np.linalg.eigvalsh(V_sym)  # [..., 3]
        order = np.argsort(np.abs(evals), axis=-1)
        vxx = np.take_along_axis(evals, order[..., 0:1], axis=-1).squeeze(-1)
        vyy = np.take_along_axis(evals, order[..., 1:2], axis=-1).squeeze(-1)
        vzz = np.take_along_axis(evals, order[..., 2:3], axis=-1).squeeze(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(np.abs(vzz) > 1e-6, np.abs(vxx - vyy) / np.abs(vzz), 0.0)
            eta = np.clip(eta, 0.0, 1.0)
        return vxx, vyy, vzz, eta
