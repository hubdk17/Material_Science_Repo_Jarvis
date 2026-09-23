"""Evaluation metrics for Task A (scalar) and Task B (site-resolved tensor)."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from src.features.tensor_transforms import (
    cartesian_5d_to_matrix,
    compute_principal_components,
    project_symmetric_traceless
)


def compute_task_a_scalar_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_mad: Optional[float] = None,
    high_efg_threshold: float = 50.0
) -> Dict[str, float]:
    """Computes all publication-grade metrics for Task A scalar max-EFG prediction."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    med_ae = float(np.median(np.abs(y_t - y_p)))
    r2 = float(r2_score(y_t, y_p))

    nmae = mae / train_mad if (train_mad is not None and train_mad > 0) else None

    # Error by target quartiles
    q25, q50, q75 = np.percentile(y_t, [25, 50, 75])
    mae_q1 = float(np.mean(np.abs(y_t[y_t <= q25] - y_p[y_t <= q25]))) if np.any(y_t <= q25) else 0.0
    mae_q4 = float(np.mean(np.abs(y_t[y_t >= q75] - y_p[y_t >= q75]))) if np.any(y_t >= q75) else 0.0

    # High-EFG candidate ranking precision and recall (top candidates)
    actual_high = y_t >= high_efg_threshold
    pred_high = y_p >= high_efg_threshold

    tp = np.sum(actual_high & pred_high)
    fp = np.sum(~actual_high & pred_high)
    fn = np.sum(actual_high & ~pred_high)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "median_ae": med_ae,
        "r2": r2,
        "mae_q1": mae_q1,
        "mae_q4": mae_q4,
        "high_efg_precision": precision,
        "high_efg_recall": recall,
        "high_efg_f1": f1
    }
    if nmae is not None:
        metrics["nmae_mad"] = nmae

    return metrics


def compute_task_b_tensor_metrics(
    V_true: Union[np.ndarray, List[np.ndarray]],
    V_pred: Union[np.ndarray, List[np.ndarray]],
    degeneracy_threshold: float = 1.0  # Threshold |lambda_3 - lambda_2| in V/A^2
) -> Dict[str, float]:
    """Computes publication-grade site-resolved tensor evaluation metrics.

    Parameters:
    - V_true: [N_sites, 3, 3] or [N_sites, 5] ground truth tensors
    - V_pred: [N_sites, 3, 3] or [N_sites, 5] predicted tensors
    - degeneracy_threshold: minimum difference between largest eigenvalues to define non-degenerate principal axes
    """
    if isinstance(V_true, list):
        V_true = np.array(V_true, dtype=float)
    if isinstance(V_pred, list):
        V_pred = np.array(V_pred, dtype=float)

    # Convert 5D to 3x3 if necessary
    if V_true.ndim == 2 and V_true.shape[-1] == 5:
        V_true_mat = cartesian_5d_to_matrix(V_true)
    else:
        V_true_mat = V_true

    if V_pred.ndim == 2 and V_pred.shape[-1] == 5:
        V_pred_mat = cartesian_5d_to_matrix(V_pred)
    else:
        V_pred_mat = V_pred

    N = V_true_mat.shape[0]

    # 1. Frobenius error per site
    diff_mat = V_true_mat - V_pred_mat
    frob_err = np.linalg.norm(diff_mat, axis=(-2, -1))  # [N]
    mean_frob = float(np.mean(frob_err))
    median_frob = float(np.median(frob_err))

    true_frob = np.linalg.norm(V_true_mat, axis=(-2, -1))  # [N]
    norm_frob = np.sum(frob_err) / (np.sum(true_frob) + 1e-6)

    # 2. Symmetry and trace residuals of predictions
    sym_residuals = np.linalg.norm(V_pred_mat - V_pred_mat.swapaxes(-1, -2), axis=(-2, -1))
    mean_sym_residual = float(np.mean(sym_residuals))

    trace_residuals = np.abs(np.trace(V_pred_mat, axis1=-2, axis2=-1))
    mean_trace_residual = float(np.mean(trace_residuals))

    # 3. Component MAE
    comp_mae = float(np.mean(np.abs(diff_mat)))

    # 4. Principal components & eigenvalue metrics
    vzz_errs = []
    eta_errs = []
    eval_set_errs = []
    axis_angular_errs = []

    for i in range(N):
        Vt = V_true_mat[i]
        Vp = V_pred_mat[i]

        vxx_t, vyy_t, vzz_t, eta_t = compute_principal_components(Vt)
        vxx_p, vyy_p, vzz_p, eta_p = compute_principal_components(Vp)

        vzz_errs.append(abs(vzz_t - vzz_p))
        if abs(vzz_t) > 1e-3 and abs(vzz_p) > 1e-3:
            eta_errs.append(abs(eta_t - eta_p))

        # Sorted eigenvalue set error
        evals_t = np.sort(np.linalg.eigvalsh((Vt + Vt.T) / 2.0))
        evals_p = np.sort(np.linalg.eigvalsh((Vp + Vp.T) / 2.0))
        eval_set_errs.append(float(np.mean(np.abs(evals_t - evals_p))))

        # Principal-axis angular error for non-degenerate tensors
        # Non-degenerate if |lambda_3| - |lambda_2| > degeneracy_threshold
        evals_t_abs = np.abs(evals_t)
        sort_order = np.argsort(evals_t_abs)
        if (evals_t_abs[sort_order[2]] - evals_t_abs[sort_order[1]]) > degeneracy_threshold:
            _, evecs_t = np.linalg.eigh((Vt + Vt.T) / 2.0)
            _, evecs_p = np.linalg.eigh((Vp + Vp.T) / 2.0)
            # Largest principal axis is the eigenvector of |Vzz|
            z_axis_t = evecs_t[:, sort_order[2]]
            z_axis_p = evecs_p[:, np.argsort(np.abs(evals_p))[2]]
            # Angle between undirected axes: cos(theta) = |u . v|
            cos_theta = np.clip(abs(np.dot(z_axis_t, z_axis_p)), 0.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_theta))
            axis_angular_errs.append(angle_deg)

    return {
        "frobenius_error_mean": mean_frob,
        "frobenius_error_median": median_frob,
        "normalized_frobenius_error": float(norm_frob),
        "component_mae": comp_mae,
        "mean_symmetry_residual": mean_sym_residual,
        "mean_trace_residual": mean_trace_residual,
        "largest_principal_vzz_mae": float(np.mean(vzz_errs)) if vzz_errs else 0.0,
        "asymmetry_eta_mae": float(np.mean(eta_errs)) if eta_errs else 0.0,
        "eigenvalue_set_mae": float(np.mean(eval_set_errs)) if eval_set_errs else 0.0,
        "principal_axis_angle_deg": float(np.mean(axis_angular_errs)) if axis_angular_errs else 0.0,
        "non_degenerate_fraction": float(len(axis_angular_errs) / N)
    }
