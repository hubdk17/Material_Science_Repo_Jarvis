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

    # Track exclusion reasons
    excluded_degenerate_true = 0
    excluded_degenerate_pred = 0
    excluded_low_magnitude = 0
    excluded_undefined_b0 = 0

    is_b0 = bool(np.all(np.abs(V_pred_mat) < 1e-6))

    for i in range(N):
        Vt = V_true_mat[i]
        Vp = V_pred_mat[i]
        frob_t = float(true_frob[i])
        frob_p = float(np.linalg.norm(Vp, 'fro'))

        vxx_t, vyy_t, vzz_t, eta_t = compute_principal_components(Vt)

        # Eigenvalues sorted by absolute magnitude: |lambda_1| <= |lambda_2| <= |lambda_3|
        evals_t = np.linalg.eigvalsh((Vt + Vt.T) / 2.0)
        ord_t = np.argsort(np.abs(evals_t))
        lam_t = evals_t[ord_t]

        if is_b0:
            vzz_p = 0.0
            eta_p = np.nan
            evals_p = np.zeros(3)
            excluded_undefined_b0 += 1
        else:
            vxx_p, vyy_p, vzz_p, eta_p = compute_principal_components(Vp)
            evals_p = np.linalg.eigvalsh((Vp + Vp.T) / 2.0)
            ord_p = np.argsort(np.abs(evals_p))
            lam_p = evals_p[ord_p]

        vzz_errs.append(abs(vzz_t - vzz_p))

        # Sorted eigenvalue set error (permutation-invariant set error)
        eval_set_errs.append(float(np.mean(np.abs(np.sort(evals_t) - np.sort(evals_p)))))

        # Asymmetry parameter eta: valid only when |Vzz_true| > 1.0 and |Vzz_pred| > 1.0
        if not is_b0 and abs(vzz_t) > 1.0 and abs(vzz_p) > 1.0:
            eta_errs.append(abs(eta_t - eta_p))

        # Principal-axis angular error with scale-aware degeneracy check
        if is_b0:
            continue

        if frob_t < 1.0 or frob_p < 1.0:
            excluded_low_magnitude += 1
            continue

        # Scale-aware relative gap: min(|lam_3| - |lam_2|, |lam_2| - |lam_1|) / ||V||_F
        gap_t = (np.abs(lam_t[2]) - np.abs(lam_t[1])) / max(frob_t, 1e-6)
        gap_p = (np.abs(lam_p[2]) - np.abs(lam_p[1])) / max(frob_p, 1e-6)

        tau = 0.05  # Scale-aware non-degeneracy threshold
        if gap_t <= tau:
            excluded_degenerate_true += 1
            continue
        if gap_p <= tau:
            excluded_degenerate_pred += 1
            continue

        _, evecs_t = np.linalg.eigh((Vt + Vt.T) / 2.0)
        _, evecs_p = np.linalg.eigh((Vp + Vp.T) / 2.0)
        z_axis_t = evecs_t[:, ord_t[2]]
        z_axis_p = evecs_p[:, ord_p[2]]

        # Angle between undirected axes: cos(theta) = |u . v|
        cos_theta = np.clip(abs(np.dot(z_axis_t, z_axis_p)), 0.0, 1.0)
        axis_angular_errs.append(float(np.degrees(np.arccos(cos_theta))))

    return {
        "frobenius_error_mean": mean_frob,
        "frobenius_error_median": median_frob,
        "normalized_frobenius_error": float(norm_frob),
        "component_mae": comp_mae,
        "mean_symmetry_residual": mean_sym_residual,
        "mean_trace_residual": mean_trace_residual,
        "largest_principal_vzz_mae": float(np.mean(vzz_errs)) if vzz_errs else 0.0,
        "asymmetry_eta_mae": float(np.mean(eta_errs)) if eta_errs else np.nan,
        "eigenvalue_set_mae": float(np.mean(eval_set_errs)) if eval_set_errs else 0.0,
        "principal_axis_angle_mean_deg": float(np.mean(axis_angular_errs)) if axis_angular_errs else np.nan,
        "principal_axis_angle_median_deg": float(np.median(axis_angular_errs)) if axis_angular_errs else np.nan,
        "principal_axis_angle_p90_deg": float(np.percentile(axis_angular_errs, 90)) if axis_angular_errs else np.nan,
        "valid_eta_count": len(eta_errs),
        "valid_orientation_count": len(axis_angular_errs),
        "excluded_orientation_count": N - len(axis_angular_errs),
        "excluded_degenerate_true": excluded_degenerate_true,
        "excluded_degenerate_pred": excluded_degenerate_pred,
        "excluded_low_magnitude": excluded_low_magnitude,
        "excluded_undefined_b0": excluded_undefined_b0
    }


def compute_crystal_macro_tensor_metrics(
    V_true_mat: np.ndarray,
    V_pred_mat: np.ndarray,
    jids: List[str]
) -> Dict[str, float]:
    """Computes crystal-macro metrics where each crystal has equal weight."""
    from collections import defaultdict
    diff_mat = V_true_mat - V_pred_mat
    site_frob_err = np.linalg.norm(diff_mat, axis=(-2, -1))
    site_true_frob = np.linalg.norm(V_true_mat, axis=(-2, -1))

    crystal_errors = defaultdict(list)
    crystal_true = defaultdict(list)

    for jid, err, tr in zip(jids, site_frob_err, site_true_frob):
        crystal_errors[jid].append(err)
        crystal_true[jid].append(tr)

    crystal_mean_errs = [float(np.mean(errs)) for errs in crystal_errors.values()]
    crystal_mean_true = [float(np.mean(trs)) for trs in crystal_true.values()]

    return {
        "crystal_macro_frobenius_mean": float(np.mean(crystal_mean_errs)),
        "crystal_macro_frobenius_median": float(np.median(crystal_mean_errs)),
        "crystal_macro_frobenius_p90": float(np.percentile(crystal_mean_errs, 90)),
        "crystal_macro_frobenius_norm": float(np.sum(crystal_mean_errs) / (np.sum(crystal_mean_true) + 1e-6)),
        "num_crystals": len(crystal_mean_errs)
    }

