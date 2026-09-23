"""Uncertainty quantification, calibration, and selective prediction analysis."""

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    nominal_coverages: List[float] = [0.50, 0.68, 0.80, 0.90, 0.95]
) -> Dict[str, Any]:
    """Computes empirical interval coverage, interval widths, and calibration error.

    For Gaussian prediction intervals [mu - z * sigma, mu + z * sigma].
    """
    from scipy.stats import norm

    y_t = np.asarray(y_true, dtype=float).ravel()
    mu = np.asarray(y_pred_mean, dtype=float).ravel()
    sigma = np.asarray(y_pred_std, dtype=float).ravel()
    sigma = np.maximum(sigma, 1e-6)

    empirical_coverages = []
    mean_widths = []
    calibration_errors = []

    for nom in nominal_coverages:
        alpha = 1.0 - nom
        z = norm.ppf(1.0 - alpha / 2.0)
        lower = mu - z * sigma
        upper = mu + z * sigma

        covered = (y_t >= lower) & (y_t <= upper)
        emp = float(np.mean(covered))
        width = float(np.mean(upper - lower))

        empirical_coverages.append(emp)
        mean_widths.append(width)
        calibration_errors.append(abs(emp - nom))

    expected_calib_error = float(np.mean(calibration_errors))

    return {
        "nominal_coverages": nominal_coverages,
        "empirical_coverages": empirical_coverages,
        "mean_interval_widths": mean_widths,
        "expected_calibration_error": expected_calib_error,
        "coverage_68": empirical_coverages[1] if len(empirical_coverages) > 1 else None,
        "coverage_95": empirical_coverages[4] if len(empirical_coverages) > 4 else None
    }


def compute_error_retention_curve(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    fractions_retained: np.ndarray = np.linspace(0.1, 1.0, 19)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes error-retention curve: sorts predictions by ascending uncertainty,
    evaluating mean error as higher-uncertainty predictions are rejected.

    Returns:
    - fractions: retained sample fractions
    - retained_errors: mean error on retained subset
    - retention_auc: area under the retention curve
    """
    err = np.asarray(errors, dtype=float).ravel()
    unc = np.asarray(uncertainties, dtype=float).ravel()

    # Sort ascending by uncertainty (keep most confident first)
    sort_idx = np.argsort(unc)
    sorted_err = err[sort_idx]
    N = len(sorted_err)

    retained_errors = []
    for f in fractions_retained:
        k = max(1, int(round(f * N)))
        retained_errors.append(float(np.mean(sorted_err[:k])))

    retained_errors = np.array(retained_errors)
    auc = float(np.trapezoid(retained_errors, fractions_retained))

    return fractions_retained, retained_errors, auc
