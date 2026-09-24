"""Evaluates saved predictions from disk without model retraining.

Usage:
    python scripts/evaluate_saved_predictions.py --pred_file <path_to_csv>
    python scripts/evaluate_saved_predictions.py --compare --model_a <path_to_csv> --model_b <path_to_csv>
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features.tensor_transforms import cartesian_6d_to_matrix
from src.evaluation.metrics import compute_task_b_tensor_metrics, compute_crystal_macro_tensor_metrics


def evaluate_prediction_file(filepath: Path) -> Dict[str, Any]:
    df = pd.read_csv(filepath)
    v6_true = df[["Vxx_true", "Vyy_true", "Vzz_true", "Vxy_true", "Vxz_true", "Vyz_true"]].to_numpy()
    v6_pred = df[["Vxx_pred", "Vyy_pred", "Vzz_pred", "Vxy_pred", "Vxz_pred", "Vyz_pred"]].to_numpy()
    jids = df["jid"].tolist()

    V_true = cartesian_6d_to_matrix(v6_true)
    V_pred = cartesian_6d_to_matrix(v6_pred)

    site_metrics = compute_task_b_tensor_metrics(V_true, V_pred)
    macro_metrics = compute_crystal_macro_tensor_metrics(V_true, V_pred, jids)

    combined = {**site_metrics, **macro_metrics}
    return combined


def bootstrap_crystal_paired_difference(
    file_a: Path,
    file_b: Path,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    assert (df_a["jid"] == df_b["jid"]).all(), "JID mismatch between prediction files!"

    v6_t = df_a[["Vxx_true", "Vyy_true", "Vzz_true", "Vxy_true", "Vxz_true", "Vyz_true"]].to_numpy()
    v6_a = df_a[["Vxx_pred", "Vyy_pred", "Vzz_pred", "Vxy_pred", "Vxz_pred", "Vyz_pred"]].to_numpy()
    v6_b = df_b[["Vxx_pred", "Vyy_pred", "Vzz_pred", "Vxy_pred", "Vxz_pred", "Vyz_pred"]].to_numpy()

    Vt = cartesian_6d_to_matrix(v6_t)
    Va = cartesian_6d_to_matrix(v6_a)
    Vb = cartesian_6d_to_matrix(v6_b)

    frob_err_a = np.linalg.norm(Va - Vt, axis=(-2, -1))
    frob_err_b = np.linalg.norm(Vb - Vt, axis=(-2, -1))

    jids = np.array(df_a["jid"].tolist())
    unique_jids = np.unique(jids)

    # Compute crystal-level mean errors
    crystal_err_a = {jid: np.mean(frob_err_a[jids == jid]) for jid in unique_jids}
    crystal_err_b = {jid: np.mean(frob_err_b[jids == jid]) for jid in unique_jids}

    delta_c = np.array([crystal_err_b[j] - crystal_err_a[j] for j in unique_jids])

    # Bootstrap clustered on crystals
    rng = np.random.default_rng(seed)
    boot_means = []
    n_c = len(unique_jids)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_c, size=n_c, replace=True)
        boot_means.append(np.mean(delta_c[idx]))

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "num_crystals": n_c,
        "mean_paired_difference": float(np.mean(delta_c)),
        "median_paired_difference": float(np.median(delta_c)),
        "bootstrap_ci_95_low": float(ci_low),
        "bootstrap_ci_95_high": float(ci_high),
        "ci_excludes_zero": bool(ci_high < 0 or ci_low > 0)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved predictions without model retraining.")
    parser.add_argument("--pred_file", type=str, help="Path to full sitewise prediction CSV file.")
    parser.add_argument("--compare", action="store_true", help="Compare two prediction files with crystal-clustered bootstrap CI.")
    parser.add_argument("--model_a", type=str, help="Baseline model prediction CSV (e.g. B4).")
    parser.add_argument("--model_b", type=str, help="Target model prediction CSV (e.g. B5).")
    args = parser.parse_args()

    if args.compare:
        res = bootstrap_crystal_paired_difference(Path(args.model_a), Path(args.model_b))
        print("=== Paired Comparison Results (Crystal-Clustered Bootstrap) ===")
        for k, v in res.items():
            print(f"{k}: {v}")
    elif args.pred_file:
        res = evaluate_prediction_file(Path(args.pred_file))
        print(f"=== Evaluation Results for {args.pred_file} ===")
        for k, v in res.items():
            print(f"{k}: {v}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
