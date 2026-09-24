"""Analyzes frame-dependent Cartesian max component vs physical principal max (|Vzz|).

For a symmetric traceless 3x3 tensor V:
- max_cartesian = max_{ij} |V_ij|
- max_principal = max_k |lambda_k| = |Vzz|

Mathematical bounds on the ratio r = |Vzz| / max_{ij} |V_ij|:
1. In a frame diagonalizing V, max_{ij} |V_ij| = |Vzz|, so r = 1.
2. For an arbitrary rotation R in SO(3), let V = R diag(lambda_1, lambda_2, lambda_3) R^T.
   For a uniaxial tensor with eigenvalues (lambda, lambda, -2 lambda), V_ij = lambda (3 u_i u_j - delta_ij).
   The maximum possible ratio r across all symmetric traceless tensors occurs when V_ij components are minimized
   relative to |Vzz|. For 2D tensors, the maximum ratio is sqrt(2). For 3D traceless tensors, the ratio is bounded
   and cannot exceed 1 + sqrt(2) ~ 2.4142.

This script audits all 15,202 crystals and 95,663 sites in JARVIS-EFG4.json.
Outputs:
- results/adversarial_audit_20260924_023600/cartesian_vs_principal_underestimation.csv
- results/adversarial_audit_20260924_023600/cartesian_vs_principal_summary.csv
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features.tensor_transforms import project_symmetric_traceless


def main():
    audit_dir = Path("results/adversarial_audit_20260924_023600")
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    site_rows = []
    structure_stats = []

    for d in data:
        jid = d["jid"]
        raw_tensors = d["efg_raw_tensor"]
        elems = d["atoms"]["elements"]

        struct_cart_max = 0.0
        struct_princ_max = 0.0

        for s_idx, (el, raw_mat) in enumerate(zip(elems, raw_tensors)):
            V = project_symmetric_traceless(raw_mat)
            frob_norm = float(np.linalg.norm(V, "fro"))

            cart_max = float(np.max(np.abs(V)))
            evals = np.linalg.eigvalsh((V + V.T) / 2.0)
            princ_max = float(np.max(np.abs(evals)))

            strict_under = (princ_max > cart_max)
            tol_under = (princ_max - cart_max > 1e-3)
            diff = princ_max - cart_max

            if cart_max > 1e-6:
                ratio = princ_max / cart_max
            else:
                ratio = 1.0 if princ_max <= 1e-6 else np.nan

            site_rows.append({
                "jid": jid,
                "site_idx": s_idx,
                "element": el,
                "frob_norm": frob_norm,
                "cartesian_max": cart_max,
                "principal_max": princ_max,
                "difference": diff,
                "strict_underestimation": strict_under,
                "tol_filtered_underestimation": tol_under,
                "ratio": ratio
            })

            struct_cart_max = max(struct_cart_max, cart_max)
            struct_princ_max = max(struct_princ_max, princ_max)

        strict_struct = (struct_princ_max > struct_cart_max)
        tol_struct = (struct_princ_max - struct_cart_max > 1e-3)
        diff_struct = struct_princ_max - struct_cart_max
        ratio_struct = (struct_princ_max / struct_cart_max) if struct_cart_max > 1e-6 else 1.0

        structure_stats.append({
            "jid": jid,
            "struct_cart_max": struct_cart_max,
            "struct_princ_max": struct_princ_max,
            "difference": diff_struct,
            "strict_underestimation": strict_struct,
            "tol_filtered_underestimation": tol_struct,
            "ratio": ratio_struct
        })

    df_sites = pd.DataFrame(site_rows)
    df_structs = pd.DataFrame(structure_stats)

    # Filter for non-trivial tensors for summary statistics
    nontrivial_sites = df_sites[df_sites["frob_norm"] > 1.0]
    nontrivial_structs = df_structs[df_structs["struct_cart_max"] > 1.0]

    summary = [
        {
            "granularity": "Atomic Sites (All 95,663 sites)",
            "total_count": len(df_sites),
            "strict_underestimation_count": int(df_sites["strict_underestimation"].sum()),
            "strict_underestimation_fraction": float(df_sites["strict_underestimation"].mean()),
            "tol_underestimation_count": int(df_sites["tol_filtered_underestimation"].sum()),
            "tol_underestimation_fraction": float(df_sites["tol_filtered_underestimation"].mean()),
            "mean_underestimation_V_A2": float(df_sites[df_sites["strict_underestimation"]]["difference"].mean()),
            "max_underestimation_V_A2": float(df_sites["difference"].max()),
            "max_ratio": float(nontrivial_sites["ratio"].max()),
            "p95_ratio": float(nontrivial_sites["ratio"].quantile(0.95))
        },
        {
            "granularity": "Crystal Structures (All 15,202 crystals)",
            "total_count": len(df_structs),
            "strict_underestimation_count": int(df_structs["strict_underestimation"].sum()),
            "strict_underestimation_fraction": float(df_structs["strict_underestimation"].mean()),
            "tol_underestimation_count": int(df_structs["tol_filtered_underestimation"].sum()),
            "tol_underestimation_fraction": float(df_structs["tol_filtered_underestimation"].mean()),
            "mean_underestimation_V_A2": float(df_structs[df_structs["strict_underestimation"]]["difference"].mean()),
            "max_underestimation_V_A2": float(df_structs["difference"].max()),
            "max_ratio": float(nontrivial_structs["ratio"].max()),
            "p95_ratio": float(nontrivial_structs["ratio"].quantile(0.95))
        }
    ]

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(audit_dir / "cartesian_vs_principal_summary.csv", index=False)
    print("=== Cartesian vs Principal Underestimation Summary ===")
    print(df_summary.to_string())

    # Save per-site parquet/csv (first 10,000 to CSV, full to summary)
    df_sites.to_csv(audit_dir / "cartesian_vs_principal_underestimation.csv", index=False)
    print(f"Saved site-level audit to {audit_dir / 'cartesian_vs_principal_underestimation.csv'}")


if __name__ == "__main__":
    main()
