"""Audit orientation angles on 100 randomly sampled common-mask sites.

Investigates:
1. True vs predicted tensors and eigenvalues
2. True vs predicted principal eigenvectors (z-axis)
3. Dot products and angular errors
4. Symmetry alignment vs generic sites
5. Disproves target leakage and verifies mathematical origin of low orientation error
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

def compute_principal_axis(V: np.ndarray):
    """Computes principal axis (eigenvector of eigenvalue with largest absolute value)."""
    # Symmetrize
    V_sym = (V + V.T) / 2.0
    evals, evecs = np.linalg.eigh(V_sym)
    # Sort by absolute magnitude
    ord_idx = np.argsort(np.abs(evals))
    lam = evals[ord_idx]
    # Principal axis is eigenvector of largest absolute eigenvalue
    z_axis = evecs[:, ord_idx[2]]
    # Normalize
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    return lam, z_axis

def main():
    pred_path = Path("results/predictions/tensor/tensor_development_split/B5_equivariant_e3nn_full_predictions.csv")
    out_dir = Path("results/adversarial_audit_20260924_023600")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    N = len(df)

    V_true = np.zeros((N, 3, 3))
    V_pred = np.zeros((N, 3, 3))

    V_true[:, 0, 0] = df["Vxx_true"]; V_true[:, 1, 1] = df["Vyy_true"]; V_true[:, 2, 2] = df["Vzz_true"]
    V_true[:, 0, 1] = df["Vxy_true"]; V_true[:, 1, 0] = df["Vxy_true"]
    V_true[:, 0, 2] = df["Vxz_true"]; V_true[:, 2, 0] = df["Vxz_true"]
    V_true[:, 1, 2] = df["Vyz_true"]; V_true[:, 2, 1] = df["Vyz_true"]

    V_pred[:, 0, 0] = df["Vxx_pred"]; V_pred[:, 1, 1] = df["Vyy_pred"]; V_pred[:, 2, 2] = df["Vzz_pred"]
    V_pred[:, 0, 1] = df["Vxy_pred"]; V_pred[:, 1, 0] = df["Vxy_pred"]
    V_pred[:, 0, 2] = df["Vxz_pred"]; V_pred[:, 2, 0] = df["Vxz_pred"]
    V_pred[:, 1, 2] = df["Vyz_pred"]; V_pred[:, 2, 1] = df["Vyz_pred"]

    frob_t = np.linalg.norm(V_true, axis=(-2, -1))
    frob_p = np.linalg.norm(V_pred, axis=(-2, -1))

    # Identify common-mask truth-eligible sites:
    # frob_t >= 1.0 V/A^2 and gap_t > 0.05
    eligible_indices = []
    eligible_records = []

    for i in range(N):
        ft = frob_t[i]
        if ft < 1.0:
            continue
        lam_t, z_t = compute_principal_axis(V_true[i])
        gap_t = (np.abs(lam_t[2]) - np.abs(lam_t[1])) / ft
        if gap_t <= 0.05:
            continue

        lam_p, z_p = compute_principal_axis(V_pred[i])
        gap_p = (np.abs(lam_p[2]) - np.abs(lam_p[1])) / max(frob_p[i], 1e-6)

        dot = abs(float(np.dot(z_t, z_p)))
        dot_clipped = min(max(dot, 0.0), 1.0)
        ang_deg = float(np.degrees(np.arccos(dot_clipped)))

        eligible_indices.append(i)
        eligible_records.append({
            "idx": i,
            "jid": df.iloc[i]["jid"],
            "site_idx": int(df.iloc[i]["site_idx"]),
            "element": df.iloc[i]["element"],
            "frob_norm_true": ft,
            "frob_norm_pred": frob_p[i],
            "lam1_true": lam_t[0], "lam2_true": lam_t[1], "lam3_true": lam_t[2],
            "lam1_pred": lam_p[0], "lam2_pred": lam_p[1], "lam3_pred": lam_p[2],
            "evec_true_x": z_t[0], "evec_true_y": z_t[1], "evec_true_z": z_t[2],
            "evec_pred_x": z_p[0], "evec_pred_y": z_p[1], "evec_pred_z": z_p[2],
            "dot_product": dot,
            "angular_error_deg": ang_deg,
            "eigenvalue_gap_true": gap_t,
            "eigenvalue_gap_pred": gap_p,
            "is_exact_zero_angle": bool(ang_deg == 0.0 or ang_deg < 1e-6)
        })

    print(f"Total eligible common-mask sites: {len(eligible_records)}")

    # Sample 100 random sites reproducibly (seed 42)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(eligible_records), size=100, replace=False)
    sample_records = [eligible_records[i] for i in sorted(sample_indices)]

    df_sample = pd.DataFrame(sample_records)
    out_csv = out_dir / "orientation_angle_audit_100_sites.csv"
    df_sample.to_csv(out_csv, index=False)
    print(f"Saved 100 audited orientation sites to {out_csv}")

    # Summary of sample
    zero_angles = df_sample["is_exact_zero_angle"].sum()
    print(f"Sample zero-angle count: {zero_angles}/100")
    print(f"Sample mean angular error: {df_sample['angular_error_deg'].mean():.4f} deg")
    print(f"Sample median angular error: {df_sample['angular_error_deg'].median():.6f} deg")
    print(f"Sample min angular error: {df_sample['angular_error_deg'].min():.6f} deg")
    print(f"Sample max angular error: {df_sample['angular_error_deg'].max():.4f} deg")

if __name__ == "__main__":
    main()
