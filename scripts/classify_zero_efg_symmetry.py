"""Group-theoretic classification of zero and near-zero EFG sites in JARVIS-EFG4.

Uses spglib to extract the exact site stabilizer group G_i for every site,
constructs the 5D representation of G_i acting on symmetric traceless rank-2 tensors,
and determines the dimension of the invariant subspace:
    dim(Inv(G_i)) = Tr( (1 / |G_i|) * sum_{R in G_i} M_5(R) )

A site is strictly symmetry-enforced to have EFG = 0 if and only if dim(Inv(G_i)) == 0.
Sites with dim == 0 correspond to cubic site symmetries (m-3m, -43m, 432, m-3, 23).
Sites with dim == 1 have uniaxial symmetry (e.g. 4/mmm, -3m, 6/mmm, etc., where eta = 0).

Outputs:
- results/adversarial_audit_20260924_023600/zero_efg_symmetry_classification.csv
- results/adversarial_audit_20260924_023600/zero_efg_classification_summary.csv
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter
import numpy as np
import pandas as pd
import spglib

from src.features.site_symmetry import get_site_stabilizers, classify_site_efg_symmetry
from src.features.tensor_transforms import project_symmetric_traceless


def main():
    audit_dir = Path("results/adversarial_audit_20260924_023600")
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    site_records = []

    print("Classifying site symmetry for all structures...")
    for idx, d in enumerate(data):
        jid = d["jid"]
        lat = np.array(d["atoms"]["lattice_mat"], dtype=float)
        coords = np.array(d["atoms"]["coords"], dtype=float)
        elems = d["atoms"]["elements"]
        raw_tensors = d["efg_raw_tensor"]

        stabilizers = get_site_stabilizers(lat, coords, elems)

        for s_idx, (el, raw_mat, ops) in enumerate(zip(elems, raw_tensors, stabilizers)):
            V = project_symmetric_traceless(raw_mat)
            frob_norm = float(np.linalg.norm(V, "fro"))

            is_forced_zero, inv_dim, desc = classify_site_efg_symmetry(ops)
            num_ops = len(ops)

            # Categorize
            if is_forced_zero and frob_norm < 1e-4:
                category = "symmetry_forced_exact_zero"
            elif is_forced_zero and frob_norm >= 1e-4:
                category = "symmetry_forced_near_zero"
            elif not is_forced_zero and frob_norm < 1.0:
                category = "numerical_near_zero_unforced"
            else:
                category = "finite_efg"

            # Only store detailed rows for near-zero / symmetry-forced sites or sample
            if frob_norm < 1.0 or is_forced_zero:
                site_records.append({
                    "jid": jid,
                    "site_idx": s_idx,
                    "element": el,
                    "stabilizer_order": num_ops,
                    "invariant_subspace_dim": inv_dim,
                    "is_symmetry_enforced_zero": is_forced_zero,
                    "frob_norm": frob_norm,
                    "category": category
                })

    df = pd.DataFrame(site_records)
    df.to_csv(audit_dir / "zero_efg_symmetry_classification.csv", index=False)

    # Compute summary
    cat_counts = Counter(df["category"])
    summary_rows = []
    for cat, count in cat_counts.items():
        sub_df = df[df["category"] == cat]
        summary_rows.append({
            "category": cat,
            "count": count,
            "mean_frob_norm": float(sub_df["frob_norm"].mean()),
            "max_frob_norm": float(sub_df["frob_norm"].max()),
            "mean_stabilizer_order": float(sub_df["stabilizer_order"].mean())
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(audit_dir / "zero_efg_classification_summary.csv", index=False)
    print("=== Zero-EFG Group Theory Classification Summary ===")
    print(df_sum.to_string())


if __name__ == "__main__":
    main()
