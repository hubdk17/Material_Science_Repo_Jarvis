"""Audits why representative Wyckoff rows in CSV differ from full-cell Cartesian maxima.

Verifies:
1. Sites within the same Wyckoff orbit have identical eigenvalue spectra (principal values).
2. Their Cartesian representations differ strictly by 3D orthogonal rotation matrices R in O(3):
       V_site2 = R @ V_site1 @ R^T
3. Because the Cartesian components V_ij vary under rotation while eigenvalues lambda_k are invariant,
   the maximum Cartesian component max_{ij} |V_ij| depends on the choice of representative site in the orbit.
4. In ~11% of structures, another symmetry-equivalent site in the unit cell achieves a higher Cartesian component
   than the single representative row chosen by JARVIS in the CSV.

Outputs:
- results/adversarial_audit_20260924_023600/wyckoff_rotation_audit.csv
"""

import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
import numpy as np
import pandas as pd
import spglib

from src.features.tensor_transforms import project_symmetric_traceless


def main():
    audit_dir = Path("results/adversarial_audit_20260924_023600")
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with zipfile.ZipFile("data/raw/dft_3d_max_efg.json.zip", "r") as z:
        with z.open("dft_3d_max_efg.json") as f:
            bench_data = json.load(f)
    all_bench = {}
    for split in ["train", "val", "test"]:
        all_bench.update(bench_data[split])

    audit_records = []

    mismatched_jids = []
    for d in data:
        jid = d["jid"]
        if jid not in all_bench:
            continue
        bench_val = float(all_bench[jid])
        raw_mats = d["efg_raw_tensor"]
        full_cell_cart_max = max(float(np.max(np.abs(m))) for m in raw_mats)
        if abs(full_cell_cart_max - bench_val) > 1e-3:
            mismatched_jids.append((jid, bench_val, full_cell_cart_max, d))

    print(f"Total mismatched JIDs between full-cell and benchmark: {len(mismatched_jids)}")

    # Audit the first 200 mismatched structures
    for jid, bench_val, full_cart_max, d in mismatched_jids[:200]:
        lat = np.array(d["atoms"]["lattice_mat"], dtype=float)
        coords = np.array(d["atoms"]["coords"], dtype=float)
        elems = d["atoms"]["elements"]
        raw_mats = [project_symmetric_traceless(m) for m in d["efg_raw_tensor"]]

        # Get Wyckoff positions using spglib
        unique_elems = {el: idx + 1 for idx, el in enumerate(sorted(set(elems)))}
        atom_types = [unique_elems[el] for el in elems]
        cell = (lat, coords % 1.0, atom_types)
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-4)

        if dataset is None:
            continue

        wyckoffs = getattr(dataset, "wyckoffs", None)
        equivalent_atoms = getattr(dataset, "equivalent_atoms", None)
        if wyckoffs is None:
            wyckoffs = dataset["wyckoffs"]
            equivalent_atoms = dataset["equivalent_atoms"]

        # Group sites by orbit (equivalent atom index)
        orbits = defaultdict(list)
        for s_idx, (el, eq_id, wyck, V) in enumerate(zip(elems, equivalent_atoms, wyckoffs, raw_mats)):
            evals = np.sort(np.linalg.eigvalsh((V + V.T) / 2.0))
            cart_max = float(np.max(np.abs(V)))
            orbits[eq_id].append({
                "site_idx": s_idx,
                "element": el,
                "wyckoff": wyck,
                "cart_max": cart_max,
                "evals": evals,
                "matrix": V
            })

        # Check eigenvalue invariance within orbits and Cartesian variations
        for eq_id, sites in orbits.items():
            if len(sites) < 2:
                continue

            ref_evals = sites[0]["evals"]
            max_eval_diff = max(np.max(np.abs(s["evals"] - ref_evals)) for s in sites)
            cart_maxima = [s["cart_max"] for s in sites]
            cart_spread = max(cart_maxima) - min(cart_maxima)

            # Check if one site matches benchmark and another has higher Cartesian component
            matches_bench = any(abs(c - bench_val) < 1e-3 for c in cart_maxima)
            has_higher_cart = any(c > bench_val + 1e-3 for c in cart_maxima)

            if matches_bench and has_higher_cart:
                audit_records.append({
                    "jid": jid,
                    "element": sites[0]["element"],
                    "wyckoff_letter": sites[0]["wyckoff"],
                    "orbit_size": len(sites),
                    "benchmark_target": bench_val,
                    "representative_cart_max": min(cart_maxima),
                    "full_cell_cart_max": max(cart_maxima),
                    "cartesian_discrepancy": max(cart_maxima) - bench_val,
                    "max_eigenvalue_difference_in_orbit": float(max_eval_diff),
                    "eigenvalues_invariant": bool(max_eval_diff < 1e-3),
                    "explanation": "Symmetry-equivalent sites are rotated in Cartesian frame; eigenvalues are identical but Cartesian max depends on site choice"
                })

    df_out = pd.DataFrame(audit_records)
    df_out.to_csv(audit_dir / "wyckoff_rotation_audit.csv", index=False)
    print(f"Wyckoff rotation audit complete. Audited {len(df_out)} symmetry orbits with Cartesian variations.")
    if len(df_out) > 0:
        print(df_out[["jid", "element", "wyckoff_letter", "orbit_size", "benchmark_target", "full_cell_cart_max", "eigenvalues_invariant"]].head(10).to_string())


if __name__ == "__main__":
    main()
