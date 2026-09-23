"""Automated frozen 5-fold outer evaluation runner for Task B."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import PeriodicCrystalDataset, collate_crystal_graphs
from src.evaluation.metrics import (
    compute_crystal_macro_tensor_metrics,
    compute_task_b_tensor_metrics
)
from src.features.descriptors import extract_site_local_descriptors
from src.features.tensor_transforms import (
    cartesian_5d_to_matrix,
    project_symmetric_traceless
)
from src.models.equivariant_gnn import EquivariantTensorGNN
from src.models.invariant_gnn import InvariantTensorGNN
from src.training.logger import setup_logger
from src.training.losses import FrobeniusLoss
from src.training.seed import set_seed
from src.training.trainer import fit_model


class SiteLocalMLP(torch.nn.Module):
    def __init__(self, in_dim: int = 10, hidden_dim: int = 128, out_dim: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor):
        out_5d = self.net(x)
        V_matrix = cartesian_5d_to_matrix(out_5d)
        return out_5d, V_matrix


def run_5fold_evaluation(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    audit_dir_path: str = "results/adversarial_audit_20260924_023600",
    epochs: int = 5,
    seed: int = 42
):
    set_seed(seed)
    audit_dir = Path(audit_dir_path)
    audit_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("five_fold_evaluation", log_file=str(audit_dir / "five_fold.log"))

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    records_by_jid = {r["jid"]: r for r in raw_data}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting frozen 5-fold evaluation on device: {device}")

    folds = [
        ("fold_0", "splits/tensor_grouped_fold_0.json"),
        ("fold_1", "splits/tensor_grouped_fold_1.json"),
        ("fold_2", "splits/tensor_grouped_fold_2.json"),
        ("fold_3", "splits/tensor_grouped_fold_3.json"),
        ("fold_4", "splits/tensor_grouped_fold_4.json")
    ]

    site_micro_rows = []
    crystal_macro_rows = []
    paired_comparison_rows = []

    for fold_name, fold_path in folds:
        logger.info(f"=== Evaluating {fold_name} ({fold_path}) ===")
        with open(fold_path, "r", encoding="utf-8") as f:
            f_split = json.load(f)

        train_jids = f_split["train"]["jids"]
        val_jids = f_split["val"]["jids"]
        test_jids = f_split["test"]["jids"]

        # Ground truth test tensors
        test_true_list = []
        test_jids_sitewise = []
        for jid in test_jids:
            r = records_by_jid[jid]
            for mat in r["efg_raw_tensor"]:
                test_true_list.append(project_symmetric_traceless(mat))
                test_jids_sitewise.append(jid)
        V_test_true = np.array(test_true_list, dtype=float)
        num_sites = len(V_test_true)
        num_crystals = len(test_jids)

        fold_preds = {}

        # 1. B0: Zero baseline
        fold_preds["B0_zero"] = np.zeros_like(V_test_true)

        # 2. B2: Local MLP baseline
        logger.info(f"[{fold_name}] Computing B2 Local MLP predictions...")
        def get_site_desc(rec, s):
            return extract_site_local_descriptors(
                np.array(rec["atoms"]["lattice_mat"], dtype=float),
                np.array(rec["atoms"]["coords"], dtype=float),
                rec["atoms"]["elements"],
                s
            )

        X_te = torch.from_numpy(np.array([
            get_site_desc(records_by_jid[j], s)
            for j in test_jids
            for s in range(len(records_by_jid[j]["atoms"]["elements"]))
        ], dtype=np.float32)).to(device)

        mlp = SiteLocalMLP().to(device)
        mlp.eval()
        with torch.no_grad():
            _, V_pred_b2 = mlp(X_te)
        fold_preds["B2_local_mlp"] = V_pred_b2.cpu().numpy()

        # 3. Crystal graph loaders
        train_ds = PeriodicCrystalDataset(train_jids, records_by_jid, target_mode="tensor_5d")
        val_ds = PeriodicCrystalDataset(val_jids, records_by_jid, target_mode="tensor_5d")
        test_ds = PeriodicCrystalDataset(test_jids, records_by_jid, target_mode="tensor_5d")

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_crystal_graphs)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_crystal_graphs)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_crystal_graphs)
        criterion = FrobeniusLoss()

        # 4. B4: Invariant GNN (5D projected)
        logger.info(f"[{fold_name}] Training B4 Invariant GNN (5D)...")
        b4_model = InvariantTensorGNN(node_dim=64, hidden_dim=128, num_layers=4, mode="symmetric_traceless_5d").to(device)
        b4_opt = torch.optim.AdamW(b4_model.parameters(), lr=1e-3, weight_decay=1e-5)
        b4_model, _ = fit_model(b4_model, train_loader, val_loader, b4_opt, criterion, device, epochs=epochs, patience=5, task="tensor", logger=logger)
        b4_model.eval()
        b4_preds = []
        with torch.no_grad():
            for batch in test_loader:
                _, V_mat = b4_model(batch.to(device))
                b4_preds.append(V_mat.cpu().numpy())
        fold_preds["B4_invariant_projected_5d"] = np.concatenate(b4_preds, axis=0)

        # 5. B5: Equivariant GNN (e3nn)
        logger.info(f"[{fold_name}] Training B5 Equivariant GNN (e3nn)...")
        b5_model = EquivariantTensorGNN().to(device)
        b5_opt = torch.optim.AdamW(b5_model.parameters(), lr=1e-3, weight_decay=1e-5)
        b5_model, _ = fit_model(b5_model, train_loader, val_loader, b5_opt, criterion, device, epochs=epochs, patience=5, task="tensor", logger=logger)
        b5_model.eval()
        b5_preds = []
        with torch.no_grad():
            for batch in test_loader:
                _, V_mat = b5_model(batch.to(device))
                b5_preds.append(V_mat.cpu().numpy())
        fold_preds["B5_equivariant_e3nn"] = np.concatenate(b5_preds, axis=0)

        # Compute metrics for this fold
        for m_name, pred_mat in fold_preds.items():
            site_m = compute_task_b_tensor_metrics(V_test_true, pred_mat)
            macro_m = compute_crystal_macro_tensor_metrics(V_test_true, pred_mat, test_jids_sitewise)

            site_micro_rows.append({
                "fold": fold_name,
                "model": m_name,
                "seed": seed,
                "num_crystals": num_crystals,
                "num_sites": num_sites,
                "mean_frobenius_micro": site_m["frobenius_error_mean"],
                "median_frobenius_micro": site_m["frobenius_error_median"],
                "normalized_frobenius_error": site_m["normalized_frobenius_error"],
                "Vzz_MAE": site_m["largest_principal_vzz_mae"],
                "eta_MAE": site_m["asymmetry_eta_mae"] if not np.isnan(site_m["asymmetry_eta_mae"]) else "N/A",
                "orientation_error_mean_deg": site_m["principal_axis_angle_mean_deg"] if not np.isnan(site_m["principal_axis_angle_mean_deg"]) else "N/A",
                "valid_eta_count": site_m["valid_eta_count"],
                "valid_orientation_count": site_m["valid_orientation_count"]
            })

            crystal_macro_rows.append({
                "fold": fold_name,
                "model": m_name,
                "seed": seed,
                "num_crystals": num_crystals,
                "num_sites": num_sites,
                "mean_frobenius_macro": macro_m["crystal_macro_frobenius_mean"],
                "median_frobenius_macro": macro_m["crystal_macro_frobenius_median"],
                "p90_frobenius_macro": macro_m["crystal_macro_frobenius_p90"],
                "normalized_frobenius_macro": macro_m["crystal_macro_frobenius_norm"]
            })

        # Paired B5 vs B4 comparison on this fold
        frob_err_b4 = np.linalg.norm(V_test_true - fold_preds["B4_invariant_projected_5d"], axis=(-2, -1))
        frob_err_b5 = np.linalg.norm(V_test_true - fold_preds["B5_equivariant_e3nn"], axis=(-2, -1))

        # Crystal-clustered bootstrapping
        crystal_err_b4 = defaultdict(list)
        crystal_err_b5 = defaultdict(list)
        for jid, e4, e5 in zip(test_jids_sitewise, frob_err_b4, frob_err_b5):
            crystal_err_b4[jid].append(e4)
            crystal_err_b5[jid].append(e5)

        c_mean_b4 = np.array([np.mean(errs) for errs in crystal_err_b4.values()])
        c_mean_b5 = np.array([np.mean(errs) for errs in crystal_err_b5.values()])
        c_paired_diff = c_mean_b5 - c_mean_b4  # negative means B5 is better

        # Bootstrap 1000 resamples
        boot_diffs = []
        n_c = len(c_paired_diff)
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            idx = rng.integers(0, n_c, size=n_c)
            boot_diffs.append(float(np.mean(c_paired_diff[idx])))

        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
        rel_impr = float((np.mean(c_mean_b4) - np.mean(c_mean_b5)) / np.mean(c_mean_b4) * 100)

        paired_comparison_rows.append({
            "fold": fold_name,
            "seed": seed,
            "num_crystals": num_crystals,
            "mean_b4_macro": float(np.mean(c_mean_b4)),
            "mean_b5_macro": float(np.mean(c_mean_b5)),
            "paired_difference_macro": float(np.mean(c_paired_diff)),
            "relative_improvement_pct": rel_impr,
            "bootstrap_95_ci_lower": ci_low,
            "bootstrap_95_ci_upper": ci_high,
            "statistically_significant": bool(ci_high < 0.0)
        })

    # Save all 3 tables
    df_micro = pd.DataFrame(site_micro_rows)
    df_macro = pd.DataFrame(crystal_macro_rows)
    df_paired = pd.DataFrame(paired_comparison_rows)

    df_micro.to_csv(audit_dir / "outer_fold_site_micro_metrics.csv", index=False)
    df_macro.to_csv(audit_dir / "outer_fold_crystal_macro_metrics.csv", index=False)
    df_paired.to_csv(audit_dir / "outer_fold_paired_comparisons.csv", index=False)

    logger.info("Five-fold outer evaluation completed successfully.")


if __name__ == "__main__":
    run_5fold_evaluation()
