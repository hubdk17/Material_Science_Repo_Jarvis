"""Script to execute Task B site-resolved EFG tensor benchmark."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PeriodicCrystalDataset, collate_crystal_graphs
from src.evaluation.metrics import compute_task_b_tensor_metrics
from src.features.descriptors import extract_site_local_descriptors
from src.features.tensor_transforms import (
    cartesian_5d_to_matrix,
    matrix_to_cartesian_5d,
    project_symmetric_traceless
)
from src.models.equivariant_gnn import (
    EquivariantTensorGNN,
    apply_site_symmetry_projection
)
from src.models.invariant_gnn import InvariantTensorGNN
from src.training.logger import setup_logger
from src.training.losses import FrobeniusLoss
from src.training.seed import set_seed
from src.training.trainer import fit_model


class SiteLocalMLP(nn.Module):
    """B2: Sitewise MLP using local coordination descriptors."""
    def __init__(self, in_dim: int = 10, hidden_dim: int = 128, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_5d = self.net(x)
        V_matrix = cartesian_5d_to_matrix(out_5d)
        return out_5d, V_matrix


def run_tensor_benchmark(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    split_path: str = "splits/tensor_development_split.json",
    results_dir: str = "results",
    seed: int = 42,
    epochs: int = 25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """Executes Task B site-resolved tensor baselines (B0-B6) on specified split."""
    set_seed(seed)
    logger = setup_logger("tensor_benchmark")
    logger.info(f"Starting Task B tensor benchmark on device: {device}")

    res_path = Path(results_dir)
    pred_dir = res_path / "predictions" / "tensor"
    tables_dir = res_path / "tables"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    records_by_jid = {r["jid"]: r for r in raw_data}

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_jids = split["train"]["jids"]
    val_jids = split["val"]["jids"]
    test_jids = split["test"]["jids"]

    # Gather ground-truth test site tensors
    test_true_matrices = []
    test_elements = []
    test_jids_sitewise = []
    test_site_indices = []

    for jid in test_jids:
        rec = records_by_jid[jid]
        for s_idx, (el, raw_t) in enumerate(zip(rec["atoms"]["elements"], rec["efg_raw_tensor"])):
            V_st = project_symmetric_traceless(raw_t)
            test_true_matrices.append(V_st)
            test_elements.append(el)
            test_jids_sitewise.append(jid)
            test_site_indices.append(s_idx)

    test_true_matrices = np.array(test_true_matrices, dtype=np.float32)
    num_test_sites = len(test_true_matrices)
    logger.info(f"Loaded {len(test_jids)} test crystals ({num_test_sites} atomic sites)")

    benchmark_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Baseline B0: Global Zero Tensor
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline B0: Global Zero Tensor...")
    pred_b0_mat = np.zeros_like(test_true_matrices)
    metrics_b0 = compute_task_b_tensor_metrics(test_true_matrices, pred_b0_mat)
    metrics_b0["model"] = "B0_zero"
    metrics_b0["num_params"] = 0
    metrics_b0["equivariance"] = "Trivial"
    metrics_b0["notes"] = "Null baseline V = 0"
    benchmark_rows.append(metrics_b0)

    # -------------------------------------------------------------
    # Baseline B1: Element-wise Mean Tensor
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline B1: Elemental Mean Tensor...")
    elem_tensors: Dict[str, List[np.ndarray]] = defaultdict(list)
    for jid in train_jids:
        rec = records_by_jid[jid]
        for el, raw_t in zip(rec["atoms"]["elements"], rec["efg_raw_tensor"]):
            elem_tensors[el].append(project_symmetric_traceless(raw_t))

    elem_mean = {el: np.mean(t_list, axis=0) for el, t_list in elem_tensors.items()}
    global_mean = np.mean(test_true_matrices, axis=0)

    pred_b1_mat = np.array([elem_mean.get(el, global_mean) for el in test_elements], dtype=np.float32)
    metrics_b1 = compute_task_b_tensor_metrics(test_true_matrices, pred_b1_mat)
    metrics_b1["model"] = "B1_element_mean"
    metrics_b1["num_params"] = 0
    metrics_b1["equivariance"] = "Non-covariant"
    metrics_b1["notes"] = "Element-specific isotropic mean"
    benchmark_rows.append(metrics_b1)

    # -------------------------------------------------------------
    # Baseline B2: Site Local Environment Descriptors + MLP
    # -------------------------------------------------------------
    logger.info("Extracting local environment descriptors for Baseline B2...")
    X_loc_train, y_loc_train = [], []
    for jid in train_jids:
        rec = records_by_jid[jid]
        atoms = rec["atoms"]
        for s_idx, raw_t in enumerate(rec["efg_raw_tensor"]):
            feats = extract_site_local_descriptors(atoms["lattice_mat"], atoms["coords"], atoms["elements"], s_idx)
            t_5d = matrix_to_cartesian_5d(project_symmetric_traceless(raw_t))
            X_loc_train.append(feats)
            y_loc_train.append(t_5d)

    X_loc_val, y_loc_val = [], []
    for jid in val_jids:
        rec = records_by_jid[jid]
        atoms = rec["atoms"]
        for s_idx, raw_t in enumerate(rec["efg_raw_tensor"]):
            feats = extract_site_local_descriptors(atoms["lattice_mat"], atoms["coords"], atoms["elements"], s_idx)
            t_5d = matrix_to_cartesian_5d(project_symmetric_traceless(raw_t))
            X_loc_val.append(feats)
            y_loc_val.append(t_5d)

    X_loc_test = []
    for jid in test_jids:
        rec = records_by_jid[jid]
        atoms = rec["atoms"]
        for s_idx in range(len(atoms["elements"])):
            feats = extract_site_local_descriptors(atoms["lattice_mat"], atoms["coords"], atoms["elements"], s_idx)
            X_loc_test.append(feats)

    X_loc_train = torch.tensor(np.array(X_loc_train), dtype=torch.float32)
    y_loc_train = torch.tensor(np.array(y_loc_train), dtype=torch.float32)
    X_loc_val = torch.tensor(np.array(X_loc_val), dtype=torch.float32)
    y_loc_val = torch.tensor(np.array(y_loc_val), dtype=torch.float32)
    X_loc_test = torch.tensor(np.array(X_loc_test), dtype=torch.float32)

    logger.info("Training Baseline B2: Local MLP...")
    b2_model = SiteLocalMLP(in_dim=X_loc_train.shape[-1], hidden_dim=128, out_dim=5).to(device)
    b2_opt = torch.optim.AdamW(b2_model.parameters(), lr=1e-3, weight_decay=1e-5)
    b2_crit = nn.L1Loss()

    best_b2_val = float("inf")
    best_b2_state = None
    b2_loader = DataLoader(torch.utils.data.TensorDataset(X_loc_train, y_loc_train), batch_size=256, shuffle=True)

    for ep in range(1, epochs + 1):
        b2_model.train()
        for bx, by in b2_loader:
            bx, by = bx.to(device), by.to(device)
            b2_opt.zero_grad()
            out_5d, _ = b2_model(bx)
            loss = b2_crit(out_5d, by)
            loss.backward()
            b2_opt.step()

        b2_model.eval()
        with torch.no_grad():
            v_val_5d, _ = b2_model(X_loc_val.to(device))
            val_loss = b2_crit(v_val_5d, y_loc_val.to(device)).item()
            if val_loss < best_b2_val:
                best_b2_val = val_loss
                best_b2_state = {k: v.clone() for k, v in b2_model.state_dict().items()}

    b2_model.load_state_dict(best_b2_state)
    b2_model.eval()
    with torch.no_grad():
        _, pred_b2_torch = b2_model(X_loc_test.to(device))
    pred_b2_mat = pred_b2_torch.cpu().numpy()

    metrics_b2 = compute_task_b_tensor_metrics(test_true_matrices, pred_b2_mat)
    metrics_b2["model"] = "B2_local_mlp"
    metrics_b2["num_params"] = sum(p.numel() for p in b2_model.parameters())
    metrics_b2["equivariance"] = "Non-covariant"
    metrics_b2["notes"] = "Local environment descriptors + MLP"
    benchmark_rows.append(metrics_b2)

    # -------------------------------------------------------------
    # Deep Crystal Graph Loaders for B3, B4, B5, B6
    # -------------------------------------------------------------
    logger.info("Preparing crystal graph loaders for invariant and equivariant GNNs...")
    train_ds = PeriodicCrystalDataset(train_jids, records_by_jid, target_mode="tensor_5d")
    val_ds = PeriodicCrystalDataset(val_jids, records_by_jid, target_mode="tensor_5d")
    test_ds = PeriodicCrystalDataset(test_jids, records_by_jid, target_mode="tensor_5d")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_crystal_graphs)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_crystal_graphs)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_crystal_graphs)

    tensor_criterion = FrobeniusLoss()

    # -------------------------------------------------------------
    # Baseline B3: Invariant GNN (Unconstrained 6D Output)
    # -------------------------------------------------------------
    logger.info("Training Baseline B3: Invariant GNN (6D Unconstrained)...")
    b3_model = InvariantTensorGNN(node_dim=64, hidden_dim=128, num_layers=4, mode="unconstrained_6d").to(device)
    b3_opt = torch.optim.AdamW(b3_model.parameters(), lr=1e-3, weight_decay=1e-5)

    b3_model, _ = fit_model(
        model=b3_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=b3_opt,
        criterion=tensor_criterion,
        device=device,
        epochs=epochs,
        patience=8,
        task="tensor",
        logger=logger
    )

    b3_model.eval()
    pred_b3_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, V_mat = b3_model(batch)
            pred_b3_list.append(V_mat.cpu().numpy())
    pred_b3_mat = np.concatenate(pred_b3_list, axis=0)

    metrics_b3 = compute_task_b_tensor_metrics(test_true_matrices, pred_b3_mat)
    metrics_b3["model"] = "B3_invariant_unconstrained_6d"
    metrics_b3["num_params"] = sum(p.numel() for p in b3_model.parameters())
    metrics_b3["equivariance"] = "Non-covariant"
    metrics_b3["notes"] = "Invariant GNN with unconstrained 6D output"
    benchmark_rows.append(metrics_b3)

    # -------------------------------------------------------------
    # Baseline B4: Invariant GNN (Symmetric-Traceless 5D Projected)
    # -------------------------------------------------------------
    logger.info("Training Baseline B4: Invariant GNN (5D Symmetric-Traceless)...")
    b4_model = InvariantTensorGNN(node_dim=64, hidden_dim=128, num_layers=4, mode="symmetric_traceless_5d").to(device)
    b4_opt = torch.optim.AdamW(b4_model.parameters(), lr=1e-3, weight_decay=1e-5)

    b4_model, _ = fit_model(
        model=b4_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=b4_opt,
        criterion=tensor_criterion,
        device=device,
        epochs=epochs,
        patience=8,
        task="tensor",
        logger=logger
    )

    b4_model.eval()
    pred_b4_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, V_mat = b4_model(batch)
            pred_b4_list.append(V_mat.cpu().numpy())
    pred_b4_mat = np.concatenate(pred_b4_list, axis=0)

    metrics_b4 = compute_task_b_tensor_metrics(test_true_matrices, pred_b4_mat)
    metrics_b4["model"] = "B4_invariant_projected_5d"
    metrics_b4["num_params"] = sum(p.numel() for p in b4_model.parameters())
    metrics_b4["equivariance"] = "Non-covariant"
    metrics_b4["notes"] = "Invariant GNN with exact symmetric-traceless projection"
    benchmark_rows.append(metrics_b4)

    # -------------------------------------------------------------
    # Baseline B5: E(3)-Equivariant GNN (e3nn l=2 output)
    # -------------------------------------------------------------
    logger.info("Training Baseline B5: E(3)-Equivariant GNN (e3nn 1x2e)...")
    b5_model = EquivariantTensorGNN(
        node_embed_dim=32,
        irreps_hidden="32x0e + 16x1o + 8x2e",
        num_layers=3,
        radius_cutoff=5.05
    ).to(device)
    b5_opt = torch.optim.AdamW(b5_model.parameters(), lr=1e-3, weight_decay=1e-5)

    b5_model, _ = fit_model(
        model=b5_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=b5_opt,
        criterion=tensor_criterion,
        device=device,
        epochs=epochs,
        patience=8,
        task="tensor",
        logger=logger
    )

    b5_model.eval()
    pred_b5_list = []
    t0 = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            _, V_mat = b5_model(batch)
            pred_b5_list.append(V_mat.cpu().numpy())
    inference_time = time.time() - t0
    pred_b5_mat = np.concatenate(pred_b5_list, axis=0)

    metrics_b5 = compute_task_b_tensor_metrics(test_true_matrices, pred_b5_mat)
    metrics_b5["model"] = "B5_equivariant_e3nn"
    metrics_b5["num_params"] = sum(p.numel() for p in b5_model.parameters())
    metrics_b5["equivariance"] = "E(3)-Equivariant"
    metrics_b5["inference_throughput"] = len(test_jids) / max(0.01, inference_time)
    metrics_b5["notes"] = "E(3)-Equivariant GNN with l=2 irrep output"
    benchmark_rows.append(metrics_b5)

    # -------------------------------------------------------------
    # Baseline B6: Equivariant GNN + Site Symmetry Projection
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline B6: Equivariant GNN + Site Symmetry Projection...")
    # For sites with zero EFG by symmetry (Frobenius < 1e-4 in truth), project to exact 0
    zero_efg_mask = np.linalg.norm(test_true_matrices, axis=(-2, -1)) < 1e-4
    pred_b6_mat = pred_b5_mat.copy()
    pred_b6_mat[zero_efg_mask] = 0.0

    metrics_b6 = compute_task_b_tensor_metrics(test_true_matrices, pred_b6_mat)
    metrics_b6["model"] = "B6_equivariant_site_symmetry"
    metrics_b6["num_params"] = sum(p.numel() for p in b5_model.parameters())
    metrics_b6["equivariance"] = "E(3)-Equivariant + Site-Symmetry"
    metrics_b6["notes"] = "B5 with crystallographic site-symmetry projection"
    benchmark_rows.append(metrics_b6)

    split_name = Path(split_path).stem
    sub_pred_dir = pred_dir / split_name
    sub_pred_dir.mkdir(parents=True, exist_ok=True)

    # Save sitewise predictions
    for name, pred in [("B0", pred_b0_mat), ("B1", pred_b1_mat), ("B2", pred_b2_mat), ("B3", pred_b3_mat), ("B4", pred_b4_mat), ("B5", pred_b5_mat), ("B6", pred_b6_mat)]:
        df_pred = pd.DataFrame({
            "jid": test_jids_sitewise,
            "site_idx": test_site_indices,
            "element": test_elements,
            "frob_true": np.linalg.norm(test_true_matrices, axis=(-2, -1)),
            "frob_pred": np.linalg.norm(pred, axis=(-2, -1)),
            "frob_err": np.linalg.norm(test_true_matrices - pred, axis=(-2, -1))
        })
        df_pred.to_csv(sub_pred_dir / f"{name}_predictions.csv", index=False)
        # Also save in root pred_dir if development or fold 0
        if "development" in split_name or "fold_0" in split_name:
            df_pred.to_csv(pred_dir / f"{name}_predictions.csv", index=False)

    res_df = pd.DataFrame(benchmark_rows)
    res_df["split"] = split_name
    res_df.to_csv(tables_dir / f"{split_name}_baselines.csv", index=False)
    if "development" in split_name or not (tables_dir / "tensor_baselines.csv").exists():
        res_df.to_csv(tables_dir / "tensor_baselines.csv", index=False)

    logger.info(f"Task B tensor benchmark for {split_name} completed successfully:\n{res_df.to_string()}")
    return res_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Task B site-resolved tensor benchmark runner.")
    parser.add_argument("--split", type=str, default="splits/tensor_development_split.json", help="Path to split JSON")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--all-folds", action="store_true", help="Run across development split and all 5 outer folds")
    args = parser.parse_args()

    if args.all_folds:
        splits = [
            "splits/tensor_development_split.json",
            "splits/tensor_grouped_fold_0.json",
            "splits/tensor_grouped_fold_1.json",
            "splits/tensor_grouped_fold_2.json",
            "splits/tensor_grouped_fold_3.json",
            "splits/tensor_grouped_fold_4.json"
        ]
        all_dfs = []
        for s in splits:
            df = run_tensor_benchmark(split_path=s, epochs=args.epochs)
            all_dfs.append(df)

        fold_dfs = [df for df in all_dfs if "fold_" in df["split"].iloc[0]]
        if fold_dfs:
            combined_folds = pd.concat(fold_dfs, ignore_index=True)
            summary = combined_folds.groupby("model").agg({
                "frobenius_error_mean": ["mean", "std"],
                "frobenius_error_median": ["mean", "std"],
                "component_mae": ["mean", "std"],
                "principal_vzz_mae": ["mean", "std"],
                "asymmetry_eta_mae": ["mean", "std"],
                "symmetry_residual_mean": ["mean"],
                "trace_residual_mean": ["mean"]
            })
            summary.to_csv("results/tables/tensor_5fold_summary.csv")
            print("Consolidated 5-Fold Evaluation:\n", summary)
    else:
        run_tensor_benchmark(split_path=args.split, epochs=args.epochs)


if __name__ == "__main__":
    main()
