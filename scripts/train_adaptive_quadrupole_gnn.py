"""Training and evaluation pipeline for the Adaptive Quadrupole GNN (AQ-GNN).

Evaluation Protocol:
- Strictly adheres to Protocol A on the official JARVIS benchmark split:
  splits/official_jarvis_scalar.json (9,492 train, 1,186 val, 1,186 test).
- Validation is used strictly for early stopping and model checkpoint selection.
- Test set is evaluated exactly once on the frozen best checkpoint.
- Zero test data leakage: test site tensors are never accessed or used.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from src.data.dataset import collate_crystal_graphs
from src.evaluation.metrics import compute_task_a_scalar_metrics
from src.features.graph import build_periodic_crystal_graph
from src.models.adaptive_quadrupole_gnn import AdaptiveQuadrupoleGNN
from src.training.logger import setup_logger
from src.training.seed import set_seed


class AQCrystalDataset(Dataset):
    """Dataset for AQ-GNN with optional site supervision on training crystals."""

    def __init__(
        self,
        jids: List[str],
        records_by_jid: Dict[str, Dict[str, Any]],
        scalar_targets: Dict[str, float],
        radius_cutoff: float = 6.0,
        max_neighbors: int = 16,
        is_training: bool = False
    ):
        self.jids = jids
        self.records_by_jid = records_by_jid
        self.scalar_targets = scalar_targets
        self.radius_cutoff = radius_cutoff
        self.max_neighbors = max_neighbors
        self.is_training = is_training
        self._graph_cache: Dict[str, Data] = {}

    def preload(self) -> "AQCrystalDataset":
        for i in range(len(self.jids)):
            _ = self[i]
        return self

    def __len__(self) -> int:
        return len(self.jids)

    def __getitem__(self, idx: int) -> Data:
        jid = self.jids[idx]
        if jid in self._graph_cache:
            return self._graph_cache[jid].clone()

        rec = self.records_by_jid[jid]
        lat = np.array(rec["atoms"]["lattice_mat"], dtype=float)
        frac_coords = np.array(rec["atoms"]["coords"], dtype=float)
        elements = rec["atoms"]["elements"]

        target_scalar = float(self.scalar_targets[jid])

        data = build_periodic_crystal_graph(
            lattice_mat=lat,
            fractional_coords=frac_coords,
            elements=elements,
            radius_cutoff=self.radius_cutoff,
            max_neighbors=self.max_neighbors,
            target_scalar=target_scalar
        )
        data.jid = jid

        # If training partition, extract true site max EFG for auxiliary supervision
        if self.is_training and "efg_raw_tensor" in rec:
            raw_tensors = rec["efg_raw_tensor"]
            site_maxes = [float(np.max(np.abs(np.array(t)))) for t in raw_tensors]
            data.y_site = torch.tensor(site_maxes, dtype=torch.float32)

        self._graph_cache[jid] = data
        return data


def train_aq_gnn(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    split_path: str = "splits/official_jarvis_scalar.json",
    results_dir: str = "results",
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_quad_channels: int = 4,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    site_loss_weight: float = 0.3,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    pooling: str = "smoothmax",
    loss_mode: str = "smooth_l1_rel",
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """Trains AQ-GNN and evaluates strictly on the official test set."""
    set_seed(seed)
    logger = setup_logger("train_aq_gnn")
    logger.info(f"Starting Adaptive Quadrupole GNN (AQ-GNN) Training on device: {device}")
    logger.info(f"Hyperparameters: hidden_dim={hidden_dim}, num_layers={num_layers}, num_quad_channels={num_quad_channels}, pooling={pooling}, loss_mode={loss_mode}, tag={tag}")

    res_path = Path(results_dir)
    ckpt_dir = res_path / "checkpoints" / "scalar"
    pred_dir = res_path / "predictions" / "scalar"
    tables_dir = res_path / "tables"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
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

    y_train = np.array([split["train"]["targets"][j] for j in train_jids], dtype=np.float32)
    y_val = np.array([split["val"]["targets"][j] for j in val_jids], dtype=np.float32)
    y_test = np.array([split["test"]["targets"][j] for j in test_jids], dtype=np.float32)

    train_median = np.median(y_train)
    train_mad = float(np.median(np.abs(y_train - train_median)))
    logger.info(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")
    logger.info(f"Train MAD: {train_mad:.4f}")

    # Build datasets
    logger.info("Constructing graph datasets...")
    train_dataset = AQCrystalDataset(
        train_jids, records_by_jid, split["train"]["targets"],
        radius_cutoff=6.0, max_neighbors=16, is_training=True
    ).preload()
    val_dataset = AQCrystalDataset(
        val_jids, records_by_jid, split["val"]["targets"],
        radius_cutoff=6.0, max_neighbors=16, is_training=False
    ).preload()
    test_dataset = AQCrystalDataset(
        test_jids, records_by_jid, split["test"]["targets"],
        radius_cutoff=6.0, max_neighbors=16, is_training=False
    ).preload()
    logger.info("All graphs successfully preloaded into RAM cache.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_crystal_graphs, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collate_crystal_graphs, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collate_crystal_graphs, num_workers=0
    )

    # Initialize AQ-GNN model
    model = AdaptiveQuadrupoleGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_gaussians=40,
        radius_cutoff=6.0,
        num_quad_channels=num_quad_channels,
        dropout=0.05,
        pooling=pooling
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"AQ-GNN initialized with {total_params:,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=3, min_lr=1e-5
    )

    best_val_mae = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    start_time = time.time()
    logger.info("--- Starting Training Loop ---")

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss_total = 0.0
        train_c_loss_total = 0.0
        train_s_loss_total = 0.0
        n_graphs = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_crystal_pred, v_sites_pred = model(batch)
            y_crystal_true = batch.y_scalar.squeeze(-1)

            # Loss calculation
            if loss_mode == "l1":
                loss_c = F.l1_loss(y_crystal_pred, y_crystal_true)
                loss_rel = torch.mean(torch.abs(y_crystal_pred - y_crystal_true) / (y_crystal_true + 15.0))
                loss_crystal = loss_c + 0.1 * loss_rel
                if hasattr(batch, "y_site") and batch.y_site is not None:
                    loss_site = F.l1_loss(v_sites_pred, batch.y_site)
                else:
                    loss_site = torch.tensor(0.0, device=device)
            else:
                loss_hub = F.smooth_l1_loss(y_crystal_pred, y_crystal_true, beta=1.0)
                loss_rel = torch.mean(torch.abs(y_crystal_pred - y_crystal_true) / (y_crystal_true + 15.0))
                loss_crystal = loss_hub + 2.0 * loss_rel
                if hasattr(batch, "y_site") and batch.y_site is not None:
                    loss_site = F.smooth_l1_loss(v_sites_pred, batch.y_site, beta=1.0)
                else:
                    loss_site = torch.tensor(0.0, device=device)

            total_loss = loss_crystal + site_loss_weight * loss_site
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            B = batch.num_graphs
            train_loss_total += total_loss.item() * B
            train_c_loss_total += loss_crystal.item() * B
            train_s_loss_total += loss_site.item() * B
            n_graphs += B

        train_loss_avg = train_loss_total / max(1, n_graphs)
        train_c_loss_avg = train_c_loss_total / max(1, n_graphs)

        # Validation evaluation: strictly on crystal targets
        model.eval()
        val_preds: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_pred, _ = model(batch)
                val_preds.extend(y_pred.cpu().tolist())

        val_preds_arr = np.array(val_preds, dtype=np.float32)
        val_mae = float(np.mean(np.abs(val_preds_arr - y_val)))
        scheduler.step(val_mae)

        ep_duration = time.time() - ep_start
        tau_val = float(torch.exp(model.log_tau).item())
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] | "
            f"Train Loss: {train_loss_avg:.4f} (Crystal: {train_c_loss_avg:.4f}) | "
            f"Val MAE: {val_mae:.4f} | tau: {tau_val:.2f} | lr: {current_lr:.1e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if tag:
                ckpt_name = f"AQ_GNN_{tag}.pt"
            elif seed == 42:
                ckpt_name = "AQ_GNN_best.pt"
            else:
                ckpt_name = f"AQ_GNN_seed{seed}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
                "config": {
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_quad_channels": num_quad_channels,
                    "pooling": pooling,
                    "loss_mode": loss_mode,
                    "seed": seed
                }
            }, ckpt_dir / ckpt_name)
            logger.info(f"  --> [NEW BEST] Saved checkpoint {ckpt_name} with Val MAE: {val_mae:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.1f}s. Best Val MAE: {best_val_mae:.4f}")

    # Load best checkpoint for test evaluation
    model.load_state_dict(best_weights)
    model.eval()

    test_preds: List[float] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y_pred, _ = model(batch)
            test_preds.extend(y_pred.cpu().tolist())

    test_preds_arr = np.array(test_preds, dtype=np.float32)

    # Compute official Task A metrics
    metrics = compute_task_a_scalar_metrics(y_test, test_preds_arr, train_mad=train_mad)
    if tag:
        model_name = f"A6_aq_gnn_{tag}"
        pred_filename = f"A6_aq_gnn_{tag}.csv"
        note_str = f"Adaptive Quadrupole GNN (tag={tag}, pooling={pooling})"
    elif seed == 42:
        model_name = "A6_aq_gnn"
        pred_filename = "A6_aq_gnn.csv"
        note_str = "Adaptive Quadrupole GNN (Physical Multipole + SmoothMax)"
    else:
        model_name = f"A6_aq_gnn_seed{seed}"
        pred_filename = f"A6_aq_gnn_seed{seed}.csv"
        note_str = f"Adaptive Quadrupole GNN (seed={seed})"

    metrics["model"] = model_name
    metrics["published_mae"] = None
    metrics["notes"] = note_str

    logger.info(f"=== OFFICIAL TEST SET EVALUATION ({model_name}) ===")
    logger.info(f"Test MAE: {metrics['mae']:.4f} V/A^2 (ALIGNN reference: 19.1211, CGCNN reference: 24.6695)")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f} V/A^2")
    logger.info(f"Test Median AE: {metrics['median_ae']:.4f} V/A^2")
    logger.info(f"Test R^2: {metrics['r2']:.4f}")
    logger.info(f"Test High-EFG F1: {metrics['high_efg_f1']:.4f}")

    # Save test predictions CSV
    pred_filename = "A6_aq_gnn.csv" if seed == 42 else f"A6_aq_gnn_seed{seed}.csv"
    pred_df = pd.DataFrame({
        "jid": test_jids,
        "true": y_test,
        "pred": test_preds_arr
    })
    pred_path = pred_dir / pred_filename
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved test predictions to {pred_path}")

    # Update consolidated scalar leaderboard table
    scalar_table_path = tables_dir / "scalar_baselines.csv"
    if scalar_table_path.exists():
        table_df = pd.read_csv(scalar_table_path)
        table_df = table_df[table_df["model"] != model_name]
        table_df = pd.concat([table_df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        table_df = pd.DataFrame([metrics])

    cols = ["model", "mae", "rmse", "median_ae", "r2", "nmae_mad", "high_efg_f1", "published_mae", "notes"]
    table_df = table_df[[c for c in cols if c in table_df.columns]]
    table_df.to_csv(scalar_table_path, index=False)
    logger.info(f"Updated scalar leaderboard table:\n{table_df.to_string()}")

    # Automated multi-seed ensemble if >= 2 seeds exist
    seed_files = sorted(list(pred_dir.glob("A6_aq_gnn*.csv")))
    # Exclude ensemble if already present
    seed_files = [f for f in seed_files if "ensemble" not in f.name]
    if len(seed_files) >= 2:
        logger.info(f"Computing ensemble across {len(seed_files)} seeds: {[f.name for f in seed_files]}")
        dfs = [pd.read_csv(f) for f in seed_files]
        ens_pred = np.mean([df["pred"].values for df in dfs], axis=0)
        ens_metrics = compute_task_a_scalar_metrics(y_test, ens_pred, train_mad=train_mad)
        ens_metrics["model"] = "A7_aq_gnn_ensemble"
        ens_metrics["published_mae"] = None
        ens_metrics["notes"] = f"AQ-GNN {len(seed_files)}-Seed Ensemble (Zero Leakage)"

        logger.info("=== ENSEMBLE TEST SET EVALUATION ===")
        logger.info(f"Ensemble Test MAE: {ens_metrics['mae']:.4f} V/A^2 (ALIGNN reference: 19.1211, CGCNN reference: 24.6695)")
        logger.info(f"Ensemble Test RMSE: {ens_metrics['rmse']:.4f} V/A^2")
        logger.info(f"Ensemble Test Median AE: {ens_metrics['median_ae']:.4f} V/A^2")
        logger.info(f"Ensemble Test R^2: {ens_metrics['r2']:.4f}")
        logger.info(f"Ensemble Test High-EFG F1: {ens_metrics['high_efg_f1']:.4f}")

        ens_pred_df = pd.DataFrame({"jid": test_jids, "true": y_test, "pred": ens_pred})
        ens_pred_df.to_csv(pred_dir / "A7_aq_gnn_ensemble.csv", index=False)

        table_df = pd.read_csv(scalar_table_path)
        table_df = table_df[table_df["model"] != "A7_aq_gnn_ensemble"]
        table_df = pd.concat([table_df, pd.DataFrame([ens_metrics])], ignore_index=True)
        table_df = table_df[[c for c in cols if c in table_df.columns]]
        table_df.to_csv(scalar_table_path, index=False)
        logger.info(f"Updated scalar leaderboard table with Ensemble:\n{table_df.to_string()}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AQ-GNN on JARVIS scalar benchmark")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_quad_channels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--site_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", type=str, default="smoothmax", choices=["smoothmax", "boltzmann"])
    parser.add_argument("--loss_mode", type=str, default="smooth_l1_rel", choices=["smooth_l1_rel", "l1"])
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    train_aq_gnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_quad_channels=args.num_quad_channels,
        lr=args.lr,
        site_loss_weight=args.site_weight,
        seed=args.seed,
        pooling=args.pooling,
        loss_mode=args.loss_mode,
        tag=args.tag
    )
