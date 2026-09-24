"""Compute validation-optimal ensemble of Adaptive Quadrupole GNN (AQ-GNN) seeds.

Protocol:
- Weights are solved strictly on the official validation partition (1,186 crystals)
  via constrained SLSQP optimization: min_w ||y_val - V_val @ w||_1, s.t. sum(w)=1, w >= 0.
- Test set is evaluated strictly once using the validation-frozen weights.
- Zero test data leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

from scripts.train_adaptive_quadrupole_gnn import AQCrystalDataset, collate_crystal_graphs
from src.evaluation.metrics import compute_task_a_scalar_metrics
from src.models.adaptive_quadrupole_gnn import AdaptiveQuadrupoleGNN
from src.training.logger import setup_logger


def compute_optimal_ensemble(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    split_path: str = "splits/official_jarvis_scalar.json",
    results_dir: str = "results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    logger = setup_logger("optimal_ensemble")
    logger.info("Computing validation-optimal AQ-GNN ensemble...")

    res_path = Path(results_dir)
    ckpt_dir = res_path / "checkpoints" / "scalar"
    pred_dir = res_path / "predictions" / "scalar"
    tables_dir = res_path / "tables"

    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    recs = {r["jid"]: r for r in raw_data}

    val_jids = split["val"]["jids"]
    test_jids = split["test"]["jids"]

    y_train = np.array([split["train"]["targets"][j] for j in split["train"]["jids"]], dtype=np.float32)
    y_val = np.array([split["val"]["targets"][j] for j in val_jids], dtype=np.float32)
    y_test = np.array([split["test"]["targets"][j] for j in test_jids], dtype=np.float32)

    train_median = np.median(y_train)
    train_mad = float(np.median(np.abs(y_train - train_median)))

    # Discover trained checkpoints dynamically
    ckpt_files = [
        ("seed42", "AQ_GNN_best.pt", "A6_aq_gnn.csv"),
        ("seed43", "AQ_GNN_seed43.pt", "A6_aq_gnn_seed43.csv"),
        ("seed44", "AQ_GNN_seed44.pt", "A6_aq_gnn_seed44.csv"),
        ("seed45", "AQ_GNN_seed45.pt", "A6_aq_gnn_seed45.csv"),
        ("seed46", "AQ_GNN_seed46.pt", "A6_aq_gnn_seed46.csv"),
        ("seed47_boltz", "AQ_GNN_seed47_boltz.pt", "A6_aq_gnn_seed47_boltz.csv"),
    ]
    # Also discover any other AQ_GNN_*.pt files
    for p in ckpt_dir.glob("AQ_GNN_*.pt"):
        tag_cand = p.stem.replace("AQ_GNN_", "")
        pred_cand = f"A6_aq_gnn_{tag_cand}.csv" if tag_cand != "best" else "A6_aq_gnn.csv"
        if not any(t[1] == p.name for t in ckpt_files):
            ckpt_files.append((tag_cand, p.name, pred_cand))

    valid_ckpts = []
    for tag, ckpt_name, pred_name in ckpt_files:
        if (ckpt_dir / ckpt_name).exists() and (pred_dir / pred_name).exists():
            valid_ckpts.append((tag, ckpt_name, pred_name))

    logger.info(f"Discovered {len(valid_ckpts)} valid seed models: {[t[0] for t in valid_ckpts]}")

    # Preload validation dataset
    val_ds = AQCrystalDataset(val_jids, recs, split["val"]["targets"], radius_cutoff=6.0, max_neighbors=16, is_training=False).preload()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_crystal_graphs)

    # Evaluate validation predictions for each checkpoint
    val_preds_list = []
    test_preds_list = []

    for tag, ckpt_name, pred_name in valid_ckpts:
        logger.info(f"Evaluating {tag} on validation split...")
        ckpt = torch.load(ckpt_dir / ckpt_name, map_location=device)
        cfg = ckpt.get("config", {})
        h_dim = cfg.get("hidden_dim", 128)
        n_layers = cfg.get("num_layers", 4)
        n_quad = cfg.get("num_quad_channels", 4)
        pool = cfg.get("pooling", "smoothmax")
        model = AdaptiveQuadrupoleGNN(
            hidden_dim=h_dim,
            num_layers=n_layers,
            num_quad_channels=n_quad,
            pooling=pool
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        v_preds = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                yc, _ = model(batch)
                v_preds.extend(yc.cpu().tolist())

        val_preds_list.append(np.array(v_preds, dtype=np.float32))
        test_df = pd.read_csv(pred_dir / pred_name)
        test_preds_list.append(test_df["pred"].values)

    V_val = np.column_stack(val_preds_list)
    P_test = np.column_stack(test_preds_list)
    K = len(valid_ckpts)

    # 1. Simple Equal-Weight Ensemble
    simple_ens_pred = np.mean(P_test, axis=1)
    simple_metrics = compute_task_a_scalar_metrics(y_test, simple_ens_pred, train_mad=train_mad)
    simple_metrics["model"] = f"A7_aq_gnn_{K}seed_mean"
    simple_metrics["published_mae"] = None
    simple_metrics["notes"] = f"AQ-GNN {K}-Seed Equal-Weight Mean"

    # 2. Validation-Optimal Constrained SLSQP Weighted Ensemble
    def obj_fun(w):
        return np.mean(np.abs(y_val - (V_val @ w)))

    bounds = [(0.0, 1.0) for _ in range(K)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    init_w = np.ones(K) / K
    opt_res = minimize(obj_fun, init_w, bounds=bounds, constraints=cons, method="SLSQP")
    opt_weights = opt_res.x

    opt_val_mae = float(obj_fun(opt_weights))
    logger.info(f"Validation-Optimal Weights: {dict(zip([t[0] for t in valid_ckpts], np.round(opt_weights, 4)))}")
    logger.info(f"Optimized Validation MAE: {opt_val_mae:.4f} V/A^2")

    # Evaluate test set strictly with validation-optimal weights
    opt_test_pred = P_test @ opt_weights
    opt_metrics = compute_task_a_scalar_metrics(y_test, opt_test_pred, train_mad=train_mad)
    opt_metrics["model"] = "A7_aq_gnn_optimal_ensemble"
    opt_metrics["published_mae"] = None
    opt_metrics["notes"] = f"AQ-GNN {K}-Seed Validation-Optimal Ensemble (SLSQP)"

    logger.info("=== OFFICIAL TEST SET EVALUATION (OPTIMAL ENSEMBLE) ===")
    logger.info(f"Test MAE: {opt_metrics['mae']:.4f} V/A^2 (ALIGNN reference: 19.1211, CGCNN reference: 24.6695)")
    logger.info(f"Test RMSE: {opt_metrics['rmse']:.4f} V/A^2")
    logger.info(f"Test Median AE: {opt_metrics['median_ae']:.4f} V/A^2")
    logger.info(f"Test R^2: {opt_metrics['r2']:.4f}")
    logger.info(f"Test High-EFG F1: {opt_metrics['high_efg_f1']:.4f}")

    # Save test predictions CSV
    pd.DataFrame({
        "jid": test_jids,
        "true": y_test,
        "pred": opt_test_pred
    }).to_csv(pred_dir / "A7_aq_gnn_optimal_ensemble.csv", index=False)

    # Update consolidated scalar leaderboard table
    table_path = tables_dir / "scalar_baselines.csv"
    table_df = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    for m in [simple_metrics, opt_metrics]:
        table_df = table_df[table_df["model"] != m["model"]]
        table_df = pd.concat([table_df, pd.DataFrame([m])], ignore_index=True)

    cols = ["model", "mae", "rmse", "median_ae", "r2", "nmae_mad", "high_efg_f1", "published_mae", "notes"]
    table_df = table_df[[c for c in cols if c in table_df.columns]]
    table_df.to_csv(table_path, index=False)
    logger.info(f"Consolidated Scalar Table Updated:\n{table_df.to_string()}")

    return table_df


if __name__ == "__main__":
    compute_optimal_ensemble()
