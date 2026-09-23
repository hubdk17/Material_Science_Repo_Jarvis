"""Script to execute Task A scalar max-EFG compatibility benchmark."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PeriodicCrystalDataset, collate_crystal_graphs
from src.evaluation.metrics import compute_task_a_scalar_metrics
from src.features.descriptors import (
    extract_crystal_structural_features,
    extract_magpie_composition_features
)
from src.models.scalar_baselines import (
    ALIGNNEquivalent,
    CGCNN,
    MedianBaseline,
    TabularBaselines
)
from src.training.logger import setup_logger
from src.training.seed import set_seed
from src.training.trainer import fit_model


def run_scalar_benchmark(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    split_path: str = "splits/official_jarvis_scalar.json",
    results_dir: str = "results",
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> pd.DataFrame:
    """Executes all Task A baselines (A0-A5) on the official JARVIS benchmark split."""
    set_seed(seed)
    logger = setup_logger("scalar_benchmark")
    logger.info(f"Starting Task A scalar benchmark on device: {device}")

    res_path = Path(results_dir)
    pred_dir = res_path / "predictions" / "scalar"
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

    y_train = np.array([split["train"]["targets"][j] for j in train_jids], dtype=np.float32)
    y_val = np.array([split["val"]["targets"][j] for j in val_jids], dtype=np.float32)
    y_test = np.array([split["test"]["targets"][j] for j in test_jids], dtype=np.float32)

    # Compute training MAD for normalized MAE
    train_median = np.median(y_train)
    train_mad = float(np.median(np.abs(y_train - train_median)))
    logger.info(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")
    logger.info(f"Train median: {train_median:.4f}, Train MAD: {train_mad:.4f}")

    benchmark_results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # Baseline A0: Median Predictor
    # -------------------------------------------------------------
    logger.info("Running Baseline A0: Training-set Median...")
    a0 = MedianBaseline().fit(y_train)
    pred_a0 = a0.predict(len(y_test))
    metrics_a0 = compute_task_a_scalar_metrics(y_test, pred_a0, train_mad=train_mad)
    metrics_a0["model"] = "A0_median"
    metrics_a0["published_mae"] = None
    metrics_a0["notes"] = "Dummy baseline"
    benchmark_results.append(metrics_a0)

    pd.DataFrame({
        "jid": test_jids,
        "true": y_test,
        "pred": pred_a0
    }).to_csv(pred_dir / "A0_median.csv", index=False)

    # -------------------------------------------------------------
    # Feature extraction for Tabular baselines A1, A2, A3
    # -------------------------------------------------------------
    logger.info("Extracting composition and structural descriptors...")
    X_comp_train = np.array([extract_magpie_composition_features(records_by_jid[j]["atoms"]["elements"]) for j in train_jids])
    X_comp_val = np.array([extract_magpie_composition_features(records_by_jid[j]["atoms"]["elements"]) for j in val_jids])
    X_comp_test = np.array([extract_magpie_composition_features(records_by_jid[j]["atoms"]["elements"]) for j in test_jids])

    X_struct_train = np.array([extract_crystal_structural_features(records_by_jid[j]) for j in train_jids])
    X_struct_val = np.array([extract_crystal_structural_features(records_by_jid[j]) for j in val_jids])
    X_struct_test = np.array([extract_crystal_structural_features(records_by_jid[j]) for j in test_jids])

    # -------------------------------------------------------------
    # Baseline A1: Composition Magpie + Random Forest
    # -------------------------------------------------------------
    logger.info("Running Baseline A1: Magpie + Random Forest...")
    a1 = TabularBaselines("rf", n_estimators=100, max_depth=15, random_state=seed)
    a1.fit(X_comp_train, y_train)
    pred_a1 = a1.predict(X_comp_test)
    metrics_a1 = compute_task_a_scalar_metrics(y_test, pred_a1, train_mad=train_mad)
    metrics_a1["model"] = "A1_magpie_rf"
    metrics_a1["published_mae"] = None
    metrics_a1["notes"] = "Composition only"
    benchmark_results.append(metrics_a1)

    pd.DataFrame({"jid": test_jids, "true": y_test, "pred": pred_a1}).to_csv(pred_dir / "A1_magpie_rf.csv", index=False)

    # -------------------------------------------------------------
    # Baseline A2: Composition Magpie + XGBoost
    # -------------------------------------------------------------
    logger.info("Running Baseline A2: Magpie + XGBoost...")
    a2 = TabularBaselines("xgb", n_estimators=200, learning_rate=0.05, max_depth=6, random_state=seed)
    a2.fit(X_comp_train, y_train)
    pred_a2 = a2.predict(X_comp_test)
    metrics_a2 = compute_task_a_scalar_metrics(y_test, pred_a2, train_mad=train_mad)
    metrics_a2["model"] = "A2_magpie_xgb"
    metrics_a2["published_mae"] = 19.4382  # Matminer reference on leaderboard
    metrics_a2["notes"] = "Composition only (Matminer proxy)"
    benchmark_results.append(metrics_a2)

    pd.DataFrame({"jid": test_jids, "true": y_test, "pred": pred_a2}).to_csv(pred_dir / "A2_magpie_xgb.csv", index=False)

    # -------------------------------------------------------------
    # Baseline A3: Structural Descriptors + HistGradientBoosting
    # -------------------------------------------------------------
    logger.info("Running Baseline A3: Structural + GBDT...")
    a3 = TabularBaselines("gbdt", n_estimators=200, learning_rate=0.05, max_depth=6, random_state=seed)
    a3.fit(X_struct_train, y_train)
    pred_a3 = a3.predict(X_struct_test)
    metrics_a3 = compute_task_a_scalar_metrics(y_test, pred_a3, train_mad=train_mad)
    metrics_a3["model"] = "A3_structural_gbdt"
    metrics_a3["published_mae"] = None
    metrics_a3["notes"] = "Composition + Cell Volume/Density/Lattice"
    benchmark_results.append(metrics_a3)

    pd.DataFrame({"jid": test_jids, "true": y_test, "pred": pred_a3}).to_csv(pred_dir / "A3_structural_gbdt.csv", index=False)

    # -------------------------------------------------------------
    # Deep Learning Datasets & Loaders for A4 and A5
    # -------------------------------------------------------------
    logger.info("Preparing crystal graph dataloaders for neural networks...")
    scalar_target_dict = {
        **{j: float(split["train"]["targets"][j]) for j in train_jids},
        **{j: float(split["val"]["targets"][j]) for j in val_jids},
        **{j: float(split["test"]["targets"][j]) for j in test_jids}
    }

    train_ds = PeriodicCrystalDataset(train_jids, records_by_jid, target_mode="scalar", scalar_targets=scalar_target_dict)
    val_ds = PeriodicCrystalDataset(val_jids, records_by_jid, target_mode="scalar", scalar_targets=scalar_target_dict)
    test_ds = PeriodicCrystalDataset(test_jids, records_by_jid, target_mode="scalar", scalar_targets=scalar_target_dict)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_crystal_graphs)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_crystal_graphs)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_crystal_graphs)

    # -------------------------------------------------------------
    # Baseline A4: CGCNN
    # -------------------------------------------------------------
    logger.info("Training Baseline A4: CGCNN...")
    model_a4 = CGCNN(atom_fea_len=64, n_conv=3, h_fea_len=128).to(device)
    optimizer_a4 = torch.optim.AdamW(model_a4.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = torch.nn.L1Loss()

    model_a4, _ = fit_model(
        model=model_a4,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_a4,
        criterion=criterion,
        device=device,
        epochs=30,
        patience=8,
        task="scalar",
        logger=logger
    )

    # Predict on test
    model_a4.eval()
    preds_a4 = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_a4.extend(model_a4(batch).cpu().tolist())
    pred_a4 = np.array(preds_a4, dtype=np.float32)

    metrics_a4 = compute_task_a_scalar_metrics(y_test, pred_a4, train_mad=train_mad)
    metrics_a4["model"] = "A4_cgcnn"
    metrics_a4["published_mae"] = 24.6695
    metrics_a4["notes"] = "Crystal Graph Convolutional Neural Network"
    benchmark_results.append(metrics_a4)

    pd.DataFrame({"jid": test_jids, "true": y_test, "pred": pred_a4}).to_csv(pred_dir / "A4_cgcnn.csv", index=False)

    # -------------------------------------------------------------
    # Baseline A5: ALIGNN Equivalent
    # -------------------------------------------------------------
    logger.info("Training Baseline A5: ALIGNN Equivalent...")
    model_a5 = ALIGNNEquivalent(atom_dim=64, edge_dim=64, num_layers=4).to(device)
    optimizer_a5 = torch.optim.AdamW(model_a5.parameters(), lr=1e-3, weight_decay=1e-5)

    model_a5, _ = fit_model(
        model=model_a5,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer_a5,
        criterion=criterion,
        device=device,
        epochs=30,
        patience=8,
        task="scalar",
        logger=logger
    )

    model_a5.eval()
    preds_a5 = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds_a5.extend(model_a5(batch).cpu().tolist())
    pred_a5 = np.array(preds_a5, dtype=np.float32)

    metrics_a5 = compute_task_a_scalar_metrics(y_test, pred_a5, train_mad=train_mad)
    metrics_a5["model"] = "A5_alignn_eq"
    metrics_a5["published_mae"] = 19.1211
    metrics_a5["notes"] = "Line Graph Atomistic Neural Network"
    benchmark_results.append(metrics_a5)

    pd.DataFrame({"jid": test_jids, "true": y_test, "pred": pred_a5}).to_csv(pred_dir / "A5_alignn_eq.csv", index=False)

    # -------------------------------------------------------------
    # Save consolidated results table
    # -------------------------------------------------------------
    res_df = pd.DataFrame(benchmark_results)
    # Order columns
    cols = ["model", "mae", "rmse", "median_ae", "r2", "nmae_mad", "high_efg_f1", "published_mae", "notes"]
    res_df = res_df[[c for c in cols if c in res_df.columns]]
    res_df.to_csv(tables_dir / "scalar_baselines.csv", index=False)
    logger.info(f"Task A scalar benchmark completed:\n{res_df.to_string()}")

    return res_df


if __name__ == "__main__":
    run_scalar_benchmark()
