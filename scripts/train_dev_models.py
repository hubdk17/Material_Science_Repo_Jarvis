"""Train development split models (B2, B4, B5) with proper checkpoints and manifests.

Strict pre-registration rules:
- Train on splits/tensor_development_split.json train set only.
- Validation set used strictly for early stopping / model selection.
- Save trained checkpoints, scalers, and full 6-component sitewise predictions.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.dataset import PeriodicCrystalDataset, collate_crystal_graphs
from src.features.descriptors import extract_site_local_descriptors
from src.features.tensor_transforms import (
    project_symmetric_traceless,
    matrix_to_cartesian_6d,
    matrix_to_cartesian_5d,
    cartesian_5d_to_matrix
)
from src.models.invariant_gnn import InvariantTensorGNN
from src.models.equivariant_gnn import EquivariantTensorGNN
from src.models.local_mlp import SiteLocalMLP
from src.training.losses import FrobeniusLoss
from src.training.trainer import fit_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("train_dev_models")


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    ckpt_dir = Path("results/checkpoints/dev_split")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path("results/predictions/tensor/tensor_development_split")
    pred_dir.mkdir(parents=True, exist_ok=True)

    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_by_jid = {d["jid"]: d for d in raw_data}

    with open("splits/tensor_development_split.json", "r", encoding="utf-8") as f:
        dev_split = json.load(f)

    train_jids = dev_split["train"]["jids"]
    val_jids = dev_split["val"]["jids"]
    test_jids = dev_split["test"]["jids"]

    # =========================================================================
    # 1. B2: Site Local Descriptors + Ridge Regression (Deterministic Baseline)
    # =========================================================================
    logger.info("Building B2 local descriptors...")
    def extract_split_descriptors(jids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for j in jids:
            r = raw_by_jid[j]
            lat = np.array(r["atoms"]["lattice_mat"], dtype=float)
            coords = np.array(r["atoms"]["coords"], dtype=float)
            elems = r["atoms"]["elements"]
            tensors = r["efg_raw_tensor"]
            for s in range(len(elems)):
                desc = extract_site_local_descriptors(lat, coords, elems, s)
                V_st = project_symmetric_traceless(tensors[s])
                v5 = matrix_to_cartesian_5d(V_st)
                X_list.append(desc)
                y_list.append(v5)
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    b2_ckpt_path = ckpt_dir / "B2_ridge.joblib"
    b2_scaler_path = ckpt_dir / "B2_scaler.joblib"

    if not b2_ckpt_path.exists():
        logger.info("Extracting B2 training descriptors...")
        X_train, y_train = extract_split_descriptors(train_jids)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        logger.info(f"Fitting Ridge regression on {len(X_train)} training sites...")
        ridge = Ridge(alpha=10.0)
        ridge.fit(X_train_scaled, y_train)

        joblib.dump(ridge, b2_ckpt_path)
        joblib.dump(scaler, b2_scaler_path)
        logger.info(f"Saved B2 Ridge model to {b2_ckpt_path}")
    else:
        logger.info(f"Loading existing B2 Ridge model from {b2_ckpt_path}")
        ridge = joblib.load(b2_ckpt_path)
        scaler = joblib.load(b2_scaler_path)

    # Predict B2 on test set
    logger.info("Predicting B2 on test set...")
    X_test, _ = extract_split_descriptors(test_jids)
    X_test_scaled = scaler.transform(X_test)
    pred_v5_b2 = ridge.predict(X_test_scaled)
    V_pred_b2 = cartesian_5d_to_matrix(pred_v5_b2)
    np.save(pred_dir / "B2_predictions_tensors.npy", V_pred_b2)

    # =========================================================================
    # 2. PyG Datasets & Loaders for B4 and B5
    # =========================================================================
    logger.info("Initializing Graph Datasets...")
    train_ds = PeriodicCrystalDataset(train_jids, raw_by_jid, target_mode="tensor_5d")
    val_ds = PeriodicCrystalDataset(val_jids, raw_by_jid, target_mode="tensor_5d")
    test_ds = PeriodicCrystalDataset(test_jids, raw_by_jid, target_mode="tensor_5d")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=collate_crystal_graphs)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_crystal_graphs)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_crystal_graphs)

    criterion = FrobeniusLoss()

    # =========================================================================
    # 3. B4: Invariant GNN (5D symmetric traceless head)
    # =========================================================================
    b4_ckpt_path = ckpt_dir / "B4_invariant.pt"
    if not b4_ckpt_path.exists():
        logger.info("Training B4 Invariant GNN...")
        b4 = InvariantTensorGNN(node_dim=64, hidden_dim=128, num_layers=4, mode="symmetric_traceless_5d").to(device)
        b4_opt = torch.optim.AdamW(b4.parameters(), lr=1e-3, weight_decay=1e-5)
        b4, b4_hist = fit_model(b4, train_loader, val_loader, b4_opt, criterion, device, epochs=12, patience=5, task="tensor", logger=logger)
        torch.save({
            "model_state_dict": b4.state_dict(),
            "config": {"node_dim": 64, "hidden_dim": 128, "num_layers": 4, "mode": "symmetric_traceless_5d"},
            "history": b4_hist
        }, b4_ckpt_path)
        logger.info(f"Saved B4 checkpoint to {b4_ckpt_path}")
    else:
        logger.info(f"Loading existing B4 checkpoint from {b4_ckpt_path}")
        ckpt = torch.load(b4_ckpt_path, map_location=device)
        b4 = InvariantTensorGNN(node_dim=64, hidden_dim=128, num_layers=4, mode="symmetric_traceless_5d").to(device)
        b4.load_state_dict(ckpt["model_state_dict"])

    b4.eval()
    pred_b4_list = []
    with torch.no_grad():
        for b in test_loader:
            _, V_mat = b4(b.to(device))
            pred_b4_list.append(V_mat.cpu().numpy())
    V_pred_b4 = np.concatenate(pred_b4_list, axis=0)
    np.save(pred_dir / "B4_predictions_tensors.npy", V_pred_b4)

    # =========================================================================
    # 4. B5: Equivariant GNN (e3nn l=2e irreps)
    # =========================================================================
    b5_ckpt_path = ckpt_dir / "B5_equivariant.pt"
    if not b5_ckpt_path.exists():
        logger.info("Training B5 Equivariant GNN...")
        b5 = EquivariantTensorGNN().to(device)
        b5_opt = torch.optim.AdamW(b5.parameters(), lr=1e-3, weight_decay=1e-5)
        b5, b5_hist = fit_model(b5, train_loader, val_loader, b5_opt, criterion, device, epochs=12, patience=5, task="tensor", logger=logger)
        torch.save({
            "model_state_dict": b5.state_dict(),
            "config": {"irreps_in": "64x0e", "irreps_hidden": "64x0e + 32x1o + 16x2e", "irreps_out": "1x2e"},
            "history": b5_hist
        }, b5_ckpt_path)
        logger.info(f"Saved B5 checkpoint to {b5_ckpt_path}")
    else:
        logger.info(f"Loading existing B5 checkpoint from {b5_ckpt_path}")
        ckpt = torch.load(b5_ckpt_path, map_location=device)
        b5 = EquivariantTensorGNN().to(device)
        b5.load_state_dict(ckpt["model_state_dict"])

    b5.eval()
    pred_b5_list = []
    with torch.no_grad():
        for b in test_loader:
            _, V_mat = b5(b.to(device))
            pred_b5_list.append(V_mat.cpu().numpy())
    V_pred_b5 = np.concatenate(pred_b5_list, axis=0)
    np.save(pred_dir / "B5_predictions_tensors.npy", V_pred_b5)

    logger.info("Development split training and prediction generation complete!")


if __name__ == "__main__":
    main()
