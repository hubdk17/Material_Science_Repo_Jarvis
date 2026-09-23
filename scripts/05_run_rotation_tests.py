"""Script to execute mandatory rotation invariance and covariance tests."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from pathlib import Path
import pandas as pd
import torch

from src.evaluation.rotation_tests import (
    run_rotation_covariance_test_tensor,
    run_rotation_invariance_test_scalar
)
from src.models.equivariant_gnn import EquivariantTensorGNN
from src.models.invariant_gnn import InvariantTensorGNN
from src.models.scalar_baselines import ALIGNNEquivalent, CGCNN
from src.training.logger import setup_logger


def main():
    logger = setup_logger("rotation_tests")
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Select 20 diverse sample structures
    sample_records = data[:20]
    logger.info(f"Loaded {len(sample_records)} crystals for numerical rotation testing.")

    results = []

    # 1. Scalar ALIGNN Invariance
    logger.info("Testing ALIGNN scalar rotational invariance...")
    m_alignn = ALIGNNEquivalent()
    res_alignn = run_rotation_invariance_test_scalar(m_alignn, sample_records, num_rotations=5)
    results.append({
        "model": "ALIGNN (Scalar)",
        "task": "Task A: Scalar Max-EFG",
        "expected_symmetry": "SO(3) Invariant",
        "tested_metric": "Max |f(RX) - f(X)|",
        "max_error": res_alignn["max_invariance_error"],
        "mean_error": res_alignn["mean_invariance_error"],
        "tolerance": res_alignn["tolerance"],
        "passed": res_alignn["passed"]
    })

    # 2. Scalar CGCNN Invariance
    logger.info("Testing CGCNN scalar rotational invariance...")
    m_cgcnn = CGCNN()
    res_cgcnn = run_rotation_invariance_test_scalar(m_cgcnn, sample_records, num_rotations=5)
    results.append({
        "model": "CGCNN (Scalar)",
        "task": "Task A: Scalar Max-EFG",
        "expected_symmetry": "SO(3) Invariant",
        "tested_metric": "Max |f(RX) - f(X)|",
        "max_error": res_cgcnn["max_invariance_error"],
        "mean_error": res_cgcnn["mean_invariance_error"],
        "tolerance": res_cgcnn["tolerance"],
        "passed": res_cgcnn["passed"]
    })

    # 3. Invariant GNN Tensor Covariance (B3/B4: Expected to FAIL covariance because Cartesian frame is unaligned!)
    logger.info("Testing Invariant GNN tensor rotational covariance (Control)...")
    m_inv = InvariantTensorGNN(mode="symmetric_traceless_5d")
    res_inv = run_rotation_covariance_test_tensor(m_inv, sample_records, num_rotations=5, tolerance=1e-4)
    results.append({
        "model": "Invariant GNN (B4)",
        "task": "Task B: Site Tensor",
        "expected_symmetry": "Non-covariant (Control)",
        "tested_metric": "Max ||V(RX) - R V(X) R^T||_F",
        "max_error": res_inv["max_covariance_error"],
        "mean_error": res_inv["mean_covariance_error"],
        "tolerance": res_inv["tolerance"],
        "passed": res_inv["passed"]
    })

    # 4. Equivariant GNN Tensor Covariance (B5: Expected to PASS covariance!)
    logger.info("Testing Equivariant GNN tensor rotational covariance (B5)...")
    m_eq = EquivariantTensorGNN()
    res_eq = run_rotation_covariance_test_tensor(m_eq, sample_records, num_rotations=5, tolerance=1e-4)
    results.append({
        "model": "Equivariant GNN (B5)",
        "task": "Task B: Site Tensor",
        "expected_symmetry": "SO(3) Equivariant",
        "tested_metric": "Max ||V(RX) - R V(X) R^T||_F",
        "max_error": res_eq["max_covariance_error"],
        "mean_error": res_eq["mean_covariance_error"],
        "tolerance": res_eq["tolerance"],
        "passed": res_eq["passed"]
    })

    df = pd.DataFrame(results)
    out_file = tables_dir / "rotation_test_results.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Rotation tests completed:\n{df.to_string()}")


if __name__ == "__main__":
    main()
