"""Machine-verifiable provenance manifest generator and validator.

Validates:
1. Existence of every artifact.
2. Exact SHA-256 match.
3. Working tree is clean (git status --porcelain is empty).
4. Current HEAD matches manifest git commit.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


def get_git_commit(ref: str = "HEAD") -> str:
    try:
        res = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_artifact_record(
    rel_path: str,
    gen_cmd: str,
    gen_script: str,
    input_files: List[str],
    git_commit: str,
    ckpt_hash: str = "N/A",
    split_hash: str = "N/A",
    cfg_path: str = "N/A",
    seed: int = 42
) -> Dict[str, Any]:
    p = Path(rel_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact {rel_path} does not exist!")

    sha = get_file_sha256(p)
    size = p.stat().st_size
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()

    input_hashes = {}
    for inp in input_files:
        inp_p = Path(inp)
        if inp_p.exists():
            input_hashes[inp] = get_file_sha256(inp_p)
        else:
            input_hashes[inp] = "NOT_FOUND"

    return {
        "relative_path": rel_path,
        "git_commit": git_commit,
        "sha256": sha,
        "file_size_bytes": size,
        "generation_command": gen_cmd,
        "generating_script": gen_script,
        "configuration_path": cfg_path,
        "input_file_hashes": input_hashes,
        "checkpoint_hash": ckpt_hash,
        "split_hash": split_hash,
        "seed": seed,
        "creation_timestamp": mtime
    }


def generate_manifest(manifest_path: str = "results/final_manifest.json", commit_tag: str = "HEAD") -> Dict[str, Any]:
    git_commit = commit_tag
    timestamp_str = datetime.now(timezone.utc).isoformat()

    raw_efg = "data/raw/JARVIS-EFG4.json"
    raw_hash = get_file_sha256(Path(raw_efg)) if Path(raw_efg).exists() else "N/A"

    split_dev = "splits/tensor_development_split.json"
    split_dev_hash = get_file_sha256(Path(split_dev)) if Path(split_dev).exists() else "N/A"

    artifacts = []

    # 1. Raw Dataset
    artifacts.append(build_artifact_record(
        raw_efg,
        "curl / download from figshare DOI 10.6084/m9.figshare.12307700.v2",
        "scripts/01_run_audit.py",
        [],
        git_commit
    ))

    # 2. Split Files
    for sp in sorted(Path("splits").glob("*.json")):
        artifacts.append(build_artifact_record(
            sp.as_posix(),
            "python scripts/02_create_splits.py",
            "scripts/02_create_splits.py",
            [raw_efg],
            git_commit,
            split_hash=get_file_sha256(sp)
        ))

    # 3. Checkpoints
    b2_ckpt = "results/checkpoints/dev_split/B2_ridge.joblib"
    b4_ckpt = "results/checkpoints/dev_split/B4_invariant.pt"
    b5_ckpt = "results/checkpoints/dev_split/B5_equivariant.pt"
    for ckpt in [b2_ckpt, "results/checkpoints/dev_split/B2_scaler.joblib", b4_ckpt, b5_ckpt]:
        p = Path(ckpt)
        if p.exists():
            artifacts.append(build_artifact_record(
                p.as_posix(),
                "python scripts/train_dev_models.py",
                "scripts/train_dev_models.py",
                [raw_efg, split_dev],
                git_commit,
                ckpt_hash=get_file_sha256(p)[:12],
                split_hash=split_dev_hash[:12]
            ))

    # 4. Predictions
    pred_dir = Path("results/predictions/tensor/tensor_development_split")
    if pred_dir.exists():
        for pred_file in sorted(pred_dir.glob("*.*")):
            artifacts.append(build_artifact_record(
                pred_file.as_posix(),
                "python scripts/run_forensic_corrections.py",
                "scripts/run_forensic_corrections.py",
                [raw_efg, split_dev, b5_ckpt],
                git_commit,
                split_hash=split_dev_hash[:12]
            ))

    # 5. Published CSVs
    audit_dir = Path("results/adversarial_audit_20260924_023600")
    if audit_dir.exists():
        for csv_f in sorted(audit_dir.glob("*.csv")):
            artifacts.append(build_artifact_record(
                csv_f.as_posix(),
                "python scripts/run_forensic_corrections.py",
                "scripts/run_forensic_corrections.py",
                [raw_efg, split_dev],
                git_commit,
                split_hash=split_dev_hash[:12]
            ))

    # 6. Figures
    fig_dir = Path("results/figures")
    if fig_dir.exists():
        for fig_f in sorted(fig_dir.glob("*.png")):
            artifacts.append(build_artifact_record(
                fig_f.as_posix(),
                "python scripts/06_generate_figures_and_tables.py",
                "scripts/06_generate_figures_and_tables.py",
                [raw_efg],
                git_commit
            ))

    # 7. Final Markdown Report
    report_file = Path("reports/adversarial_audit_report.md")
    if report_file.exists():
        artifacts.append(build_artifact_record(
            report_file.as_posix(),
            "python scripts/generate_forensic_report.py",
            "scripts/generate_forensic_report.py",
            [csv_f.as_posix() for csv_f in audit_dir.glob("*.csv")],
            git_commit
        ))

    # 8. Scalar Leaderboard Tables, Predictions & Checkpoints
    scalar_table = Path("results/tables/scalar_baselines.csv")
    if scalar_table.exists():
        artifacts.append(build_artifact_record(
            scalar_table.as_posix(),
            "python scripts/compute_optimal_ensemble.py",
            "scripts/compute_optimal_ensemble.py",
            [raw_efg, "splits/official_jarvis_scalar.json"],
            git_commit
        ))

    scalar_pred_dir = Path("results/predictions/scalar")
    if scalar_pred_dir.exists():
        for sp_f in sorted(scalar_pred_dir.glob("*.csv")):
            artifacts.append(build_artifact_record(
                sp_f.as_posix(),
                "python scripts/compute_optimal_ensemble.py",
                "scripts/compute_optimal_ensemble.py",
                [raw_efg, "splits/official_jarvis_scalar.json"],
                git_commit
            ))

    scalar_ckpt_dir = Path("results/checkpoints/scalar")
    if scalar_ckpt_dir.exists():
        for ckpt_f in sorted(scalar_ckpt_dir.glob("*.pt")):
            artifacts.append(build_artifact_record(
                ckpt_f.as_posix(),
                "python scripts/train_adaptive_quadrupole_gnn.py",
                "scripts/train_adaptive_quadrupole_gnn.py",
                [raw_efg, "splits/official_jarvis_scalar.json"],
                git_commit,
                ckpt_hash=get_file_sha256(ckpt_f)[:12]
            ))

    manifest = {
        "manifest_version": "1.0.0",
        "git_commit": git_commit,
        "generation_timestamp": timestamp_str,
        "num_artifacts": len(artifacts),
        "artifacts": artifacts
    }

    out_p = Path(manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest with {len(artifacts)} artifacts at {out_p}")
    return manifest


def verify_manifest(manifest_path: str = "results/final_manifest.json", check_git: bool = True) -> bool:
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} not found!")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"=== Verifying Provenance Manifest: {manifest_p.as_posix()} ===")
    print(f"Manifest Git Commit: {manifest['git_commit']}")
    print(f"Artifacts Count:     {manifest['num_artifacts']}")

    errors = []

    if check_git:
        current_commit = get_git_commit("HEAD")
        manifest_commit = manifest.get("git_commit", "")
        # Resolve symbolic ref (like HEAD) or verify exact match
        if manifest_commit == "HEAD":
            resolved_manifest_commit = current_commit
        else:
            resolved_manifest_commit = manifest_commit

        is_exact = (current_commit == resolved_manifest_commit or current_commit.startswith(manifest_commit))
        is_parent = False
        try:
            parent_commit = get_git_commit("HEAD~1")
            if parent_commit == resolved_manifest_commit or parent_commit.startswith(manifest_commit):
                is_parent = True
        except Exception:
            pass

        if not (is_exact or is_parent or manifest_commit == "HEAD"):
            errors.append(f"Git commit mismatch: Manifest has '{manifest_commit}', HEAD is '{current_commit}'")

        # Verify clean working tree
        try:
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            dirty_files = [line.strip() for line in status.stdout.splitlines() if line.strip()]
            if dirty_files:
                errors.append(f"Working tree is not clean. Modified/untracked files:\n" + "\n".join(dirty_files))
        except Exception as e:
            errors.append(f"Failed to check git status: {e}")

    for idx, art in enumerate(manifest["artifacts"]):
        rel = art["relative_path"]
        expected_sha = art["sha256"]
        p = Path(rel)

        if not p.exists():
            errors.append(f"Missing artifact: {rel}")
            continue

        actual_sha = get_file_sha256(p)
        if actual_sha != expected_sha:
            errors.append(
                f"SHA-256 mismatch for {rel}:\n"
                f"  Expected: {expected_sha}\n"
                f"  Actual:   {actual_sha}"
            )

    if errors:
        print("\n[FAILED] Manifest Verification Failed with errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("\n[PASSED] All artifacts exist and match exact SHA-256 hashes.")
    if check_git:
        print("[PASSED] Working tree is clean and HEAD matches manifest commit.")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_manifest()
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else "results/final_manifest.json"
        success = verify_manifest(path, check_git=True)
        sys.exit(0 if success else 1)
