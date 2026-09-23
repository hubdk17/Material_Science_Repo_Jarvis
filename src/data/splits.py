"""Dataset splitting module implementing Protocol A and Protocol B."""

import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np


def compute_reduced_formula(elements: List[str]) -> str:
    """Computes canonical reduced formula."""
    counts = Counter(elements)
    g = math.gcd(*counts.values())
    parts = []
    for el in sorted(counts.keys()):
        c = counts[el] // g
        parts.append(f"{el}{c}" if c > 1 else el)
    return "".join(parts)


def compute_structure_hash(lattice_mat: np.ndarray, coords: np.ndarray, elements: List[str]) -> str:
    """Computes geometric structure hash."""
    rounded_lat = np.round(lattice_mat, 3)
    rounded_coords = np.round(coords % 1.0, 3)
    site_tuples = sorted(zip(elements, [tuple(c) for c in rounded_coords]))
    rep = (tuple(elements), tuple(map(tuple, rounded_lat)), tuple(site_tuples))
    return hashlib.sha256(repr(rep).encode("utf-8")).hexdigest()


def compute_partition_stats(jids: List[str], records_by_jid: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Computes summary statistics and elemental coverage for a partition."""
    num_structures = len(jids)
    num_sites = 0
    all_elements: Set[str] = set()
    vzz_magnitudes: List[float] = []
    frob_norms: List[float] = []

    for jid in jids:
        rec = records_by_jid[jid]
        elements = rec["atoms"]["elements"]
        num_sites += len(elements)
        all_elements.update(elements)

        diag = rec["efg_diag_tensor"]
        for d in diag:
            vzz_magnitudes.append(abs(float(d[2])))

        raw = rec["efg_raw_tensor"]
        for r in raw:
            frob_norms.append(float(np.linalg.norm(np.array(r, dtype=float), "fro")))

    return {
        "num_structures": num_structures,
        "num_sites": num_sites,
        "elemental_coverage": sorted(list(all_elements)),
        "element_count": len(all_elements),
        "target_vzz_stats": {
            "mean": float(np.mean(vzz_magnitudes)) if vzz_magnitudes else 0.0,
            "std": float(np.std(vzz_magnitudes)) if vzz_magnitudes else 0.0,
            "median": float(np.median(vzz_magnitudes)) if vzz_magnitudes else 0.0,
            "max": float(np.max(vzz_magnitudes)) if vzz_magnitudes else 0.0
        },
        "target_frobenius_stats": {
            "mean": float(np.mean(frob_norms)) if frob_norms else 0.0,
            "median": float(np.median(frob_norms)) if frob_norms else 0.0,
            "max": float(np.max(frob_norms)) if frob_norms else 0.0
        }
    }


def create_protocol_a_splits(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    bench_zip_path: str = "data/raw/dft_3d_max_efg.json.zip",
    output_path: str = "splits/official_jarvis_scalar.json"
) -> Dict[str, Any]:
    """Extracts and verifies the exact official JARVIS leaderboard split for Task A."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records_by_jid = {r["jid"]: r for r in data}

    with zipfile.ZipFile(bench_zip_path, "r") as z:
        bench_data = json.loads(z.read("dft_3d_max_efg.json").decode("utf-8"))

    train_bench = bench_data["train"]
    val_bench = bench_data["val"]
    test_bench = bench_data["test"]

    # Map IDs and report missing
    train_ids = [j for j in train_bench.keys() if j in records_by_jid]
    missing_train = [j for j in train_bench.keys() if j not in records_by_jid]
    val_ids = [j for j in val_bench.keys() if j in records_by_jid]
    missing_val = [j for j in val_bench.keys() if j not in records_by_jid]
    test_ids = [j for j in test_bench.keys() if j in records_by_jid]
    missing_test = [j for j in test_bench.keys() if j not in records_by_jid]

    # Compute SHA-256 of source dataset
    hasher = hashlib.sha256()
    with open(json_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    source_dataset_hash = hasher.hexdigest()

    def make_partition_a(j_list: List[str], b_targets: Dict[str, Any]) -> Dict[str, Any]:
        c_sys = sorted(list({"-".join(sorted(set(records_by_jid[j]["atoms"]["elements"]))) for j in j_list}))
        r_form = sorted(list({compute_reduced_formula(records_by_jid[j]["atoms"]["elements"]) for j in j_list}))
        return {
            "jids": j_list,
            "chemical_systems": c_sys,
            "reduced_formulas": r_form,
            "num_structures": len(j_list),
            "num_sites": sum(len(records_by_jid[j]["atoms"]["elements"]) for j in j_list),
            "targets": {j: float(b_targets[j]) for j in j_list},
            "stats": compute_partition_stats(j_list, records_by_jid)
        }

    split_obj: Dict[str, Any] = {
        "protocol": "Protocol A: Official JARVIS dft_3d_max_efg Benchmark",
        "description": "Predefined train/val/test split from official JARVIS-Leaderboard repository",
        "source_archive": bench_zip_path,
        "seed": None,
        "source_dataset_sha256": source_dataset_hash,
        "missing_ids_report": {
            "train_missing": missing_train,
            "val_missing": missing_val,
            "test_missing": missing_test
        },
        "train": make_partition_a(train_ids, train_bench),
        "val": make_partition_a(val_ids, val_bench),
        "test": make_partition_a(test_ids, test_bench)
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2)

    return split_obj


def create_protocol_b_splits(
    json_path: str = "data/raw/JARVIS-EFG4.json",
    splits_dir: str = "splits",
    seed: int = 42
) -> Dict[str, Any]:
    """Generates 5-fold stratified chemical-system grouped splits and a 70:15:15 development split."""
    out_dir = Path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compute SHA-256 of source dataset
    hasher = hashlib.sha256()
    with open(json_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    source_dataset_hash = hasher.hexdigest()

    records_by_jid = {r["jid"]: r for r in data}

    # Group structures by chemical system
    chem_sys_to_jids: Dict[str, List[str]] = defaultdict(list)
    chem_sys_meta: Dict[str, Dict[str, Any]] = {}

    for r in data:
        jid = r["jid"]
        elems = r["atoms"]["elements"]
        cs = "-".join(sorted(list(set(elems))))
        chem_sys_to_jids[cs].append(jid)

    # For each chemical system, compute stratification metrics
    for cs, j_list in chem_sys_to_jids.items():
        n_structs = len(j_list)
        n_sites = sum(len(records_by_jid[j]["atoms"]["elements"]) for j in j_list)
        max_vzz = max(
            abs(float(d[2]))
            for j in j_list
            for d in records_by_jid[j]["efg_diag_tensor"]
        )
        is_high_efg = 1 if max_vzz > 50.0 else 0
        chem_sys_meta[cs] = {
            "n_structs": n_structs,
            "n_sites": n_sites,
            "max_vzz": max_vzz,
            "is_high_efg": is_high_efg
        }

    # Sort chemical systems deterministically by size, then max_vzz, then name
    rng = np.random.default_rng(seed)
    all_chem_systems = sorted(
        chem_sys_to_jids.keys(),
        key=lambda c: (chem_sys_meta[c]["is_high_efg"], chem_sys_meta[c]["n_structs"], chem_sys_meta[c]["max_vzz"], c),
        reverse=True
    )

    # 1. Generate 5 balanced outer folds using greedy serpentine bin packing
    num_folds = 5
    fold_chem_systems: List[List[str]] = [[] for _ in range(num_folds)]
    fold_struct_counts = [0] * num_folds
    fold_high_efg_counts = [0] * num_folds

    for cs in all_chem_systems:
        meta = chem_sys_meta[cs]
        # Assign to fold with smallest structure count (tie break by high_efg count)
        best_fold = min(
            range(num_folds),
            key=lambda k: (fold_struct_counts[k], fold_high_efg_counts[k])
        )
        fold_chem_systems[best_fold].append(cs)
        fold_struct_counts[best_fold] += meta["n_structs"]
        fold_high_efg_counts[best_fold] += meta["is_high_efg"]

    def make_partition_b(j_list: List[str], cs_list: List[str]) -> Dict[str, Any]:
        r_form = sorted(list({compute_reduced_formula(records_by_jid[j]["atoms"]["elements"]) for j in j_list}))
        return {
            "chemical_systems": sorted(cs_list),
            "reduced_formulas": r_form,
            "jids": j_list,
            "num_structures": len(j_list),
            "num_sites": sum(len(records_by_jid[j]["atoms"]["elements"]) for j in j_list),
            "stats": compute_partition_stats(j_list, records_by_jid)
        }

    fold_files = {}
    for fold_idx in range(num_folds):
        outer_test_cs = fold_chem_systems[fold_idx]
        remaining_cs = [cs for k in range(num_folds) if k != fold_idx for cs in fold_chem_systems[k]]

        # Shuffle remaining chemical systems deterministically for 85:15 inner split
        perm = rng.permutation(len(remaining_cs))
        shuffled_remaining = [remaining_cs[i] for i in perm]

        total_rem_structs = sum(chem_sys_meta[c]["n_structs"] for c in shuffled_remaining)
        val_target_structs = int(total_rem_structs * 0.15)

        inner_val_cs = []
        inner_val_structs = 0
        inner_train_cs = []

        for cs in shuffled_remaining:
            n_s = chem_sys_meta[cs]["n_structs"]
            if inner_val_structs + n_s <= val_target_structs:
                inner_val_cs.append(cs)
                inner_val_structs += n_s
            else:
                inner_train_cs.append(cs)

        # Gather JIDs
        train_jids = [j for cs in inner_train_cs for j in chem_sys_to_jids[cs]]
        val_jids = [j for cs in inner_val_cs for j in chem_sys_to_jids[cs]]
        test_jids = [j for cs in outer_test_cs for j in chem_sys_to_jids[cs]]

        fold_data = {
            "protocol": f"Protocol B: Outer Fold {fold_idx}",
            "fold_index": fold_idx,
            "seed": seed,
            "source_dataset_sha256": source_dataset_hash,
            "train": make_partition_b(train_jids, inner_train_cs),
            "val": make_partition_b(val_jids, inner_val_cs),
            "test": make_partition_b(test_jids, outer_test_cs)
        }

        fold_path = out_dir / f"tensor_grouped_fold_{fold_idx}.json"
        with open(fold_path, "w", encoding="utf-8") as f:
            json.dump(fold_data, f, indent=2)
        fold_files[f"fold_{fold_idx}"] = str(fold_path)

    # 3. Create frozen 70:15:15 development split
    dev_rng = np.random.default_rng(seed + 100)
    perm_dev = dev_rng.permutation(len(all_chem_systems))
    shuffled_dev = [all_chem_systems[i] for i in perm_dev]

    total_structs = len(data)
    test_target = int(total_structs * 0.15)
    val_target = int(total_structs * 0.15)

    dev_test_cs, dev_val_cs, dev_train_cs = [], [], []
    c_test, c_val = 0, 0

    for cs in shuffled_dev:
        ns = chem_sys_meta[cs]["n_structs"]
        if c_test + ns <= test_target:
            dev_test_cs.append(cs)
            c_test += ns
        elif c_val + ns <= val_target:
            dev_val_cs.append(cs)
            c_val += ns
        else:
            dev_train_cs.append(cs)

    dev_train_jids = [j for cs in dev_train_cs for j in chem_sys_to_jids[cs]]
    dev_val_jids = [j for cs in dev_val_cs for j in chem_sys_to_jids[cs]]
    dev_test_jids = [j for cs in dev_test_cs for j in chem_sys_to_jids[cs]]

    dev_data = {
        "protocol": "Protocol B: 70:15:15 Grouped Development Split",
        "seed": seed + 100,
        "source_dataset_sha256": source_dataset_hash,
        "train": make_partition_b(dev_train_jids, dev_train_cs),
        "val": make_partition_b(dev_val_jids, dev_val_cs),
        "test": make_partition_b(dev_test_jids, dev_test_cs)
    }

    dev_path = out_dir / "tensor_development_split.json"
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_data, f, indent=2)
    fold_files["development"] = str(dev_path)

    generate_split_manifest(splits_dir=splits_dir)

    return fold_files


def generate_split_manifest(splits_dir: str = "splits") -> Dict[str, Any]:
    """Generates a cryptographic manifest recording SHA-256 hashes and metrics for all split files."""
    s_dir = Path(splits_dir)
    manifest = {}

    for p in sorted(s_dir.glob("*.json")):
        if p.name == "split_manifest.json":
            continue
        with open(p, "rb") as f:
            content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest[p.name] = {
            "file_path": str(p),
            "sha256": sha256,
            "size_bytes": len(content),
            "protocol": data.get("protocol", "Unknown"),
            "seed": data.get("seed", None),
            "source_dataset_sha256": data.get("source_dataset_sha256", None),
            "train_structures": data["train"]["num_structures"],
            "val_structures": data["val"]["num_structures"],
            "test_structures": data["test"]["num_structures"],
            "train_sites": data["train"]["num_sites"],
            "val_sites": data["val"]["num_sites"],
            "test_sites": data["test"]["num_sites"],
        }

    manifest_path = s_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    p_a = create_protocol_a_splits()
    print("Created Protocol A split:", len(p_a["train"]["jids"]), len(p_a["val"]["jids"]), len(p_a["test"]["jids"]))
    p_b = create_protocol_b_splits()
    print("Created Protocol B splits:", list(p_b.keys()))
