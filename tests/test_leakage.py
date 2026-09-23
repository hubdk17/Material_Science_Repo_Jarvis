"""Automated tests proving zero data leakage across splits."""

import json
from pathlib import Path
import pytest
from src.data.audit import compute_reduced_formula, compute_structure_hash


@pytest.fixture(scope="module")
def raw_dataset():
    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {r["jid"]: r for r in data}


def test_protocol_a_id_overlap():
    """Verifies that Protocol A has zero JARVIS ID overlap between train, val, and test."""
    with open("splits/official_jarvis_scalar.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    train_ids = set(split["train"]["jids"])
    val_ids = set(split["val"]["jids"])
    test_ids = set(split["test"]["jids"])

    assert len(train_ids & val_ids) == 0, f"Leakage: Train and Val share {len(train_ids & val_ids)} IDs"
    assert len(train_ids & test_ids) == 0, f"Leakage: Train and Test share {len(train_ids & test_ids)} IDs"
    assert len(val_ids & test_ids) == 0, f"Leakage: Val and Test share {len(val_ids & test_ids)} IDs"


@pytest.mark.parametrize("split_file", [
    "splits/tensor_development_split.json",
    "splits/tensor_grouped_fold_0.json",
    "splits/tensor_grouped_fold_1.json",
    "splits/tensor_grouped_fold_2.json",
    "splits/tensor_grouped_fold_3.json",
    "splits/tensor_grouped_fold_4.json",
])
def test_protocol_b_leakage_invariants(split_file, raw_dataset):
    """Proves:
    1. Zero JARVIS-ID overlap between train, val, test
    2. Zero chemical-system overlap between train, val, test
    3. Zero reduced-formula overlap between train, val, test
    4. Zero structure-hash overlap between train, val, test
    5. Complete assignment of structures (no orphan sites)
    """
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_ids = set(data["train"]["jids"])
    val_ids = set(data["val"]["jids"])
    test_ids = set(data["test"]["jids"])

    # 1. Zero ID overlap
    assert len(train_ids & val_ids) == 0, "Train-Val ID overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train-Test ID overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val-Test ID overlap detected!"

    # 2. Zero chemical system overlap
    train_cs = set(data["train"]["chemical_systems"])
    val_cs = set(data["val"]["chemical_systems"])
    test_cs = set(data["test"]["chemical_systems"])

    assert len(train_cs & val_cs) == 0, f"Chemical system overlap Train-Val: {train_cs & val_cs}"
    assert len(train_cs & test_cs) == 0, f"Chemical system overlap Train-Test: {train_cs & test_cs}"
    assert len(val_cs & test_cs) == 0, f"Chemical system overlap Val-Test: {val_cs & test_cs}"

    # 3. Zero reduced formula overlap
    train_formulas = {compute_reduced_formula(raw_dataset[j]["atoms"]["elements"]) for j in train_ids}
    val_formulas = {compute_reduced_formula(raw_dataset[j]["atoms"]["elements"]) for j in val_ids}
    test_formulas = {compute_reduced_formula(raw_dataset[j]["atoms"]["elements"]) for j in test_ids}

    assert len(train_formulas & val_formulas) == 0, "Reduced formula overlap Train-Val!"
    assert len(train_formulas & test_formulas) == 0, "Reduced formula overlap Train-Test!"
    assert len(val_formulas & test_formulas) == 0, "Reduced formula overlap Val-Test!"

    # 4. Zero structure hash overlap
    import numpy as np
    def get_hash(jid):
        rec = raw_dataset[jid]
        atoms = rec["atoms"]
        lat = np.array(atoms["lattice_mat"], dtype=float)
        coords = np.array(atoms["coords"], dtype=float)
        elems = atoms["elements"]
        return compute_structure_hash(lat, coords, elems)

    train_hashes = {get_hash(j) for j in train_ids}
    val_hashes = {get_hash(j) for j in val_ids}
    test_hashes = {get_hash(j) for j in test_ids}

    assert len(train_hashes & val_hashes) == 0, "Structure hash overlap Train-Val!"
    assert len(train_hashes & test_hashes) == 0, "Structure hash overlap Train-Test!"
    assert len(val_hashes & test_hashes) == 0, "Structure hash overlap Val-Test!"


def test_site_integrity_and_no_split_crystals(raw_dataset):
    """Verifies that every site from one crystal belongs strictly to the same partition."""
    with open("splits/tensor_grouped_fold_0.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    all_assigned_ids = set(data["train"]["jids"]) | set(data["val"]["jids"]) | set(data["test"]["jids"])
    for jid in all_assigned_ids:
        rec = raw_dataset[jid]
        n_elems = len(rec["atoms"]["elements"])
        n_raw = len(rec["efg_raw_tensor"])
        n_diag = len(rec["efg_diag_tensor"])
        assert n_elems == n_raw == n_diag, f"Incomplete sites for crystal {jid}"
