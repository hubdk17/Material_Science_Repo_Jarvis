"""Unit tests verifying site alignment and preventing library-induced reordering."""

import json
import numpy as np
import pytest


@pytest.fixture(scope="module")
def raw_sample():
    with open("data/raw/JARVIS-EFG4.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:50]


def test_site_alignment_identity(raw_sample):
    """Verifies that atomic site count, coordinate count, and tensor count match 1-to-1."""
    for record in raw_sample:
        jid = record["jid"]
        atoms = record["atoms"]
        elements = atoms["elements"]
        coords = atoms["coords"]
        raw_tensors = record["efg_raw_tensor"]
        diag_tensors = record["efg_diag_tensor"]

        n_atoms = len(elements)
        assert len(coords) == n_atoms, f"Coords length mismatch for {jid}"
        assert len(raw_tensors) == n_atoms, f"Raw tensor length mismatch for {jid}"
        assert len(diag_tensors) == n_atoms, f"Diag tensor length mismatch for {jid}"

        # Check each site has valid 3x3 matrix and 4-component diag
        for i in range(n_atoms):
            V = np.array(raw_tensors[i], dtype=float)
            D = np.array(diag_tensors[i], dtype=float)
            assert V.shape == (3, 3), f"Site {i} of {jid} has invalid shape {V.shape}"
            assert D.shape == (4,), f"Site {i} of {jid} has invalid diag shape {D.shape}"

            # Check trace is near zero
            assert abs(np.trace(V)) < 1e-2, f"Site {i} of {jid} has large trace {np.trace(V)}"
