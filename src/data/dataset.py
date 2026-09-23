"""PyTorch dataset and graph caching for periodic crystals."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.features.graph import build_periodic_crystal_graph
from src.features.tensor_transforms import (
    matrix_to_cartesian_5d,
    matrix_to_cartesian_6d,
    matrix_to_irrep_2e,
    project_symmetric_traceless
)


class PeriodicCrystalDataset(Dataset):
    """PyTorch Dataset for crystal graphs with site tensors and scalar properties."""

    def __init__(
        self,
        jids: List[str],
        records_by_jid: Dict[str, Dict[str, Any]],
        radius_cutoff: float = 5.0,
        max_neighbors: int = 16,
        target_mode: str = "tensor_5d",  # "tensor_5d", "tensor_6d", "tensor_2e", "scalar_max"
        scalar_targets: Optional[Dict[str, float]] = None,
        transform: Optional[Any] = None
    ):
        self.jids = jids
        self.records_by_jid = records_by_jid
        self.radius_cutoff = radius_cutoff
        self.max_neighbors = max_neighbors
        self.target_mode = target_mode
        self.scalar_targets = scalar_targets or {}
        self.transform = transform
        self._graph_cache: Dict[str, Data] = {}

    def __len__(self) -> int:
        return len(self.jids)

    def __getitem__(self, idx: int) -> Data:
        jid = self.jids[idx]
        if jid in self._graph_cache:
            data = self._graph_cache[jid].clone()
        else:
            rec = self.records_by_jid[jid]
            lat = np.array(rec["atoms"]["lattice_mat"], dtype=float)
            frac_coords = np.array(rec["atoms"]["coords"], dtype=float)
            elements = rec["atoms"]["elements"]

            target_tensor = None
            target_scalar = None

            if "tensor" in self.target_mode:
                raw_tensors = rec["efg_raw_tensor"]
                if self.target_mode == "tensor_5d":
                    t_list = [matrix_to_cartesian_5d(project_symmetric_traceless(t)) for t in raw_tensors]
                elif self.target_mode == "tensor_6d":
                    t_list = [matrix_to_cartesian_6d(project_symmetric_traceless(t)) for t in raw_tensors]
                elif self.target_mode == "tensor_2e":
                    t_list = [matrix_to_irrep_2e(project_symmetric_traceless(t)) for t in raw_tensors]
                else:
                    raise ValueError(f"Unknown target_mode: {self.target_mode}")
                target_tensor = np.array(t_list, dtype=np.float32)

            if jid in self.scalar_targets:
                target_scalar = float(self.scalar_targets[jid])

            data = build_periodic_crystal_graph(
                lattice_mat=lat,
                fractional_coords=frac_coords,
                elements=elements,
                radius_cutoff=self.radius_cutoff,
                max_neighbors=self.max_neighbors,
                target_tensor_st=target_tensor,
                target_scalar=target_scalar
            )
            data.jid = jid
            self._graph_cache[jid] = data

        if self.transform is not None:
            data = self.transform(data)

        return data


def collate_crystal_graphs(batch: List[Data]) -> Batch:
    """Collates a list of PyG Data objects into a single Batch object."""
    return Batch.from_data_list(batch)
