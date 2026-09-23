"""Deterministic seed utility for reproducibility across PyTorch, NumPy, and random."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> int:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed
