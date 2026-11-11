"""Random-state initialization shared by model construction and training."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and PyTorch CPU/CUDA random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
