"""Reproducibility utilities for setting random seeds.

Provides comprehensive seeding for Python, NumPy, and PyTorch
to ensure reproducible training results.
"""

from __future__ import annotations

import random
import torch
import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Seeds Python's random, NumPy (if available), and PyTorch
    random number generators. Optionally enables deterministic algorithms.

    Args:
        seed: The random seed to use.
        deterministic: If True, use deterministic algorithms (may reduce
            performance).

    Note:
        When deterministic=True, cuDNN benchmark mode is disabled and
        deterministic algorithms are enforced, which may slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True can speed up training when input sizes don't vary
        torch.backends.cudnn.benchmark = True
