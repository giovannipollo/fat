"""!
@file utils/seed.py
@brief Reproducibility utilities for setting random seeds.

@details Provides comprehensive seeding for Python, NumPy, and PyTorch
to ensure reproducible training results.
"""

from __future__ import annotations

import random

import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """!
    @brief Set random seeds for reproducibility.
    
    @details Seeds Python's random, NumPy (if available), and PyTorch
    random number generators. Optionally enables deterministic algorithms.
    
    @param seed The random seed to use
    @param deterministic If True, use deterministic algorithms (may reduce performance)
    
    @note When deterministic=True, cuDNN benchmark mode is disabled and
    deterministic algorithms are enforced, which may slow down training.
    """
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
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
