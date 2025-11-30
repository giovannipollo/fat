"""Device selection utilities for training.

Provides automatic selection of the best available compute device.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Get the best available device for training.

    Checks for available accelerators in order of preference:

    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        torch.device for the selected compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
