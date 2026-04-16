"""Reproducibility utilities for setting random seeds.

Provides comprehensive seeding for Python, NumPy, and PyTorch
to ensure reproducible training results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import torch


@dataclass
class RngState:
    """Snapshot of all active RNG streams.

    Attributes:
        python_state: Python stdlib random module state (getstate() tuple).
        numpy_state: NumPy random legacy generator state tuple.
        torch_state: PyTorch CPU RNG state tensor.
        cuda_states: Per-device CUDA RNG state tensors (empty on CPU-only).
    """

    python_state: object
    numpy_state: Any
    torch_state: torch.Tensor
    cuda_states: List[torch.Tensor] = field(default_factory=list)


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


def save_rng_state() -> RngState:
    """Capture the current state of all RNG streams.

    Returns:
        RngState snapshot to be passed to restore_rng_state().
    """
    cuda_states: List[torch.Tensor] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cuda_states.append(torch.cuda.get_rng_state(device=i))

    return RngState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(),
        cuda_states=cuda_states,
    )


def set_eval_seed(seed: int) -> None:
    """Reset all RNG streams to a fixed seed for evaluation.

    This function intentionally does not modify cuDNN flags.

    Args:
        seed: The evaluation seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def restore_rng_state(state: RngState) -> None:
    """Restore all RNG streams from a previously captured snapshot.

    Args:
        state: RngState snapshot produced by save_rng_state().

    Raises:
        RuntimeError: If the CUDA device count changed since the snapshot.
    """
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state)

    if torch.cuda.is_available():
        current_device_count = torch.cuda.device_count()
        if len(state.cuda_states) != current_device_count:
            raise RuntimeError(
                f"CUDA device count changed between save ({len(state.cuda_states)}) "
                f"and restore ({current_device_count})."
            )

        for i, cuda_state in enumerate(state.cuda_states):
            torch.cuda.set_rng_state(cuda_state, device=i)
