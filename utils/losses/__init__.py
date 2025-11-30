"""!
@file utils/losses/__init__.py
@brief Custom loss functions module.

@details Provides custom loss function implementations beyond
standard PyTorch losses.
"""

from __future__ import annotations

from .sqr_hinge import SqrHingeLoss

__all__ = [
    "SqrHingeLoss",
]
