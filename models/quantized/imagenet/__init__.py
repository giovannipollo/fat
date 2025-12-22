"""Quantized ImageNet-style models.

Quantized versions of ImageNet architectures adapted for small images.
"""

from __future__ import annotations

from .resnet import (
    QuantResNet18,
    QuantResNet34,
    QuantResNet50,
    QuantResNet101,
    QuantResNet152,
)

__all__ = [
    "QuantResNet18",
    "QuantResNet34",
    "QuantResNet50",
    "QuantResNet101",
    "QuantResNet152",
]