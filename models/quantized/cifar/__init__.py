"""Quantized CIFAR-specific models.

Quantized versions of models optimized for small image datasets.
"""

from __future__ import annotations

from .resnet import (
    QuantResNet20,
    QuantResNet32,
    QuantResNet44,
    QuantResNet56,
    QuantResNet110,
)
from .mobilenetv1 import QuantMobileNetV1
from .cnv import QuantCNV

__all__ = [
    "QuantResNet20",
    "QuantResNet32",
    "QuantResNet44",
    "QuantResNet56",
    "QuantResNet110",
    "QuantMobileNetV1",
    "QuantCNV",
]