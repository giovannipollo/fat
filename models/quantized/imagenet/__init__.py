"""Quantized classical ImageNet models.

Quantized versions of ImageNet architectures following the original
designs with configurable bit-width quantization using Brevitas.
"""

from __future__ import annotations

from .mobilenet import QuantMobileNetV1ImageNet, QuantMobileNetV1FinnImageNet
from .resnet import (
    QuantBasicBlock,
    QuantBottleneck,
    QuantResNetBase,
    QuantResNet18,
    QuantResNet34,
    QuantResNet50,
    QuantResNet101,
    QuantResNet152,
)

__all__ = [
    "QuantMobileNetV1ImageNet",
    "QuantMobileNetV1FinnImageNet",
    "QuantBasicBlock",
    "QuantBottleneck",
    "QuantResNetBase",
    "QuantResNet18",
    "QuantResNet34",
    "QuantResNet50",
    "QuantResNet101",
    "QuantResNet152",
]