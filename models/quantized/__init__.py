"""Quantized model module initialization.

Provides quantized versions of all models using Brevitas for
quantization-aware training (QAT). Supports configurable bit widths
for weights and activations.

See: get_quantized_model for the main factory function
"""

from __future__ import annotations

from .cifar.resnet import (
    QuantResNet20,
    QuantResNet32,
    QuantResNet44,
    QuantResNet56,
    QuantResNet110,
)
from .imagenet.resnet import (
    QuantResNet18,
    QuantResNet34,
    QuantResNet50,
    QuantResNet101,
    QuantResNet152,
)
from .cifar.mobilenetv1 import QuantMobileNetV1
from .cifar.cnv import QuantCNV

__all__ = [
    # ResNet CIFAR
    "QuantResNet20",
    "QuantResNet32",
    "QuantResNet44",
    "QuantResNet56",
    "QuantResNet110",
    # ResNet ImageNet
    "QuantResNet18",
    "QuantResNet34",
    "QuantResNet50",
    "QuantResNet101",
    "QuantResNet152",
    # MobileNet
    "QuantMobileNetV1",
    "QuantMobileNetV1",
    # CNV
    "QuantCNV",
]
