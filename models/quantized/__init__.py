"""!
@file models/quantized/__init__.py
@brief Quantized model module initialization.

@details Provides quantized versions of all models using Brevitas for
quantization-aware training (QAT). Supports configurable bit widths
for weights and activations.

@see get_quantized_model for the main factory function
"""

from __future__ import annotations

from .resnet_cifar import (
    QuantResNet20,
    QuantResNet32,
    QuantResNet44,
    QuantResNet56,
    QuantResNet110,
)
from .resnet_imagenet import (
    QuantResNet18,
    QuantResNet34,
    QuantResNet50,
    QuantResNet101,
    QuantResNet152,
)
from .vgg import QuantVGG11, QuantVGG13, QuantVGG16, QuantVGG19
from .mobilenet import QuantMobileNetV1
from .cnv import QuantCNV

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
    # VGG
    "QuantVGG11",
    "QuantVGG13",
    "QuantVGG16",
    "QuantVGG19",
    # MobileNet
    "QuantMobileNetV1",
    # CNV
    "QuantCNV",
]
