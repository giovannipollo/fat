"""ImageNet-style models adapted for small images.

Models based on ImageNet architectures but modified for smaller input sizes.
"""

from __future__ import annotations

from .mobilenet import MobileNetV1
from .resnet import (
    BasicBlock,
    Bottleneck,
    ResNetBase,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNetBase",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "MobileNetV1",
]