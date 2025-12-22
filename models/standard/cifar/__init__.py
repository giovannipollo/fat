"""CIFAR-specific models.

Models optimized for small image datasets like CIFAR-10/100 (32x32 images).
"""

from __future__ import annotations

from .cnv import CNV
from .mobilenet import MobileNetV1
from .resnet import (
    BasicBlock as CIFARBasicBlock,
    ResNetCIFAR,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110,
)

__all__ = [
    "CNV",
    "MobileNetV1",
    "CIFARBasicBlock",
    "ResNetCIFAR",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
]