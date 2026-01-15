"""CIFAR-specific models.

Models optimized for small image datasets like CIFAR-10/100 (32x32 images).
"""

from __future__ import annotations

from .cnv import CNV
from .mobilenetv1 import MobileNetV1
from .mobilenetv1_finn import MobileNetV1Finn
from .resnet import (
    BasicBlock as CIFARBasicBlock,
    ResNetCIFAR,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110,
    MobileNetV1Finn
)

__all__ = [
    "CNV",
    "MobileNetV1",
    "MobileNetV1Finn",
    "CIFARBasicBlock",
    "ResNetCIFAR",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
]