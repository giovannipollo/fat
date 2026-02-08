"""CIFAR-specific models.

Models optimized for small image datasets like CIFAR-10/100 (32x32 images).
"""

from __future__ import annotations

from .cnv import CNV
from .mobilenet.mobilenetv1 import MobileNetV1 as MobileNetV1CIFAR
from .mobilenet.mobilenetv1_finn import MobileNetV1Finn as MobileNetV1FinnCIFAR
from .resnet import (
    BasicBlock as CIFARBasicBlock,
    ResNetCIFAR,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110
)

__all__ = [
    "CNV",
    "MobileNetV1CIFAR",
    "MobileNetV1FinnCIFAR",
    "CIFARBasicBlock",
    "ResNetCIFAR",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
]