"""Standard (full-precision) model implementations.

Provides standard neural network architectures for image classification
using full-precision (float32) weights and activations.

Organized by dataset type:
- cifar/: Models for CIFAR-10/100 (32x32 images)
  - CNV: Compact Neural Vision network
  - MobileNetV1: Lightweight mobile architecture
  - ResNet: ResNet-20/32/44/56/110 for 32x32 images
- imagenet/: ImageNet-style models adapted for small images
  - ResNet: ResNet-18/34/50/101/152 adapted for 32x32/28x28 images
"""

from __future__ import annotations

from .cifar.cnv import CNV
from .cifar.mobilenetv1 import MobileNetV1
from .cifar.resnet import (
    BasicBlock as CIFARBasicBlock,
    ResNetCIFAR,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110,
)
from .imagenet.resnet import (
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
    # CNV
    "CNV",
    # MobileNet
    "MobileNetV1",
    # ResNet CIFAR
    "CIFARBasicBlock",
    "ResNetCIFAR",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
    # ResNet ImageNet
    "BasicBlock",
    "Bottleneck",
    "ResNetBase",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]
