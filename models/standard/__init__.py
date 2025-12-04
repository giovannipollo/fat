"""Standard (full-precision) model implementations.

Provides standard neural network architectures for image classification
using full-precision (float32) weights and activations.

Available architectures:
    - CNV: Compact Neural Vision network
    - MobileNetV1: Lightweight mobile architecture
    - VGG: VGG-11/13/16/19 variants
    - ResNet CIFAR: ResNet-20/32/44/56/110 for 32x32 images
    - ResNet ImageNet: ResNet-18/34/50/101/152 adapted for small images
"""

from __future__ import annotations

from .cnv import CNV
from .mobilenet import MobileNetV1
from .vgg import VGG11, VGG13, VGG16, VGG19
from .resnet_cifar import (
    BasicBlock as CIFARBasicBlock,
    ResNetCIFAR,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110,
)
from .resnet_imagenet import (
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
    # VGG
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
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
