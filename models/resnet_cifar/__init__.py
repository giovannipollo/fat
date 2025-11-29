"""
CIFAR-specific ResNet models.

These models follow the original paper's CIFAR-10 architecture (Section 4.2):
- Initial conv: 3x3, 16 filters
- 3 stages with filter counts: 16 -> 32 -> 64
- Each stage has n blocks (total layers = 6n + 2)
- Global average pooling + FC layer

Available models:
- ResNet20: n=3, ~0.27M params
- ResNet32: n=5, ~0.46M params
- ResNet44: n=7, ~0.66M params
- ResNet56: n=9, ~0.85M params
- ResNet110: n=18, ~1.7M params

Reference:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
"""

from .base import BasicBlock, ResNetCIFAR
from .resnet20 import ResNet20
from .resnet32 import ResNet32
from .resnet44 import ResNet44
from .resnet56 import ResNet56
from .resnet110 import ResNet110

__all__ = [
    "BasicBlock",
    "ResNetCIFAR",
    "ResNet20",
    "ResNet32",
    "ResNet44",
    "ResNet56",
    "ResNet110",
]
