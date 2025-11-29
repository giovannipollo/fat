"""
ImageNet-style ResNet models adapted for smaller images (CIFAR, MNIST).

These models follow the original ResNet architecture for ImageNet but with
modifications for smaller input sizes:
- Initial conv: 3x3, stride=1 (vs 7x7, stride=2)
- No max pooling after initial conv

Available models:
- ResNet18: [2, 2, 2, 2] BasicBlocks, ~11.2M params
- ResNet34: [3, 4, 6, 3] BasicBlocks, ~21.3M params
- ResNet50: [3, 4, 6, 3] Bottlenecks, ~23.5M params
- ResNet101: [3, 4, 23, 3] Bottlenecks, ~42.5M params
- ResNet152: [3, 8, 36, 3] Bottlenecks, ~58.2M params
"""

from .base import BasicBlock, Bottleneck, ResNetBase
from .resnet18 import ResNet18
from .resnet34 import ResNet34
from .resnet50 import ResNet50
from .resnet101 import ResNet101
from .resnet152 import ResNet152

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNetBase",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
]
