"""!
@file models/resnet_imagenet/__init__.py
@brief ImageNet-style ResNet models adapted for small images.

@details These models follow the original ResNet architecture for ImageNet but with
modifications for smaller input sizes (32x32, 28x28):
- Initial conv: 3x3, stride=1 (vs 7x7, stride=2 in original)
- No max pooling after initial conv

@par Available Models
| Model     | Blocks          | Type       | Parameters |
|-----------|-----------------|------------|------------|
| ResNet18  | [2, 2, 2, 2]    | BasicBlock | ~11.2M     |
| ResNet34  | [3, 4, 6, 3]    | BasicBlock | ~21.3M     |
| ResNet50  | [3, 4, 6, 3]    | Bottleneck | ~23.5M     |
| ResNet101 | [3, 4, 23, 3]   | Bottleneck | ~42.5M     |
| ResNet152 | [3, 8, 36, 3]   | Bottleneck | ~58.2M     |

@see https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"
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
