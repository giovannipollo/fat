"""Classical ImageNet-style ResNet models.

These models follow the original ResNet architecture from the paper
"Deep Residual Learning for Image Recognition" (He et al., 2015).

Architecture:
- Initial conv: 7x7, stride=2, padding=3
- Max pooling: 3x3, stride=2, padding=1
- Four residual stages with [64, 128, 256, 512] base channels
- Global average pooling + fully connected classifier
- Designed for 224x224 ImageNet inputs

Available Models:
    | Model     | Blocks          | Type       | Parameters |
    |-----------|-----------------|------------|------------|
    | ResNet18  | [2, 2, 2, 2]    | BasicBlock | ~11.7M     |
    | ResNet34  | [3, 4, 6, 3]    | BasicBlock | ~21.8M     |
    | ResNet50  | [3, 4, 6, 3]    | Bottleneck | ~25.6M     |
    | ResNet101 | [3, 4, 23, 3]   | Bottleneck | ~44.5M     |
    | ResNet152 | [3, 8, 36, 3]   | Bottleneck | ~60.2M     |

See: https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"
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
