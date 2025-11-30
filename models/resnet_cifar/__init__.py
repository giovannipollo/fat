"""CIFAR-specific ResNet models.

These models follow the original paper's CIFAR-10 architecture (Section 4.2):
- Initial conv: 3x3, 16 filters
- 3 stages with filter counts: 16 -> 32 -> 64
- Each stage has n blocks (total layers = 6n + 2)
- Global average pooling + FC layer

Available Models:
    | Model     | n  | Parameters |
    |-----------|----|-----------:|
    | ResNet20  | 3  | ~0.27M     |
    | ResNet32  | 5  | ~0.46M     |
    | ResNet44  | 7  | ~0.66M     |
    | ResNet56  | 9  | ~0.85M     |
    | ResNet110 | 18 | ~1.7M      |

See: https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"
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
