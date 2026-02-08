"""CIFAR-specific Quantized ResNet models.

These models follow the original paper's CIFAR-10 architecture (Section 4.2):
- Initial conv: 3x3, 16 filters
- 3 stages with filter counts: 16 -> 32 -> 64
- Each stage has n blocks (total layers = 6n + 2)
- Global average pooling + FC layer

Available Models:
    | Model      | n  | Parameters |
    |------------|----|-----------:|
    | QuantResNet20  | 3  | ~0.27M     |
    | QuantResNet32  | 5  | ~0.46M     |
    | QuantResNet44  | 7  | ~0.66M     |
    | QuantResNet56  | 9  | ~0.85M     |
    | QuantResNet110 | 18 | ~1.7M      |

See: https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"
"""

from .base import QuantBasicBlock, QuantResNetCIFAR
from .resnet20 import QuantResNet20
from .resnet32 import QuantResNet32
from .resnet44 import QuantResNet44
from .resnet56 import QuantResNet56
from .resnet110 import QuantResNet110

__all__ = [
    "QuantBasicBlock",
    "QuantResNetCIFAR",
    "QuantResNet20",
    "QuantResNet32",
    "QuantResNet44",
    "QuantResNet56",
    "QuantResNet110",
]
