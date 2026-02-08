"""Quantized classical ImageNet ResNet models.

Quantized versions of ResNet-18 through ResNet-152 following the original
architecture from "Deep Residual Learning for Image Recognition" (He et al., 2015)
with configurable bit widths for weights and activations using Brevitas.

Available Models:
    | Model     | Blocks          | Type               | Parameters |
    |-----------|-----------------|--------------------|------------|
    | QuantResNet18  | [2, 2, 2, 2]    | QuantBasicBlock | ~11.7M     |
    | QuantResNet34  | [3, 4, 6, 3]    | QuantBasicBlock | ~21.8M     |
    | QuantResNet50  | [3, 4, 6, 3]    | QuantBottleneck | ~25.6M     |
    | QuantResNet101 | [3, 4, 23, 3]   | QuantBottleneck | ~44.5M     |
    | QuantResNet152 | [3, 8, 36, 3]   | QuantBottleneck | ~60.2M     |

See: https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"
"""

from .base import QuantBasicBlock, QuantBottleneck, QuantResNetBase
from .quant_resnet18 import QuantResNet18
from .quant_resnet34 import QuantResNet34
from .quant_resnet50 import QuantResNet50
from .quant_resnet101 import QuantResNet101
from .quant_resnet152 import QuantResNet152

__all__ = [
    "QuantBasicBlock",
    "QuantBottleneck",
    "QuantResNetBase",
    "QuantResNet18",
    "QuantResNet34",
    "QuantResNet50",
    "QuantResNet101",
    "QuantResNet152",
]
