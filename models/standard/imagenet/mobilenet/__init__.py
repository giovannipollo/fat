"""MobileNetV1 architecture for ImageNet.

Implementation of MobileNetV1 with depthwise separable convolutions,
designed for ImageNet-1K (224x224) inputs.

See: https://arxiv.org/abs/1704.04861
"""

from __future__ import annotations

from .mobilenetv1 import MobileNetV1ImageNet
from .mobilenetv1 import DepthwiseSeparableBlock

__all__ = [
    "DepthwiseSeparableBlock",
    "MobileNetV1ImageNet",
]
