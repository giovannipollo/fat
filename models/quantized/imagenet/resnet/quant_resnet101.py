"""Quantized ResNet-101 model for ImageNet classification."""

from .base import QuantResNetBase, QuantBottleneck


class QuantResNet101(QuantResNetBase):
    """Quantized ResNet-101 model following the original architecture.

    Uses QuantBottleneck blocks with [3, 4, 23, 3] blocks per stage.
    Approximately 44.5M parameters for ImageNet-1K (1000 classes).

    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (QuantBottleneck with 4x expansion)
    block = QuantBottleneck

    #: Number of blocks per stage
    layers = [3, 4, 23, 3]
