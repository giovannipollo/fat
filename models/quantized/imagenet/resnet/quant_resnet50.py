"""Quantized ResNet-50 model for ImageNet classification."""

from .base import QuantResNetBase, QuantBottleneck


class QuantResNet50(QuantResNetBase):
    """Quantized ResNet-50 model following the original architecture.

    Uses QuantBottleneck blocks with [3, 4, 6, 3] blocks per stage.
    Approximately 25.6M parameters for ImageNet-1K (1000 classes).

    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (QuantBottleneck with 4x expansion)
    block = QuantBottleneck

    #: Number of blocks per stage
    layers = [3, 4, 6, 3]
