"""Quantized ResNet-152 model for ImageNet classification."""

from .base import QuantResNetBase, QuantBottleneck


class QuantResNet152(QuantResNetBase):
    """Quantized ResNet-152 model following the original architecture.

    Uses QuantBottleneck blocks with [3, 8, 36, 3] blocks per stage.
    Approximately 60.2M parameters for ImageNet-1K (1000 classes).

    This is the deepest standard quantized ResNet variant.
    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (QuantBottleneck with 4x expansion)
    block = QuantBottleneck

    #: Number of blocks per stage
    layers = [3, 8, 36, 3]
