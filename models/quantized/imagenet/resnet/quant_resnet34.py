"""Quantized ResNet-34 model for ImageNet classification."""

from .base import QuantResNetBase, QuantBasicBlock


class QuantResNet34(QuantResNetBase):
    """Quantized ResNet-34 model following the original architecture.

    Uses QuantBasicBlock with [3, 4, 6, 3] blocks per stage.
    Approximately 21.8M parameters for ImageNet-1K (1000 classes).

    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (QuantBasicBlock)
    block = QuantBasicBlock

    #: Number of blocks per stage
    layers = [3, 4, 6, 3]
