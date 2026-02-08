"""ResNet-18 model for ImageNet classification."""

from .base import ResNetBase, BasicBlock


class ResNet18(ResNetBase):
    """ResNet-18 model following the original architecture.

    Uses BasicBlock with [2, 2, 2, 2] blocks per stage.
    Approximately 11.7M parameters for ImageNet-1K (1000 classes).
    
    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (BasicBlock)
    block = BasicBlock

    #: Number of blocks per stage
    layers = [2, 2, 2, 2]
