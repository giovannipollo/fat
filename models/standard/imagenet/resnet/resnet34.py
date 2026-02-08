"""ResNet-34 model for ImageNet classification."""

from .base import ResNetBase, BasicBlock


class ResNet34(ResNetBase):
    """ResNet-34 model following the original architecture.

    Uses BasicBlock with [3, 4, 6, 3] blocks per stage.
    Approximately 21.8M parameters for ImageNet-1K (1000 classes).
    
    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (BasicBlock)
    block = BasicBlock

    #: Number of blocks per stage
    layers = [3, 4, 6, 3]
