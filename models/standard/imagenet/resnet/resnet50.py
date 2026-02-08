"""ResNet-50 model for ImageNet classification."""

from .base import ResNetBase, Bottleneck


class ResNet50(ResNetBase):
    """ResNet-50 model following the original architecture.

    Uses Bottleneck blocks with [3, 4, 6, 3] blocks per stage.
    Approximately 25.6M parameters for ImageNet-1K (1000 classes).
    
    Architecture designed for 224x224 RGB inputs.
    """

    #: Block type (Bottleneck with 4x expansion)
    block = Bottleneck

    #: Number of blocks per stage
    layers = [3, 4, 6, 3]
