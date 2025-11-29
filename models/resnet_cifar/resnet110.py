"""ResNet-110 for CIFAR."""

from .base import ResNetCIFAR


class ResNet110(ResNetCIFAR):
    """
    ResNet-110 for CIFAR-10/100.
    
    Architecture: 3 stages with 18 blocks each = 6*18 + 2 = 110 layers
    Parameters: ~1.7M
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__(num_blocks=18, num_classes=num_classes, in_channels=in_channels)
