"""ResNet-44 for CIFAR."""

from .base import ResNetCIFAR


class ResNet44(ResNetCIFAR):
    """
    ResNet-44 for CIFAR-10/100.
    
    Architecture: 3 stages with 7 blocks each = 6*7 + 2 = 44 layers
    Parameters: ~0.66M
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__(num_blocks=7, num_classes=num_classes, in_channels=in_channels)
