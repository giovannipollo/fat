"""ResNet-32 for CIFAR."""

from .base import ResNetCIFAR


class ResNet32(ResNetCIFAR):
    """
    ResNet-32 for CIFAR-10/100.
    
    Architecture: 3 stages with 5 blocks each = 6*5 + 2 = 32 layers
    Parameters: ~0.46M
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__(num_blocks=5, num_classes=num_classes, in_channels=in_channels)
