"""ResNet-56 for CIFAR."""

from .base import ResNetCIFAR


class ResNet56(ResNetCIFAR):
    """
    ResNet-56 for CIFAR-10/100.
    
    Architecture: 3 stages with 9 blocks each = 6*9 + 2 = 56 layers
    Parameters: ~0.85M
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__(num_blocks=9, num_classes=num_classes, in_channels=in_channels)
