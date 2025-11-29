"""ResNet-20 for CIFAR."""

from .base import ResNetCIFAR


class ResNet20(ResNetCIFAR):
    """
    ResNet-20 for CIFAR-10/100.
    
    Architecture: 3 stages with 3 blocks each = 6*3 + 2 = 20 layers
    Parameters: ~0.27M
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__(num_blocks=3, num_classes=num_classes, in_channels=in_channels)
