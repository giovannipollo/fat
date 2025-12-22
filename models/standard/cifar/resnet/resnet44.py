"""ResNet-44 model for CIFAR datasets."""

from .base import ResNetCIFAR


class ResNet44(ResNetCIFAR):
    """ResNet-44 for CIFAR-10/100 and small image datasets.

    Architecture: 3 stages with 7 blocks each = 6*7 + 2 = 44 layers.
    Approximately 0.66M parameters.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """Initialize ResNet-44.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
        """
        super().__init__(num_blocks=7, num_classes=num_classes, in_channels=in_channels)
