"""ResNet-32 model for CIFAR datasets."""

from .base import ResNetCIFAR


class ResNet32(ResNetCIFAR):
    """ResNet-32 for CIFAR-10/100 and small image datasets.

    Architecture: 3 stages with 5 blocks each = 6*5 + 2 = 32 layers.
    Approximately 0.46M parameters.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """Initialize ResNet-32.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
        """
        super().__init__(num_blocks=5, num_classes=num_classes, in_channels=in_channels)
