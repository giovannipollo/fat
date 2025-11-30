"""ResNet-56 model for CIFAR datasets."""

from .base import ResNetCIFAR


class ResNet56(ResNetCIFAR):
    """ResNet-56 for CIFAR-10/100 and small image datasets.

    Architecture: 3 stages with 9 blocks each = 6*9 + 2 = 56 layers.
    Approximately 0.85M parameters.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """Initialize ResNet-56.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
        """
        super().__init__(num_blocks=9, num_classes=num_classes, in_channels=in_channels)
