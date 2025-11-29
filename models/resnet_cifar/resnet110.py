"""!
@file models/resnet_cifar/resnet110.py
@brief ResNet-110 model for CIFAR datasets.
"""

from .base import ResNetCIFAR


class ResNet110(ResNetCIFAR):
    """!
    @brief ResNet-110 for CIFAR-10/100 and small image datasets.
    
    @details Architecture: 3 stages with 18 blocks each = 6*18 + 2 = 110 layers.
    Approximately 1.7M parameters. This is the deepest CIFAR ResNet variant.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """!
        @brief Initialize ResNet-110.
        
        @param num_classes Number of output classes
        @param in_channels Number of input channels
        """
        super().__init__(num_blocks=18, num_classes=num_classes, in_channels=in_channels)
