"""!
@file models/resnet_cifar/resnet32.py
@brief ResNet-32 model for CIFAR datasets.
"""

from .base import ResNetCIFAR


class ResNet32(ResNetCIFAR):
    """!
    @brief ResNet-32 for CIFAR-10/100 and small image datasets.
    
    @details Architecture: 3 stages with 5 blocks each = 6*5 + 2 = 32 layers.
    Approximately 0.46M parameters.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """!
        @brief Initialize ResNet-32.
        
        @param num_classes Number of output classes
        @param in_channels Number of input channels
        """
        super().__init__(num_blocks=5, num_classes=num_classes, in_channels=in_channels)
