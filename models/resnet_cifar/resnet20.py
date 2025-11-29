"""!
@file models/resnet_cifar/resnet20.py
@brief ResNet-20 model for CIFAR datasets.
"""

from .base import ResNetCIFAR


class ResNet20(ResNetCIFAR):
    """!
    @brief ResNet-20 for CIFAR-10/100 and small image datasets.
    
    @details Architecture: 3 stages with 3 blocks each = 6*3 + 2 = 20 layers.
    Approximately 0.27M parameters.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """!
        @brief Initialize ResNet-20.
        
        @param num_classes Number of output classes
        @param in_channels Number of input channels
        """
        super().__init__(num_blocks=3, num_classes=num_classes, in_channels=in_channels)
