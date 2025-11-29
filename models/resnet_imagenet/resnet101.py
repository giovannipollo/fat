"""!
@file models/resnet_imagenet/resnet101.py
@brief ResNet-101 model adapted for small images.
"""

from .base import ResNetBase, Bottleneck


class ResNet101(ResNetBase):
    """!
    @brief ResNet-101 model adapted for CIFAR-10/100 and MNIST.
    
    @details Uses Bottleneck blocks with [3, 4, 23, 3] blocks per stage.
    Approximately 42.5M parameters.
    """
    
    ## @var block
    #  @brief Block type (Bottleneck with 4x expansion)
    block = Bottleneck
    
    ## @var layers
    #  @brief Number of blocks per stage
    layers = [3, 4, 23, 3]
