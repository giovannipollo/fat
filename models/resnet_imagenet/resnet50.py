"""!
@file models/resnet_imagenet/resnet50.py
@brief ResNet-50 model adapted for small images.
"""

from .base import ResNetBase, Bottleneck


class ResNet50(ResNetBase):
    """!
    @brief ResNet-50 model adapted for CIFAR-10/100 and MNIST.
    
    @details Uses Bottleneck blocks with [3, 4, 6, 3] blocks per stage.
    Approximately 23.5M parameters.
    """
    
    ## @var block
    #  @brief Block type (Bottleneck with 4x expansion)
    block = Bottleneck
    
    ## @var layers
    #  @brief Number of blocks per stage
    layers = [3, 4, 6, 3]
