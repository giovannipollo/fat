"""!
@file models/resnet_imagenet/resnet152.py
@brief ResNet-152 model adapted for small images.
"""

from .base import ResNetBase, Bottleneck


class ResNet152(ResNetBase):
    """!
    @brief ResNet-152 model adapted for CIFAR-10/100 and MNIST.
    
    @details Uses Bottleneck blocks with [3, 8, 36, 3] blocks per stage.
    Approximately 58.2M parameters. This is the deepest ImageNet-style ResNet.
    """
    
    ## @var block
    #  @brief Block type (Bottleneck with 4x expansion)
    block = Bottleneck
    
    ## @var layers
    #  @brief Number of blocks per stage
    layers = [3, 8, 36, 3]
