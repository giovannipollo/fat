"""!
@file models/resnet_imagenet/resnet34.py
@brief ResNet-34 model adapted for small images.
"""

from .base import ResNetBase, BasicBlock


class ResNet34(ResNetBase):
    """!
    @brief ResNet-34 model adapted for CIFAR-10/100 and MNIST.
    
    @details Uses BasicBlock with [3, 4, 6, 3] blocks per stage.
    Approximately 21.3M parameters.
    """
    
    ## @var block
    #  @brief Block type (BasicBlock)
    block = BasicBlock
    
    ## @var layers
    #  @brief Number of blocks per stage
    layers = [3, 4, 6, 3]
