"""!
@file models/resnet_imagenet/resnet18.py
@brief ResNet-18 model adapted for small images.
"""

from .base import ResNetBase, BasicBlock


class ResNet18(ResNetBase):
    """!
    @brief ResNet-18 model adapted for CIFAR-10/100 and MNIST.
    
    @details Uses BasicBlock with [2, 2, 2, 2] blocks per stage.
    Approximately 11.2M parameters.
    """
    
    ## @var block
    #  @brief Block type (BasicBlock)
    block = BasicBlock
    
    ## @var layers
    #  @brief Number of blocks per stage
    layers = [2, 2, 2, 2]
