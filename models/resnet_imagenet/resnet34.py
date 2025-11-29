"""ResNet-34 model."""

from .base import ResNetBase, BasicBlock


class ResNet34(ResNetBase):
    """
    ResNet-34 model adapted for CIFAR-10/100 and MNIST.
    
    Architecture: [3, 4, 6, 3] BasicBlocks
    Parameters: ~21.3M
    """
    
    block = BasicBlock
    layers = [3, 4, 6, 3]
