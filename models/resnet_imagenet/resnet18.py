"""ResNet-18 model."""

from .base import ResNetBase, BasicBlock


class ResNet18(ResNetBase):
    """
    ResNet-18 model adapted for CIFAR-10/100 and MNIST.
    
    Architecture: [2, 2, 2, 2] BasicBlocks
    Parameters: ~11.2M
    """
    
    block = BasicBlock
    layers = [2, 2, 2, 2]
