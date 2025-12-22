"""ResNet-18 model adapted for small images."""

from .base import ResNetBase, BasicBlock


class ResNet18(ResNetBase):
    """ResNet-18 model adapted for CIFAR-10/100 and MNIST.

    Uses BasicBlock with [2, 2, 2, 2] blocks per stage.
    Approximately 11.2M parameters.
    """

    #: Block type (BasicBlock)
    block = BasicBlock

    #: Number of blocks per stage
    layers = [2, 2, 2, 2]
