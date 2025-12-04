"""ResNet-34 model adapted for small images."""

from .base import ResNetBase, BasicBlock


class ResNet34(ResNetBase):
    """ResNet-34 model adapted for CIFAR-10/100 and MNIST.

    Uses BasicBlock with [3, 4, 6, 3] blocks per stage.
    Approximately 21.3M parameters.
    """

    #: Block type (BasicBlock)
    block = BasicBlock

    #: Number of blocks per stage
    layers = [3, 4, 6, 3]
