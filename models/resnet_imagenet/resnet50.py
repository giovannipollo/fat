"""ResNet-50 model adapted for small images."""

from .base import ResNetBase, Bottleneck


class ResNet50(ResNetBase):
    """ResNet-50 model adapted for CIFAR-10/100 and MNIST.

    Uses Bottleneck blocks with [3, 4, 6, 3] blocks per stage.
    Approximately 23.5M parameters.
    """

    #: Block type (Bottleneck with 4x expansion)
    block = Bottleneck

    #: Number of blocks per stage
    layers = [3, 4, 6, 3]
