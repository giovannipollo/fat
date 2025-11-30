"""ResNet-152 model adapted for small images."""

from .base import ResNetBase, Bottleneck


class ResNet152(ResNetBase):
    """ResNet-152 model adapted for CIFAR-10/100 and MNIST.

    Uses Bottleneck blocks with [3, 8, 36, 3] blocks per stage.
    Approximately 58.2M parameters. This is the deepest ImageNet-style ResNet.
    """

    #: Block type (Bottleneck with 4x expansion)
    block = Bottleneck

    #: Number of blocks per stage
    layers = [3, 8, 36, 3]
