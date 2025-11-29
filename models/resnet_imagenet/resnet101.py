"""ResNet-101 model."""

from .base import ResNetBase, Bottleneck


class ResNet101(ResNetBase):
    """
    ResNet-101 model adapted for CIFAR-10/100 and MNIST.
    
    Architecture: [3, 4, 23, 3] Bottlenecks
    Parameters: ~42.5M
    """
    
    block = Bottleneck
    layers = [3, 4, 23, 3]
