"""ResNet-50 model."""

from .base import ResNetBase, Bottleneck


class ResNet50(ResNetBase):
    """
    ResNet-50 model adapted for CIFAR-10/100 and MNIST.
    
    Architecture: [3, 4, 6, 3] Bottlenecks
    Parameters: ~23.5M
    """
    
    block = Bottleneck
    layers = [3, 4, 6, 3]
