"""ResNet-152 model."""

from .base import ResNetBase, Bottleneck


class ResNet152(ResNetBase):
    """
    ResNet-152 model adapted for CIFAR-10/100 and MNIST.
    
    Architecture: [3, 8, 36, 3] Bottlenecks
    Parameters: ~58.2M
    """
    
    block = Bottleneck
    layers = [3, 8, 36, 3]
