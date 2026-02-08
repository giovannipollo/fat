"""Quantized ImageNet MobileNet models.

Quantized versions of MobileNet architectures following the original
designs with configurable bit-width quantization using Brevitas.
"""

from .mobilenetv1 import QuantMobileNetV1 as QuantMobileNetV1ImageNet
from .mobilenetv1_finn import QuantMobileNetV1Finn as QuantMobileNetV1FinnImageNet

__all__ = [
    "QuantMobileNetV1ImageNet",
    "QuantMobileNetV1FinnImageNet",
]
