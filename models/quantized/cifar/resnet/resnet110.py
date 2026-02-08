"""Quantized ResNet-110 model for CIFAR datasets."""

from .base import QuantResNetCIFAR
from utils.weight_quant import CommonIntWeightPerChannelQuant
from utils.weight_quant import CommonIntWeightPerTensorQuant


class QuantResNet110(QuantResNetCIFAR):
    """Quantized ResNet-110 for CIFAR-10/100 and small image datasets.

    Architecture: 3 stages with 18 blocks each = 6*18 + 2 = 110 layers.
    Approximately 1.7M parameters. This is the deepest CIFAR ResNet variant.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        in_weight_bit_width: int = 8,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """Initialize Quantized ResNet-110.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            in_weight_bit_width: Bit width for input weight quantization.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
        """
        super().__init__(
            num_blocks=18,
            num_classes=num_classes,
            in_channels=in_channels,
            in_weight_bit_width=in_weight_bit_width,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            first_layer_weight_quant=CommonIntWeightPerChannelQuant,
            weight_quant=CommonIntWeightPerChannelQuant,
            last_layer_weight_quant=CommonIntWeightPerTensorQuant,
        )
