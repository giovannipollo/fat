"""Quantized ResNet-32 model for CIFAR datasets."""

from .base import QuantResNetCIFAR


class QuantResNet32(QuantResNetCIFAR):
    """Quantized ResNet-32 for CIFAR-10/100 and small image datasets.

    Architecture: 3 stages with 5 blocks each = 6*5 + 2 = 32 layers.
    Approximately 0.46M parameters.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        in_weight_bit_width: int = 8,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        **kwargs,
    ):
        """Initialize Quantized ResNet-32.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            in_weight_bit_width: Bit width for input weight quantization.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(
            num_blocks=5,
            num_classes=num_classes,
            in_channels=in_channels,
            in_weight_bit_width=in_weight_bit_width,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
        )
