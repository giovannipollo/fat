"""Quantized MobileNetV1 architecture for ImageNet using Brevitas.

Implementation of quantized MobileNetV1 with depthwise separable convolutions,
designed for ImageNet-1K (224x224) inputs.

See: https://arxiv.org/abs/1704.04861
"""

from __future__ import annotations

from typing import ClassVar, List, Tuple

import torch
import torch.nn as nn
import brevitas.nn as qnn

from utils.weight_quant import CommonIntWeightPerChannelQuant
from utils.weight_quant import CommonIntWeightPerTensorQuant


class QuantDepthwiseSeparableBlock(nn.Module):
    """Quantized Depthwise Separable Convolution Block.

    Consists of a quantized depthwise convolution (per-channel spatial filtering)
    followed by a quantized pointwise convolution (1x1 cross-channel mixing).
    This reduces computation compared to standard convolutions.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        weight_quant=CommonIntWeightPerChannelQuant,
    ):
        """Initialize quantized depthwise separable block.

        Args:
            in_planes: Number of input channels.
            out_planes: Number of output channels.
            stride: Stride for depthwise convolution.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            weight_quant: Weight quantization method.
        """
        super().__init__()
        self.depthwise = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_planes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                return_quant_tensor=True,
            ),
            nn.BatchNorm2d(num_features=in_planes),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
        )

        self.pointwise = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                return_quant_tensor=True,
            ),
            nn.BatchNorm2d(num_features=out_planes),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized depthwise separable block.

        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H', W').
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class QuantMobileNetV1(nn.Module):
    """Quantized MobileNetV1 for ImageNet-1K.

    Original MobileNetV1 architecture designed for ImageNet (224x224) inputs,
    with quantized weights and activations using Brevitas.
    Uses depthwise separable convolutions for efficiency.

    Architecture:
        - Initial 3x3 conv stride=2 -> 32 channels
        - 27 depthwise separable layers organized in stages
        - Downsampling at stages 2, 3, 4, 5
        - Global average pooling
        - Fully connected classifier

    Parameters: ~4.2M
    """

    #: Architecture configuration: list of (out_channels, stride) tuples
    #: Each tuple represents a depthwise separable block
    CFG: ClassVar[List[Tuple[int, int]]] = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        in_weight_bit_width: int = 8,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        first_layer_weight_quant=CommonIntWeightPerChannelQuant,
        weight_quant=CommonIntWeightPerChannelQuant,
        last_layer_weight_quant=CommonIntWeightPerTensorQuant,
    ):
        """Initialize quantized MobileNetV1.

        Args:
            num_classes: Number of output classes (1000 for ImageNet-1K).
            in_channels: Number of input channels (3 for RGB).
            in_weight_bit_width: Bit width for input layer weight quantization.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            first_layer_weight_quant: Weight quantization for first layer.
            weight_quant: Weight quantization for hidden layers.
            last_layer_weight_quant: Weight quantization for last layer.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.weight_quant = weight_quant

        self.features = nn.Sequential(
            qnn.QuantIdentity(bit_width=act_bit_width),
            qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                weight_bit_width=in_weight_bit_width,
                weight_quant=first_layer_weight_quant,
                return_quant_tensor=True,
            ),
            nn.BatchNorm2d(num_features=32),
            qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True),
            self._make_layers(
                in_planes=32,
                weight_bit_width=weight_bit_width,
                act_bit_width=act_bit_width,
                weight_quant=weight_quant,
            ),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            qnn.QuantLinear(
                in_features=1024,
                out_features=num_classes,
                bias=True,
                weight_bit_width=weight_bit_width,
                weight_quant=last_layer_weight_quant,
            ),
        )

    def _make_layers(
        self,
        in_planes: int,
        weight_bit_width: int,
        act_bit_width: int,
        weight_quant=CommonIntWeightPerChannelQuant,
    ) -> nn.Sequential:
        """Build sequence of quantized depthwise separable blocks.

        Args:
            in_planes: Number of input channels to first block.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            weight_quant: Weight quantization method.

        Returns:
            Sequential container of quantized depthwise separable blocks.
        """
        layers: List[nn.Module] = []
        for out_planes, stride in self.CFG:
            layers.append(
                QuantDepthwiseSeparableBlock(
                    in_planes=in_planes,
                    out_planes=out_planes,
                    stride=stride,
                    weight_bit_width=weight_bit_width,
                    act_bit_width=act_bit_width,
                    weight_quant=weight_quant,
                )
            )
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized MobileNetV1.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
