"""Quantized MobileNetV1 for CIFAR datasets using Brevitas.

Implements quantized versions of CIFAR-specific MobileNetV1 architecture
with configurable bit widths for weights and activations.
"""

from __future__ import annotations

from typing import ClassVar, List, Tuple

import torch
import torch.nn as nn
import brevitas.nn as qnn


class QuantDepthwiseSeparableBlock(nn.Module):
    """Quantized Depthwise Separable Convolution Block for CIFAR MobileNetV1.

    Consists of a quantized depthwise convolution (per-channel spatial filtering)
    followed by a quantized pointwise convolution (1x1 cross-channel mixing),
    both with batch normalization and quantized ReLU activations.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """Initialize quantized depthwise separable block.

        Args:
            in_planes: Number of input channels.
            out_planes: Number of output channels.
            stride: Stride for the depthwise convolution.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
        """
        super().__init__()
        self.dw_conv = qnn.QuantConv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width)

        self.pw_conv = qnn.QuantConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_bit_width=weight_bit_width,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized depthwise separable block.

        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H', W').
        """
        out: torch.Tensor = self.relu1(self.bn1(self.dw_conv(x)))
        out = self.relu2(self.bn2(self.pw_conv(out)))
        return out


class QuantMobileNetV1(nn.Module):
    """Quantized CIFAR-specific MobileNetV1 architecture.

    Quantized version following the standard CIFAR-10 MobileNetV1 architecture
    with configurable bit widths for weights and activations.
    Modified for 32x32 inputs: initial stride=1 to preserve resolution,
    reduced downsampling to prevent features from becoming too small.

    Architecture:
        - Initial 3x3 conv -> 32 channels (stride=1)
        - 13 quantized depthwise separable blocks
        - Global average pooling (kernel_size=2)
        - Quantized fully connected classifier
    """

    #: Architecture configuration: list of (out_channels, stride) tuples
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
        num_classes: int = 10,
        in_channels: int = 3,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """Initialize quantized CIFAR MobileNetV1.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels (3 for RGB, 1 for grayscale).
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit_width)

        self.conv1 = qnn.QuantConv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = qnn.QuantReLU(bit_width=act_bit_width)

        self.layers = self._make_layers(in_planes=32)

        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.fc = qnn.QuantLinear(
            1024,
            num_classes,
            bias=True,
            weight_bit_width=weight_bit_width,
        )

        self._initialize_weights()

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        """Build the sequence of quantized depthwise separable blocks.

        Args:
            in_planes: Number of input channels to the first block.

        Returns:
            Sequential container of quantized depthwise separable blocks.
        """
        layers: List[nn.Module] = []
        for out_planes, stride in self.CFG:
            layers.append(
                QuantDepthwiseSeparableBlock(
                    in_planes,
                    out_planes,
                    stride,
                    self.weight_bit_width,
                    self.act_bit_width,
                )
            )
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized MobileNetV1.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        out: torch.Tensor = self.quant_inp(x)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out