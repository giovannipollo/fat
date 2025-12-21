"""Quantized MobileNetV1 using Brevitas.

Implements quantized version of MobileNetV1 with depthwise
separable convolutions and configurable bit widths.
"""

from __future__ import annotations

from typing import ClassVar, List, Tuple

import torch
import torch.nn as nn
import brevitas.nn as qnn


class QuantDepthwiseSeparableBlock(nn.Module):
    """Quantized Depthwise Separable Convolution Block.

    Quantized version consisting of a depthwise convolution
    followed by a pointwise convolution.
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
            weight_bit_width: Bit width for weights.
            act_bit_width: Bit width for activations.
        """
        super().__init__()
        # Quantized depthwise convolution
        self.conv1 = self._make_quant_conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = self._make_quant_relu(act_bit_width)

        # Quantized pointwise convolution
        self.conv2 = self._make_quant_conv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_bit_width=weight_bit_width,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = self._make_quant_relu(act_bit_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized depthwise separable block."""
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class QuantMobileNetV1(nn.Module):
    """Quantized MobileNetV1 adapted for small images.

    Quantized version with configurable bit widths for
    weights and activations.
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
        **kwargs,
    ):
        """Initialize quantized MobileNetV1.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            weight_bit_width: Bit width for weights.
            act_bit_width: Bit width for activations.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.in_channels = in_channels
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        # Input quantization
        self.quant_inp = self._make_quant_identity(act_bit_width)

        # Initial Conv Layer
        self.conv1 = self._make_quant_conv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = self._make_quant_relu(act_bit_width)

        # MobileNet Body
        self.layers = self._make_layers(in_planes=32)

        # Classifier
        self.linear = self._make_quant_linear(
            1024,
            num_classes,
            weight_bit_width=weight_bit_width,
        )

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        """Build the sequence of quantized depthwise separable blocks.

        Args:
            in_planes: Number of input channels to the first block.

        Returns:
            Sequential container of blocks.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized MobileNetV1.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        out = self.quant_inp(x)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.layers(out)
        out = nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
