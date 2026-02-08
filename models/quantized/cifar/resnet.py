"""Quantized ResNet models for CIFAR using Brevitas.

Implements quantized versions of CIFAR-specific ResNet architectures
with configurable bit widths for weights and activations.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import brevitas.nn as qnn

from utils.weight_quant import CommonIntWeightPerChannelQuant
from utils.weight_quant import CommonIntWeightPerTensorQuant


class QuantBasicBlock(nn.Module):
    """Quantized basic residual block for CIFAR ResNets.

    Two quantized 3x3 convolutions with batch normalization,
    quantized ReLU activations, and skip connection.
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        weight_quant=CommonIntWeightPerChannelQuant,
    ):
        """Initialize quantized basic residual block.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for first convolution.
            downsample: Optional downsampling layer for skip connection.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
        """
        super().__init__()
        self.conv1 = self._make_quant_conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu1 = self._make_quant_relu(bit_width=act_bit_width)

        self.conv2 = self._make_quant_conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.relu2 = self._make_quant_relu(bit_width=act_bit_width)

        self.downsample = downsample

    def _make_quant_conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight_bit_width: int = 8,
        weight_quant=CommonIntWeightPerChannelQuant,
    ) -> qnn.QuantConv2d:
        """Create quantized convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides.
            weight_bit_width: Bit width for weight quantization.

        Returns:
            Quantized convolution layer.
        """
        return qnn.QuantConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            return_quant_tensor=True,
        )

    def _make_quant_relu(self, bit_width: int = 8) -> qnn.QuantReLU:
        """Create quantized ReLU activation.

        Args:
            bit_width: Bit width for activation quantization.

        Returns:
            Quantized ReLU layer.
        """
        return qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with skip connection added.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)

        return out


class QuantResNetCIFAR(nn.Module):
    """Quantized CIFAR-specific ResNet architecture.

    Quantized version following the original paper's CIFAR-10 architecture
    with configurable bit widths for weights and activations.
    """

    def __init__(
        self,
        num_blocks: int,
        num_classes: int = 10,
        in_channels: int = 3,
        in_weight_bit_width: int = 8,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        first_layer_weight_quant=CommonIntWeightPerChannelQuant,
        weight_quant=CommonIntWeightPerChannelQuant,
        last_layer_weight_quant=CommonIntWeightPerTensorQuant,
    ):
        """Initialize quantized CIFAR ResNet.

        Args:
            num_blocks: Number of blocks per stage.
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            in_weight_bit_width: Bit width for input weight quantization.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
        """
        super().__init__()
        self.in_planes: int = 16
        self.in_weight_bit_width = in_weight_bit_width
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.quant_inp = self._make_quant_identity(act_bit_width)

        # Initial convolution
        self.conv1 = self._make_quant_conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_bit_width=in_weight_bit_width,
            weight_quant=first_layer_weight_quant,
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = self._make_quant_relu(bit_width=self.act_bit_width)

        # Three stages with increasing filter counts
        self.layer1 = self._make_layer(
            planes=16, num_blocks=num_blocks, stride=1, weight_quant=weight_quant
        )
        self.layer2 = self._make_layer(
            planes=32, num_blocks=num_blocks, stride=2, weight_quant=weight_quant
        )
        self.layer3 = self._make_layer(
            planes=64, num_blocks=num_blocks, stride=2, weight_quant=weight_quant
        )

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._make_quant_linear(
            in_features=64,
            out_features=num_classes,
            weight_bit_width=weight_bit_width,
            weight_quant=last_layer_weight_quant,
        )

        # Weight initialization
        self._initialize_weights()

    def _make_quant_identity(self, bit_width: int = 8) -> qnn.QuantIdentity:
        """Create quantized identity layer for input quantization.

        Args:
            bit_width: Bit width for activation quantization.

        Returns:
            Quantized identity layer.
        """
        return qnn.QuantIdentity(bit_width=bit_width)

    def _make_quant_conv2d(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        weight_bit_width: int = 8,
        weight_quant=CommonIntWeightPerChannelQuant,
    ) -> qnn.QuantConv2d:
        """Create quantized convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides.
            weight_bit_width: Bit width for weight quantization.

        Returns:
            Quantized convolution layer.
        """
        return qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            return_quant_tensor=True,
        )

    def _make_quant_relu(self, bit_width: int = 8) -> qnn.QuantReLU:
        """Create quantized ReLU activation.

        Args:
            bit_width: Bit width for activation quantization.

        Returns:
            Quantized ReLU layer.
        """
        return qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=True)

    def _make_quant_linear(
        self,
        in_features: int,
        out_features: int,
        weight_bit_width: int = 8,
        weight_quant=CommonIntWeightPerTensorQuant,
    ) -> qnn.QuantLinear:
        """Create quantized linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            weight_bit_width: Bit width for weight quantization.

        Returns:
            Quantized linear layer.
        """
        return qnn.QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
        )

    def _make_layer(
        self, planes: int, num_blocks: int, stride: int, weight_quant
    ) -> nn.Sequential:
        """Build a stage of quantized residual blocks.

        Args:
            planes: Number of output channels.
            num_blocks: Number of blocks in the stage.
            stride: Stride for the first block.

        Returns:
            Sequential container of residual blocks.
        """
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                self._make_quant_conv2d(
                    self.in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    weight_bit_width=self.weight_bit_width,
                    weight_quant=weight_quant,
                ),
                nn.BatchNorm2d(planes),
            )

        layers: List[nn.Module] = []
        layers.append(
            QuantBasicBlock(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.act_bit_width,
                weight_quant=weight_quant,
            )
        )
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(
                QuantBasicBlock(
                    in_planes=self.in_planes,
                    planes=planes,
                    weight_bit_width=self.weight_bit_width,
                    act_bit_width=self.act_bit_width,
                    weight_quant=weight_quant,
                )
            )

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
        """Forward pass through quantized ResNet.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def QuantResNet20(
    num_classes: int = 10,
    in_channels: int = 3,
    in_weight_bit_width: int = 8,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
    **kwargs,
) -> QuantResNetCIFAR:
    """Create quantized ResNet-20 for CIFAR.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        in_weight_bit_width: Bit width for input weight quantization.
        weight_bit_width: Bit width for weights.
        act_bit_width: Bit width for activations.
        **kwargs: Additional arguments (ignored).

    Returns:
        Quantized ResNet-20 model.
    """
    return QuantResNetCIFAR(
        num_blocks=3,
        num_classes=num_classes,
        in_channels=in_channels,
        in_weight_bit_width=in_weight_bit_width,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
    )


def QuantResNet32(
    num_classes: int = 10,
    in_channels: int = 3,
    in_weight_bit_width: int = 8,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
    **kwargs,
) -> QuantResNetCIFAR:
    """Create quantized ResNet-32 for CIFAR."""
    return QuantResNetCIFAR(
        num_blocks=5,
        num_classes=num_classes,
        in_channels=in_channels,
        in_weight_bit_width=in_weight_bit_width,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
    )


def QuantResNet44(
    num_classes: int = 10,
    in_channels: int = 3,
    in_weight_bit_width: int = 8,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
    **kwargs,
) -> QuantResNetCIFAR:
    """Create quantized ResNet-44 for CIFAR."""
    return QuantResNetCIFAR(
        num_blocks=7,
        num_classes=num_classes,
        in_channels=in_channels,
        in_weight_bit_width=in_weight_bit_width,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
    )


def QuantResNet56(
    num_classes: int = 10,
    in_channels: int = 3,
    in_weight_bit_width: int = 8,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
    **kwargs,
) -> QuantResNetCIFAR:
    """Create quantized ResNet-56 for CIFAR."""
    return QuantResNetCIFAR(
        num_blocks=9,
        num_classes=num_classes,
        in_channels=in_channels,
        in_weight_bit_width=in_weight_bit_width,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
    )


def QuantResNet110(
    num_classes: int = 10,
    in_channels: int = 3,
    in_weight_bit_width: int = 8,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
    **kwargs,
) -> QuantResNetCIFAR:
    """Create quantized ResNet-110 for CIFAR."""
    return QuantResNetCIFAR(
        num_blocks=18,
        num_classes=num_classes,
        in_channels=in_channels,
        in_weight_bit_width=in_weight_bit_width,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
    )
