"""Quantized ResNet models for ImageNet-style architectures using Brevitas.

Implements quantized versions of ResNet-18 through ResNet-152
with configurable bit widths for weights and activations.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, List, Optional, Type, Union

import torch
import torch.nn as nn
import brevitas.nn as qnn


class QuantBasicBlock(nn.Module):
    """Quantized basic residual block for ResNet-18 and ResNet-34."""

    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """Initialize quantized basic residual block.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for first convolution.
            downsample: Optional downsampling layer.
            weight_bit_width: Bit width for weights.
            act_bit_width: Bit width for activations.
        """
        super().__init__()
        self.conv1 = self._make_quant_conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = self._make_quant_relu(act_bit_width)

        self.conv2 = self._make_quant_conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = self._make_quant_relu(act_bit_width)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
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


class QuantBottleneck(nn.Module):
    """Quantized bottleneck residual block for ResNet-50, 101, 152."""

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """Initialize quantized bottleneck residual block."""
        super().__init__()
        self.conv1 = self._make_quant_conv2d(
            in_planes, planes, kernel_size=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = self._make_quant_relu(act_bit_width)

        self.conv2 = self._make_quant_conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = self._make_quant_relu(act_bit_width)

        self.conv3 = self._make_quant_conv2d(
            planes, planes * self.expansion, kernel_size=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = self._make_quant_relu(act_bit_width)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu3(out)

        return out


BlockType = Union[Type[QuantBasicBlock], Type[QuantBottleneck]]


class QuantResNetBase(nn.Module, ABC):
    """Quantized base ResNet model adapted for small images."""

    block: ClassVar[BlockType]
    layers: ClassVar[List[int]]

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        **kwargs,
    ):
        """Initialize quantized ResNet base model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            weight_bit_width: Bit width for weights.
            act_bit_width: Bit width for activations.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.in_planes: int = 64
        self.in_channels = in_channels
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        # Input quantization
        self.quant_inp = self._make_quant_identity(act_bit_width)

        # Initial convolution layer
        self.conv1 = self._make_quant_conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1,
            weight_bit_width=weight_bit_width,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = self._make_quant_relu(act_bit_width)

        # Residual layers
        self.layer1 = self._make_layer(64, self.layers[0], stride=1)
        self.layer2 = self._make_layer(128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(512, self.layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._make_quant_linear(
            512 * self.block.expansion, num_classes,
            weight_bit_width=weight_bit_width,
        )

        # Weight initialization
        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a stage of quantized residual blocks."""
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_planes != planes * self.block.expansion:
            downsample = nn.Sequential(
                self._make_quant_conv2d(
                    self.in_planes, planes * self.block.expansion,
                    kernel_size=1, stride=stride,
                    weight_bit_width=self.weight_bit_width,
                ),
                nn.BatchNorm2d(planes * self.block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(self.block(
            self.in_planes, planes, stride, downsample,
            self.weight_bit_width, self.act_bit_width,
        ))
        self.in_planes = planes * self.block.expansion
        for _ in range(1, num_blocks):
            layers.append(self.block(
                self.in_planes, planes,
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.act_bit_width,
            ))

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
        """Forward pass through quantized ResNet."""
        out = self.quant_inp(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class QuantResNet18(QuantResNetBase):
    """Quantized ResNet-18 model."""
    block = QuantBasicBlock
    layers = [2, 2, 2, 2]


class QuantResNet34(QuantResNetBase):
    """Quantized ResNet-34 model."""
    block = QuantBasicBlock
    layers = [3, 4, 6, 3]


class QuantResNet50(QuantResNetBase):
    """Quantized ResNet-50 model."""
    block = QuantBottleneck
    layers = [3, 4, 6, 3]


class QuantResNet101(QuantResNetBase):
    """Quantized ResNet-101 model."""
    block = QuantBottleneck
    layers = [3, 4, 23, 3]


class QuantResNet152(QuantResNetBase):
    """Quantized ResNet-152 model."""
    block = QuantBottleneck
    layers = [3, 8, 36, 3]
