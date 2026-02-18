"""ResNet base classes for CIFAR-specific quantized architectures.

Implements the CIFAR-specific ResNet architecture from the original paper
(Section 4.2) with quantized convolutions and activations using Brevitas.

See: https://arxiv.org/abs/1512.03385 "Deep Residual Learning"
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

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

    #: Output channel expansion factor (1 for basic block)
    expansion: ClassVar[int] = 1

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
            stride: Stride for first convolution (for downsampling).
            downsample: Optional downsampling layer for skip connection.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
        """
        super().__init__()
        self.conv1 = qnn.QuantConv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            return_quant_tensor=True,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with skip connection added.
        """
        identity: torch.Tensor = x

        out: torch.Tensor = self.conv1(x)
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

    Follows the original paper's CIFAR-10 architecture with quantized layers:
    - Initial conv: 3x3, 16 filters, no pooling
    - 3 stages with filter counts: 16 -> 32 -> 64
    - Each stage has n blocks (total layers = 6n + 2)
    - Global average pooling + FC layer
    - Quantized convolutions and activations using Brevitas

    Architecture Summary:
        - Stage 1: n blocks at 16 channels (stride=1)
        - Stage 2: n blocks at 32 channels (stride=2)
        - Stage 3: n blocks at 64 channels (stride=2)
    """

    def __init__(
        self,
        num_blocks: int,
        num_classes: int = 10,
        in_channels: int = 3,
        in_weight_bit_width: int = 8,
        weight_bit_width: int = 8,
        out_weight_bit_width: int = 8,
        act_bit_width: int = 8,
        first_layer_weight_quant=CommonIntWeightPerChannelQuant,
        weight_quant=CommonIntWeightPerChannelQuant,
        last_layer_weight_quant=CommonIntWeightPerTensorQuant,
    ):
        """Initialize quantized CIFAR ResNet.

        Args:
            num_blocks: Number of blocks per stage (n in the paper).
            num_classes: Number of output classes.
            in_channels: Number of input channels (3=RGB, 1=grayscale).
            in_weight_bit_width: Bit width for input weight quantization.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            first_layer_weight_quant: Weight quantization for first layer.
            weight_quant: Weight quantization for hidden layers.
            last_layer_weight_quant: Weight quantization for last layer.
        """
        super().__init__()
        self.in_planes: int = 16
        self.in_weight_bit_width = in_weight_bit_width
        self.weight_bit_width = weight_bit_width
        self.out_weight_bit_width = out_weight_bit_width
        self.act_bit_width = act_bit_width
        self.weight_quant = weight_quant

        # Input quantization
        self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

        # Initial convolution
        self.conv1 = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            weight_bit_width=self.in_weight_bit_width,
            weight_quant=first_layer_weight_quant,
            return_quant_tensor=True,
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = qnn.QuantReLU(
            bit_width=self.act_bit_width, return_quant_tensor=True
        )

        # Three stages with increasing filter counts
        self.layer1 = self._make_layer(planes=16, num_blocks=num_blocks, stride=1)
        self.layer2 = self._make_layer(planes=32, num_blocks=num_blocks, stride=2)
        self.layer3 = self._make_layer(planes=64, num_blocks=num_blocks, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = qnn.QuantLinear(
            in_features=64,
            out_features=num_classes,
            bias=True,
            weight_bit_width=self.out_weight_bit_width,
            weight_quant=last_layer_weight_quant,
        )

        # Weight initialization
        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage of quantized residual blocks.

        Args:
            planes: Number of output channels.
            num_blocks: Number of blocks in the stage.
            stride: Stride for the first block (downsampling).

        Returns:
            Sequential container of quantized residual blocks.
        """
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                qnn.QuantConv2d(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    weight_bit_width=self.weight_bit_width,
                    weight_quant=self.weight_quant,
                    return_quant_tensor=True,
                ),
                nn.BatchNorm2d(num_features=planes),
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
                weight_quant=self.weight_quant,
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
                    weight_quant=self.weight_quant,
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
        out: torch.Tensor = self.quant_inp(x)
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
