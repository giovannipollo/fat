"""ResNet base classes and building blocks for classical ImageNet architectures.

Provides BasicBlock, Bottleneck, and ResNetBase classes for building
ResNet-18 through ResNet-152 models following the original paper architecture.

See: https://arxiv.org/abs/1512.03385 "Deep Residual Learning for Image Recognition"
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, List, Optional, Type, Union

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18 and ResNet-34.

    Two 3x3 convolutions with batch normalization and skip connection.
    No channel expansion (expansion = 1).
    """

    #: Output channel expansion factor
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """Initialize basic residual block.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for first convolution.
            downsample: Optional downsampling layer for skip connection.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride: int = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
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


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50, ResNet-101, and ResNet-152.

    Three convolutions (1x1 -> 3x3 -> 1x1) with channel expansion.
    The 1x1 convolutions reduce and restore dimensions for efficiency.
    """

    #: Output channel expansion factor (4x for bottleneck)
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """Initialize bottleneck residual block.

        Args:
            in_planes: Number of input channels.
            planes: Number of intermediate channels (output = planes * 4).
            stride: Stride for 3x3 convolution.
            downsample: Optional downsampling layer for skip connection.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(num_features=planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride: int = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        identity: torch.Tensor = x

        out: torch.Tensor = self.conv1(x)
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


BlockType = Union[Type[BasicBlock], Type[Bottleneck]]


class ResNetBase(nn.Module, ABC):
    """Classical ImageNet ResNet base model.

    Follows the original ResNet architecture from He et al., 2015:
    - Initial 7x7 conv with stride=2, padding=3
    - 3x3 max pooling with stride=2, padding=1
    - Four residual stages: [64, 128, 256, 512] base channels
    - Global average pooling + fully connected classifier
    - Designed for 224x224 RGB inputs

    Architecture:
        - Input: (N, 3, 224, 224)
        - Conv1: 7x7, 64 channels, stride=2 -> (N, 64, 112, 112)
        - MaxPool: 3x3, stride=2 -> (N, 64, 56, 56)
        - Layer1: 64 channels -> (N, 64*expansion, 56, 56)
        - Layer2: 128 channels, stride=2 -> (N, 128*expansion, 28, 28)
        - Layer3: 256 channels, stride=2 -> (N, 256*expansion, 14, 14)
        - Layer4: 512 channels, stride=2 -> (N, 512*expansion, 7, 7)
        - AvgPool: (7, 7) -> (1, 1) -> (N, 512*expansion, 1, 1)
        - FC: num_classes -> (N, num_classes)

    Subclasses must define:
        - block: The block type (BasicBlock or Bottleneck)
        - layers: List of layer counts [layer1, layer2, layer3, layer4]
    """

    #: Block type to use (BasicBlock or Bottleneck)
    block: ClassVar[BlockType]

    #: Number of blocks in each of the 4 stages
    layers: ClassVar[List[int]]

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
    ):
        """Initialize classical ImageNet ResNet model.

        Args:
            num_classes: Number of output classes (default: 1000 for ImageNet).
            in_channels: Number of input channels (3=RGB, 1=grayscale).
        """
        super().__init__()
        self.in_planes: int = 64
        self.in_channels: int = in_channels

        # Initial convolution layer (7x7 conv with stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling layer (3x3 with stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(planes=64, num_blocks=self.layers[0], stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=self.layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=512 * self.block.expansion, out_features=num_classes
        )

        # Weight initialization
        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a stage of residual blocks.

        Args:
            planes: Base number of output channels.
            num_blocks: Number of blocks in the stage.
            stride: Stride for the first block (downsampling).

        Returns:
            Sequential container of residual blocks.
        """
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_planes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_planes,
                    out_channels=planes * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=planes * self.block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(
            self.block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * self.block.expansion
        for _ in range(1, num_blocks):
            layers.append(self.block(in_planes=self.in_planes, planes=planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet.

        Args:
            x: Input tensor of shape (N, C, H, W).
               Typically (N, 3, 224, 224) for ImageNet.

        Returns:
            Output logits of shape (N, num_classes).
        """
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
