"""ResNet base classes and building blocks for ImageNet-style architectures.

Provides BasicBlock, Bottleneck, and ResNetBase classes for building
ResNet-18 through ResNet-152 models adapted for small image inputs.

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
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

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
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBase(nn.Module, ABC):
    """Base ResNet model adapted for small images (CIFAR, MNIST).

    Modified from the original ImageNet architecture:
    - Initial conv: 3x3, stride=1, padding=1 (vs 7x7, stride=2)
    - No max pooling after initial conv (preserves resolution)
    - Supports both RGB and grayscale inputs

    Architecture:
        - Initial 3x3 conv -> 64 channels
        - 4 stages: [64, 128, 256, 512] base channels
        - Global average pooling + FC classifier

    Subclasses must define:
        - block: The block type (BasicBlock or Bottleneck)
        - layers: List of layer counts [layer1, layer2, layer3, layer4]
    """

    #: Block type to use (BasicBlock or Bottleneck)
    block: Type[Union[BasicBlock, Bottleneck]]

    #: Number of blocks in each of the 4 stages
    layers: List[int]

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        """Initialize ResNet base model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels (3=RGB, 1=grayscale).
        """
        super().__init__()
        self.in_planes: int = 64
        self.in_channels: int = in_channels

        # Initial convolution layer
        # NOTE: Smaller kernel and no stride/pooling for 32x32 inputs
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No max pooling for CIFAR (small images)

        # Residual layers
        self.layer1 = self._make_layer(64, self.layers[0], stride=1)
        self.layer2 = self._make_layer(128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(512, self.layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

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
                    self.in_planes,
                    planes * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(self.block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * self.block.expansion
        for _ in range(1, num_blocks):
            layers.append(self.block(self.in_planes, planes))

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

        Returns:
            Output logits of shape (N, num_classes).
        """
        out: torch.Tensor = self.conv1(x)
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
