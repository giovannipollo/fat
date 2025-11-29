"""
ResNet base classes and building blocks for ImageNet-style architectures.

Reference:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
"""

import torch
import torch.nn as nn
from typing import Type, List, Optional, Union
from abc import ABC


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18 and ResNet-34."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
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
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
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
    """Bottleneck residual block for ResNet-50, ResNet-101, and ResNet-152."""

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
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
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
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
    """
    Base ResNet model adapted for CIFAR-10/100 (32x32 images) and MNIST (28x28 images).

    The architecture is modified from the original ImageNet version:
    - Initial conv has kernel_size=3, stride=1, padding=1 (vs 7x7, stride 2 in ImageNet)
    - No max pooling after initial conv (preserves resolution for small images)
    - Supports both RGB (3-channel) and grayscale (1-channel) inputs
    
    Subclasses must define:
    - block: The block type (BasicBlock or Bottleneck)
    - layers: List of layer counts [layer1, layer2, layer3, layer4]
    """

    block: Type[Union[BasicBlock, Bottleneck]]
    layers: List[int]

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.in_planes = 64
        self.in_channels = in_channels

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
        downsample = None
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

        layers = []
        layers.append(self.block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * self.block.expansion
        for _ in range(1, num_blocks):
            layers.append(self.block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
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
