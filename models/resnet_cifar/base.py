"""!
@file models/resnet_cifar/base.py
@brief ResNet base classes for CIFAR-specific architectures.

@details Implements the CIFAR-specific ResNet architecture from the original paper
(Section 4.2), which differs from the ImageNet version in structure and size.

@see https://arxiv.org/abs/1512.03385 "Deep Residual Learning"
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """!
    @brief Basic residual block for CIFAR ResNets.
    
    @details Two 3x3 convolutions with batch normalization and skip connection.
    No channel expansion (expansion = 1).
    """

    ## @var expansion
    #  @brief Output channel expansion factor (1 for basic block)
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """!
        @brief Initialize basic residual block.
        
        @param in_planes Number of input channels
        @param planes Number of output channels
        @param stride Stride for first convolution (for downsampling)
        @param downsample Optional downsampling layer for skip connection
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass with residual connection.
        
        @param x Input tensor
        @return Output tensor with skip connection added
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


class ResNetCIFAR(nn.Module):
    """!
    @brief CIFAR-specific ResNet architecture.
    
    @details Follows the original paper's CIFAR-10 architecture:
    - Initial conv: 3x3, 16 filters, no pooling
    - 3 stages with filter counts: 16 -> 32 -> 64
    - Each stage has n blocks (total layers = 6n + 2)
    - Global average pooling + FC layer
    
    @par Architecture Summary
    - Stage 1: n blocks at 16 channels (stride=1)
    - Stage 2: n blocks at 32 channels (stride=2)
    - Stage 3: n blocks at 64 channels (stride=2)
    """

    def __init__(
        self,
        num_blocks: int,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        """!
        @brief Initialize CIFAR ResNet.
        
        @param num_blocks Number of blocks per stage (n in the paper)
        @param num_classes Number of output classes
        @param in_channels Number of input channels (3=RGB, 1=grayscale)
        """
        super().__init__()
        self.in_planes: int = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Three stages with increasing filter counts
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """!
        @brief Build a stage of residual blocks.
        
        @param planes Number of output channels
        @param num_blocks Number of blocks in the stage
        @param stride Stride for the first block (downsampling)
        @return Sequential container of residual blocks
        """
        downsample: Optional[nn.Module] = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers: List[nn.Module] = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """!
        @brief Initialize weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass through ResNet.
        
        @param x Input tensor of shape (N, C, H, W)
        @return Output logits of shape (N, num_classes)
        """
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
