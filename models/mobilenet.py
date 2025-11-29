"""!
@file models/mobilenet.py
@brief MobileNetV1 architecture adapted for small images.

@details Implementation of MobileNetV1 with depthwise separable convolutions,
adapted for CIFAR-10/100 (32x32) and MNIST (28x28) inputs.

@see https://arxiv.org/abs/1704.04861
"""

from __future__ import annotations

from typing import ClassVar, List, Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """!
    @brief Depthwise Separable Convolution Block.
    
    @details Consists of a depthwise convolution (per-channel spatial filtering)
    followed by a pointwise convolution (1x1 cross-channel mixing).
    This reduces computation compared to standard convolutions.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        """!
        @brief Initialize depthwise separable block.
        
        @param in_planes Number of input channels
        @param out_planes Number of output channels
        @param stride Stride for the depthwise convolution
        """
        super().__init__()
        # Depthwise convolution
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes)

        # Pointwise convolution
        self.conv2 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass through depthwise separable block.
        
        @param x Input tensor of shape (N, C_in, H, W)
        @return Output tensor of shape (N, C_out, H', W')
        """
        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Module):
    """!
    @brief MobileNetV1 adapted for small images.
    
    @details Modified from the original ImageNet architecture:
    - Initial stride is 1 (vs 2) to preserve resolution for 32x32 inputs
    - Fewer downsampling operations to prevent features from becoming too small
    - Supports both RGB (3-channel) and grayscale (1-channel) inputs
    
    @par Architecture
    - Initial 3x3 conv -> 32 channels
    - 13 depthwise separable blocks
    - Global average pooling
    - Fully connected classifier
    """

    ## @var CFG
    #  @brief Architecture configuration: list of (out_channels, stride) tuples
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

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        """!
        @brief Initialize MobileNetV1.
        
        @param num_classes Number of output classes
        @param in_channels Number of input channels (3 for RGB, 1 for grayscale)
        """
        super().__init__()

        self.in_channels: int = in_channels

        # Initial Conv Layer
        # NOTE: Stride is 1 here (vs 2 in ImageNet) to preserve resolution for 32x32 inputs
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # MobileNet Body
        self.layers = self._make_layers(in_planes=32)

        # Classifier
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        """!
        @brief Build the sequence of depthwise separable blocks.
        
        @param in_planes Number of input channels to the first block
        @return Sequential container of depthwise separable blocks
        """
        layers: List[nn.Module] = []
        for out_planes, stride in self.CFG:
            layers.append(DepthwiseSeparableBlock(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass through MobileNetV1.
        
        @param x Input tensor of shape (N, C, H, W)
        @return Output logits of shape (N, num_classes)
        """
        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = torch.nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
