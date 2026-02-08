"""MobileNetV1 architecture for ImageNet.

Implementation of MobileNetV1 with depthwise separable convolutions,
designed for ImageNet-1K (224x224) inputs.

See: https://arxiv.org/abs/1704.04861
"""

from __future__ import annotations

from typing import ClassVar, List, Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise Separable Convolution Block.

    Consists of a depthwise convolution (per-channel spatial filtering)
    followed by a pointwise convolution (1x1 cross-channel mixing).
    This reduces computation compared to standard convolutions.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        """Initialize depthwise separable block.

        Args:
            in_planes: Number of input channels.
            out_planes: Number of output channels.
            stride: Stride for the depthwise convolution.
        """
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=in_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_planes,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=in_planes),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depthwise separable block.

        Args:
            x: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H', W').
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MobileNetV1(nn.Module):
    """MobileNetV1 for ImageNet-1K.

    Original MobileNetV1 architecture designed for ImageNet (224x224) inputs.
    Uses depthwise separable convolutions for efficiency.

    Architecture:
        - Initial 3x3 conv stride=2 -> 32 channels
        - 27 depthwise separable layers organized in stages
        - Downsampling at stages 2, 3, 4, 5
        - Global average pooling
        - Fully connected classifier

    Parameters: ~4.2M
    """

    #: Architecture configuration: list of (out_channels, stride) tuples
    #: Each tuple represents a depthwise separable block
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

    def __init__(self, num_classes: int = 1000, in_channels: int = 3):
        """Initialize MobileNetV1.

        Args:
            num_classes: Number of output classes (1000 for ImageNet-1K).
            in_channels: Number of input channels (3 for RGB).
        """
        super().__init__()

        self.in_channels: int = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            self._make_layers(in_planes=32),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        """Build the sequence of depthwise separable blocks.

        Args:
            in_planes: Number of input channels to the first block.

        Returns:
            Sequential container of depthwise separable blocks.
        """
        layers: List[nn.Module] = []
        for out_planes, stride in self.CFG:
            layers.append(DepthwiseSeparableBlock(in_planes=in_planes, out_planes=out_planes, stride=stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MobileNetV1.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
