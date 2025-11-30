"""!
@file models/cnv.py
@brief CNV (Compact Neural Vision) model adapted for small images.

@details Implementation of the CNV architecture, a compact convolutional network
designed for efficient inference. Originally from AMD/Xilinx Brevitas examples.

@see https://github.com/Xilinx/brevitas
"""

from __future__ import annotations

from typing import ClassVar, List, Tuple

import torch
import torch.nn as nn
from utils.tensor_norm import TensorNorm


## @var CNV_OUT_CH_POOL
#  @brief Configuration for conv layers: (output_channels, use_pooling)
CNV_OUT_CH_POOL: List[Tuple[int, bool]] = [
    (64, True),
    (128, False),
    (128, True),
    (256, False),
    (256, False),
]

## @var INTERMEDIATE_FC_FEATURES
#  @brief Configuration for intermediate FC layers: (in_features, out_features)
INTERMEDIATE_FC_FEATURES: List[Tuple[int, int]] = [
    (256, 512),
    (512, 512),
]

## @var LAST_FC_IN_FEATURES
#  @brief Input features for the final classification layer
LAST_FC_IN_FEATURES: int = 512

## @var POOL_SIZE
#  @brief Max pooling kernel size
POOL_SIZE: int = 2

## @var KERNEL_SIZE
#  @brief Convolution kernel size
KERNEL_SIZE: int = 3


class CNV(nn.Module):
    """!
    @brief CNV (Compact Neural Vision) model.

    @details A compact convolutional neural network with:
    - 6 convolutional layers with batch normalization and ReLU
    - 2 max pooling layers
    - 2 intermediate fully connected layers
    - Final classification layer

    @par Architecture
    - Conv layers: 64 -> 64 -> 128 -> 128 -> 256 -> 256 channels
    - FC layers: 256 -> 512 -> 512 -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        """!
        @brief Initialize CNV model.

        @param num_classes Number of output classes
        @param in_channels Number of input channels (3=RGB, 1=grayscale)
        """
        super().__init__()

        self.conv_features = nn.ModuleList()
        self.linear_features = nn.ModuleList()

        # Initial conv layer
        self.conv_features.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=KERNEL_SIZE,
                bias=False,
            )
        )
        self.conv_features.append(nn.BatchNorm2d(64, eps=1e-4))
        self.conv_features.append(nn.ReLU())

        # Build remaining conv layers based on configuration
        current_channels: int = 64
        for out_channels, use_pooling in CNV_OUT_CH_POOL:
            self.conv_features.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=KERNEL_SIZE,
                    bias=False,
                )
            )
            current_channels = out_channels
            self.conv_features.append(nn.BatchNorm2d(current_channels, eps=1e-4))
            self.conv_features.append(nn.ReLU())

            if use_pooling:
                self.conv_features.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

        # Build intermediate FC layers
        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                nn.Linear(
                    in_features=in_features, out_features=out_features, bias=False
                )
            )
            self.linear_features.append(nn.BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(nn.ReLU())

        # Final classification layer
        self.linear_features.append(
            nn.Linear(
                in_features=LAST_FC_IN_FEATURES, out_features=num_classes, bias=False
            )
        )
        self.linear_features.append(TensorNorm())

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """!
        @brief Initialize weights using uniform distribution.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass through CNV network.

        @param x Input tensor of shape (N, C, H, W)
        @return Output logits of shape (N, num_classes)
        """
        for mod in self.conv_features:
            x = mod(x)

        x = x.view(x.shape[0], -1)

        for mod in self.linear_features:
            x = mod(x)

        return x
