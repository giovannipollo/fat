"""!
@file models/quantized/cnv.py
@brief Quantized CNV (Compact Neural Vision) model using Brevitas.

@details Implements quantized version of the CNV architecture with
configurable bit widths for weights and activations.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import brevitas.nn as qnn

from utils.weight_quant import CommonActQuant
from utils.weight_quant import CommonWeightQuant
from utils.tensor_norm import TensorNorm
from utils.weight_quant import CommonUintActQuant

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


class QuantCNV(nn.Module):
    """!
    @brief Quantized CNV (Compact Neural Vision) model.

    @details Quantized version with configurable bit widths for
    weights and activations using Brevitas.

    @par Architecture
    - Conv layers: 64 -> 64 -> 128 -> 128 -> 256 -> 256 channels
    - FC layers: 256 -> 512 -> 512 -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        in_weight_bit_width: int = 8,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """!
        @brief Initialize quantized CNV model.

        @param num_classes Number of output classes
        @param in_channels Number of input channels
        @param in_weight_bit_width Bit width for input quantization
        @param weight_bit_width Bit width for weight quantization
        @param act_bit_width Bit width for activation quantization
        """
        super().__init__()

        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        self.conv_features = nn.ModuleList()
        self.linear_features = nn.ModuleList()

        # Initial conv layer
        self.conv_features.append(
            qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=KERNEL_SIZE,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=in_weight_bit_width,
            )
        )
        self.conv_features.append(nn.BatchNorm2d(64, eps=1e-4))
        self.conv_features.append(
            qnn.QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                per_channel_broadcastable_shape=(1, 64, 1, 1),
                scaling_stats_permute_dims=(1, 0, 2, 3),
                scaling_per_output_channel=False,
                return_quant_tensor=True,
            )
        )

        # Build remaining conv layers based on configuration
        current_channels: int = 64
        for out_channels, use_pooling in CNV_OUT_CH_POOL:
            self.conv_features.append(
                qnn.QuantConv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=KERNEL_SIZE,
                    weight_bit_width=weight_bit_width,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                )
            )
            current_channels = out_channels
            self.conv_features.append(nn.BatchNorm2d(current_channels, eps=1e-4))
            self.conv_features.append(
                qnn.QuantReLU(
                    act_quant=CommonUintActQuant,
                    bit_width=act_bit_width,
                    per_channel_broadcastable_shape=(1, current_channels, 1, 1),
                    scaling_stats_permute_dims=(1, 0, 2, 3),
                    scaling_per_output_channel=False,
                    return_quant_tensor=True,
                )
            )

            if use_pooling:
                self.conv_features.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

        # Build intermediate FC layers
        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                qnn.QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonWeightQuant,
                )
            )
            self.linear_features.append(nn.BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                qnn.QuantReLU(
                    act_quant=CommonUintActQuant,
                    bit_width=act_bit_width,
                    per_channel_broadcastable_shape=(1, out_features, 1, 1),
                    scaling_stats_permute_dims=(1, 0, 2, 3),
                    scaling_per_output_channel=False,
                    return_quant_tensor=True,
                )
            )

        # Final classification layer
        self.linear_features.append(
            qnn.QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_bit_width=weight_bit_width,
                weight_quant=CommonWeightQuant,
            )
        )

        self.linear_features.append(TensorNorm())

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """!
        @brief Initialize weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d) or isinstance(m, qnn.QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, qnn.QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, qnn.QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass through quantized CNV network.

        @param x Input tensor of shape (N, C, H, W)
        @return Output logits of shape (N, num_classes)
        """

        for mod in self.conv_features:
            x = mod(x)

        x = x.view(x.shape[0], -1)

        for mod in self.linear_features:
            x = mod(x)

        return x
