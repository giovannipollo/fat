"""Quantized MobileNetV1 for hardware deployment using Brevitas.

Modified from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models

MIT License

Copyright (c) 2019 Xilinx, Inc (Alessandro Pappalardo)
Copyright (c) 2018 Oleg Sémery

Implements quantized MobileNetV1 with per-channel weight quantization
and per-channel activation scaling, optimized for hardware deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import IntBias

from utils.weight_quant import CommonIntWeightPerChannelQuant
from utils.weight_quant import CommonIntWeightPerTensorQuant
from utils.weight_quant import CommonUintActQuant


@dataclass
class DatasetConfig:
    """Configuration parameters that vary by dataset/input size.

    Attributes:
        avg_pool_kernel_size: Kernel size for final average pooling.
        first_layer_stride: Stride for the initial convolution.
        first_layer_padding: Padding for the initial convolution.
        stage_config: List of stages, each containing (out_channels, stride) tuples.
    """

    avg_pool_kernel_size: int
    first_layer_stride: int
    first_layer_padding: int
    stage_config: List[List[Tuple[int, int]]]


#: Number of channels in the initial convolution block
INIT_CHANNELS: int = 32

#: Final number of channels (output of last stage)
FINAL_CHANNELS: int = 1024

#: Stage configuration for ImageNet-sized inputs (224x224).
#: Downsampling at first unit of stages 2-5.
IMAGENET_STAGE_CONFIG: List[List[Tuple[int, int]]] = [
    # Stage 1: 32 -> 64 (no downsampling)
    [(64, 1)],
    # Stage 2: 64 -> 128 -> 128, downsample at first unit
    [(128, 2), (128, 1)],
    # Stage 3: 128 -> 256 -> 256, downsample at first unit
    [(256, 2), (256, 1)],
    # Stage 4: 256 -> 512 (x6), downsample at first unit
    [(512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1)],
    # Stage 5: 512 -> 1024 -> 1024, downsample at first unit
    [(1024, 2), (1024, 1)],
]

#: Stage configuration for ImageNet with first stage stride.
#: Downsampling at first unit of all stages.
IMAGENET_FIRST_STRIDE_STAGE_CONFIG: List[List[Tuple[int, int]]] = [
    # Stage 1: 32 -> 64, downsample at first unit
    [(64, 2)],
    # Stage 2: 64 -> 128 -> 128, downsample at first unit
    [(128, 2), (128, 1)],
    # Stage 3: 128 -> 256 -> 256, downsample at first unit
    [(256, 2), (256, 1)],
    # Stage 4: 256 -> 512 (x6), downsample at first unit
    [(512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1)],
    # Stage 5: 512 -> 1024 -> 1024, downsample at first unit
    [(1024, 2), (1024, 1)],
]


def get_dataset_config(first_stage_stride: bool = False) -> DatasetConfig:
    """Get the ImageNet configuration.

    Args:
        first_stage_stride: Whether to use stride=2 in first stage.

    Returns:
        DatasetConfig with appropriate parameters.
    """
    if first_stage_stride:
        stage_config = IMAGENET_FIRST_STRIDE_STAGE_CONFIG
    else:
        stage_config = IMAGENET_STAGE_CONFIG

    return DatasetConfig(
        avg_pool_kernel_size=7,
        first_layer_stride=2,
        first_layer_padding=0,
        stage_config=stage_config,
    )


class QuantConvBlock(nn.Module):
    """Quantized convolution block with batch normalization and ReLU.

    Uses per-channel weight quantization and configurable per-channel
    activation scaling for hardware deployment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weight_bit_width: int,
        act_bit_width: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bn_eps: float = 1e-5,
        weight_quant=CommonIntWeightPerChannelQuant,
        activation_scaling_per_channel: bool = False,
    ):
        """Initialize quantized convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            stride: Stride of the convolution.
            padding: Padding added to input.
            groups: Number of groups for grouped convolution.
            bn_eps: Epsilon for batch normalization.
            weight_quant: Weight quantizer class.
            activation_scaling_per_channel: Whether to use per-channel activation scaling.
        """
        super().__init__()

        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=weight_quant,
            weight_bit_width=weight_bit_width,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = qnn.QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_stats_permute_dims=(1, 0, 2, 3),
            scaling_per_output_channel=activation_scaling_per_channel,
            return_quant_tensor=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized conv block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class QuantDwsConvBlock(nn.Module):
    """Quantized Depthwise Separable Convolution Block for hardware.

    Consists of a depthwise convolution followed by a pointwise
    convolution, both quantized with per-channel weight quantization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        act_bit_width: int,
        weight_bit_width: int,
        weight_quant=CommonIntWeightPerChannelQuant,
        pw_activation_scaling_per_channel: bool = False,
    ):
        """Initialize quantized depthwise separable block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the depthwise convolution.
            act_bit_width: Bit width for activation quantization.
            weight_bit_width: Bit width for weight quantization.
            weight_quant: Weight quantizer class.
            pw_activation_scaling_per_channel: Whether to use per-channel
                activation scaling for pointwise convolution.
        """
        super().__init__()

        self.dw_conv = QuantConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            act_bit_width=act_bit_width,
        )

        self.pw_conv = QuantConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            act_bit_width=act_bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized depthwise separable block."""
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class QuantMobileNetV1(nn.Module):
    """Quantized MobileNetV1 optimized for hardware deployment.

    Uses per-channel weight quantization (CommonIntWeightPerChannelQuant)
    and per-channel activation scaling for better hardware efficiency.
    First and last layers use higher bit widths for accuracy.

    Architecture:
        - Init block: in_channels -> 32 channels
        - Stage 1: 32 -> 64 channels
        - Stage 2: 64 -> 128 -> 128 channels
        - Stage 3: 128 -> 256 -> 256 channels
        - Stage 4: 256 -> 512 (x6) channels
        - Stage 5: 512 -> 1024 -> 1024 channels
        - Classifier: 1024 -> num_classes

    Note:
        Stride patterns differ by dataset:
        - CIFAR (32x32): Downsample at stage 4 last unit and stage 5 second unit
        - ImageNet (224x224): Downsample at first unit of each stage
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        dataset: str = "cifar10",
        first_layer_bit_width: int = 8,
        last_layer_bit_width: int = 8,
        first_stage_stride: bool = False,
        round_average_pool: bool = False,
    ):
        """Initialize quantized MobileNetV1 for hardware.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            first_layer_bit_width: Bit width for first layer weights.
            last_layer_bit_width: Bit width for last layer weights.
            first_stage_stride: For ImageNet, whether to use stride=2 in first stage.
            round_average_pool: Whether to use ROUND (True) or FLOOR (False)
                for average pooling quantization.
        """
        super().__init__()

        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        # Get dataset-specific configuration
        config = get_dataset_config(first_stage_stride)

        self.features = self._build_features(
            in_channels=in_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            first_layer_bit_width=first_layer_bit_width,
            config=config,
        )

        self.final_pool = self._build_avg_pool(
            act_bit_width=act_bit_width,
            kernel_size=config.avg_pool_kernel_size,
            round_average_pool=round_average_pool,
        )

        self.classifier = self._build_classifier(
            num_classes=num_classes,
            last_layer_bit_width=last_layer_bit_width,
        )

    def _build_features(
        self,
        in_channels: int,
        weight_bit_width: int,
        act_bit_width: int,
        first_layer_bit_width: int,
        config: DatasetConfig,
    ) -> nn.Sequential:
        """Build the feature extraction layers.

        Args:
            in_channels: Number of input channels.
            weight_bit_width: Bit width for weight quantization.
            act_bit_width: Bit width for activation quantization.
            first_layer_bit_width: Bit width for first layer weights.
            config: Dataset-specific configuration.

        Returns:
            Sequential container with all feature extraction layers.
        """
        features = nn.Sequential()

        # Initial convolution block
        init_block = QuantConvBlock(
            in_channels=in_channels,
            out_channels=INIT_CHANNELS,
            kernel_size=3,
            stride=config.first_layer_stride,
            padding=config.first_layer_padding,
            weight_bit_width=first_layer_bit_width,
            weight_quant=CommonIntWeightPerChannelQuant,
            act_bit_width=act_bit_width,
            activation_scaling_per_channel=True,
        )
        features.add_module("init_block", init_block)

        # Build depthwise separable stages
        current_channels = INIT_CHANNELS
        num_stages = len(config.stage_config)

        for stage_idx, stage_units in enumerate(config.stage_config):
            stage = nn.Sequential()
            if stage_idx == num_stages - 1:
                use_per_channel_act_scaling = False
            else:
                use_per_channel_act_scaling = True

            for unit_idx, (out_channels, stride) in enumerate(stage_units):
                block = QuantDwsConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=stride,
                    act_bit_width=act_bit_width,
                    weight_bit_width=weight_bit_width,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    pw_activation_scaling_per_channel=use_per_channel_act_scaling,
                )
                stage.add_module(f"unit{unit_idx + 1}", block)
                current_channels = out_channels

            features.add_module(f"stage{stage_idx + 1}", stage)

        return features

    def _build_avg_pool(
        self,
        act_bit_width: int,
        kernel_size: int,
        round_average_pool: bool,
    ) -> qnn.QuantAvgPool2d:
        """Build the quantized average pooling layer.

        Args:
            act_bit_width: Bit width for activation quantization.
            kernel_size: Kernel size for average pooling.
            round_average_pool: Whether to use ROUND or FLOOR for quantization.

        Returns:
            Configured QuantAvgPool2d layer.
        """
        # Use 4-bit for 2-bit activations to avoid overflow
        if act_bit_width == 2:
            pool_bit_width = 4
        else:
            pool_bit_width = act_bit_width

        if round_average_pool:
            float_to_int_impl = "ROUND"
        else:
            float_to_int_impl = "FLOOR"

        return qnn.QuantAvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            bit_width=pool_bit_width,
            float_to_int_impl_type=float_to_int_impl,
        )

    def _build_classifier(
        self,
        num_classes: int,
        last_layer_bit_width: int,
    ) -> qnn.QuantLinear:
        """Build the quantized classifier layer.

        Args:
            num_classes: Number of output classes.
            last_layer_bit_width: Bit width for classifier weights.

        Returns:
            Configured QuantLinear classifier layer.
        """
        # Final channels from the last stage (1024)

        return qnn.QuantLinear(
            FINAL_CHANNELS,
            num_classes,
            bias=True,
            bias_quant=IntBias,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=last_layer_bit_width,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized MobileNetV1.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
