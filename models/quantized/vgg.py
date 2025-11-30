"""!
@file models/quantized/vgg.py
@brief Quantized VGG models using Brevitas.

@details Implements quantized versions of VGG-11, VGG-13, VGG-16, and VGG-19
with configurable bit widths for weights and activations.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn as nn
import brevitas.nn as qnn


## @var CFG
#  @brief VGG layer configurations. Numbers denote output channels, 'M' denotes max pooling.
CFG: Dict[str, List[Union[int, str]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
        512, 512, 512, "M", 512, 512, 512, "M",
    ],
    "vgg19": [
        64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
        512, 512, 512, 512, "M", 512, 512, 512, 512, "M",
    ],
}


class QuantVGG(nn.Module):
    """!
    @brief Quantized VGG model adapted for small images.
    
    @details Quantized version with configurable bit widths for
    weights and activations.
    """

    def __init__(
        self,
        cfg: List[Union[int, str]],
        num_classes: int = 10,
        in_channels: int = 3,
        batch_norm: bool = True,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        """!
        @brief Initialize quantized VGG model.
        
        @param cfg Layer configuration list
        @param num_classes Number of output classes
        @param in_channels Number of input channels
        @param batch_norm Whether to use batch normalization
        @param weight_bit_width Bit width for weights
        @param act_bit_width Bit width for activations
        """
        super().__init__()
        self.in_channels = in_channels
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width

        # Input quantization
        self.quant_inp = self._make_quant_identity(act_bit_width)

        self.features = self._make_layers(cfg, in_channels, batch_norm)

        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Quantized classifier
        self.classifier = nn.Sequential(
            self._make_quant_linear(512, 512, weight_bit_width=weight_bit_width),
            self._make_quant_relu(act_bit_width, return_quant_tensor=False),
            nn.Dropout(0.5),
            self._make_quant_linear(512, 512, weight_bit_width=weight_bit_width),
            self._make_quant_relu(act_bit_width, return_quant_tensor=False),
            nn.Dropout(0.5),
            self._make_quant_linear(512, num_classes, weight_bit_width=weight_bit_width),
        )

        # Weight initialization
        self._initialize_weights()

    def _make_layers(
        self,
        cfg: List[Union[int, str]],
        in_channels: int,
        batch_norm: bool,
    ) -> nn.Sequential:
        """!
        @brief Build the quantized feature extraction layers.
        
        @param cfg Layer configuration list
        @param in_channels Number of input channels
        @param batch_norm Whether to include batch normalization
        @return Sequential container of layers
        """
        layers: List[nn.Module] = []
        current_channels: int = in_channels

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(v, int):
                conv = self._make_quant_conv2d(
                    current_channels, v, kernel_size=3, padding=1,
                    weight_bit_width=self.weight_bit_width,
                )
                if batch_norm:
                    layers.extend([
                        conv,
                        nn.BatchNorm2d(v),
                        self._make_quant_relu(self.act_bit_width),
                    ])
                else:
                    layers.extend([
                        conv,
                        self._make_quant_relu(self.act_bit_width),
                    ])
                current_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """!
        @brief Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, qnn.QuantLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!
        @brief Forward pass through quantized VGG network.
        
        @param x Input tensor of shape (N, C, H, W)
        @return Output logits of shape (N, num_classes)
        """
        out = self.quant_inp(x)
        out = self.features(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def QuantVGG11(
    num_classes: int = 10,
    in_channels: int = 3,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
) -> QuantVGG:
    """!
    @brief Create quantized VGG-11 model with batch normalization.
    """
    return QuantVGG(
        CFG["vgg11"], num_classes, in_channels, batch_norm=True,
        weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,
    )


def QuantVGG13(
    num_classes: int = 10,
    in_channels: int = 3,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
) -> QuantVGG:
    """!
    @brief Create quantized VGG-13 model with batch normalization.
    """
    return QuantVGG(
        CFG["vgg13"], num_classes, in_channels, batch_norm=True,
        weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,
    )


def QuantVGG16(
    num_classes: int = 10,
    in_channels: int = 3,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
) -> QuantVGG:
    """!
    @brief Create quantized VGG-16 model with batch normalization.
    """
    return QuantVGG(
        CFG["vgg16"], num_classes, in_channels, batch_norm=True,
        weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,
    )


def QuantVGG19(
    num_classes: int = 10,
    in_channels: int = 3,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
) -> QuantVGG:
    """!
    @brief Create quantized VGG-19 model with batch normalization.
    """
    return QuantVGG(
        CFG["vgg19"], num_classes, in_channels, batch_norm=True,
        weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,
    )
