"""VGG models adapted for small images (CIFAR, MNIST).

Implementation of VGG-11, VGG-13, VGG-16, and VGG-19 architectures
with batch normalization, adapted for 32x32 and 28x28 input sizes.

See: https://arxiv.org/abs/1409.1556 "Very Deep Convolutional Networks"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Union


#: VGG layer configurations. Numbers denote output channels, 'M' denotes max pooling.
CFG: Dict[str, List[Union[int, str]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    """VGG model adapted for small images.

    Modified from the original ImageNet architecture:
    - Uses smaller fully connected layers (512 instead of 4096)
    - Uses adaptive average pooling to handle different input sizes
    - Supports both RGB and grayscale inputs
    - Batch normalization is enabled by default
    """

    def __init__(
        self,
        cfg: List[Union[int, str]],
        num_classes: int = 10,
        in_channels: int = 3,
        batch_norm: bool = True,
    ):
        """Initialize VGG model.

        Args:
            cfg: Layer configuration list from CFG dictionary.
            num_classes: Number of output classes.
            in_channels: Number of input channels (3=RGB, 1=grayscale).
            batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        self.in_channels: int = in_channels

        self.features = self._make_layers(cfg, in_channels, batch_norm)

        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (smaller than original for CIFAR)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # Weight initialization
        self._initialize_weights()

    def _make_layers(
        self,
        cfg: List[Union[int, str]],
        in_channels: int,
        batch_norm: bool,
    ) -> nn.Sequential:
        """Build the feature extraction layers.

        Args:
            cfg: Layer configuration list.
            in_channels: Number of input channels.
            batch_norm: Whether to include batch normalization.

        Returns:
            Sequential container of convolutional layers.
        """
        layers: List[nn.Module] = []
        current_channels: int = in_channels

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(v, int):
                conv = nn.Conv2d(current_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv, nn.ReLU(inplace=True)])
                current_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VGG network.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes).
        """
        out: torch.Tensor = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def VGG11(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """Create VGG-11 model with batch normalization.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.

    Returns:
        VGG-11 model instance.
    """
    return VGG(CFG["vgg11"], num_classes, in_channels, batch_norm=True)


def VGG13(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """Create VGG-13 model with batch normalization.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.

    Returns:
        VGG-13 model instance.
    """
    return VGG(CFG["vgg13"], num_classes, in_channels, batch_norm=True)


def VGG16(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """Create VGG-16 model with batch normalization.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.

    Returns:
        VGG-16 model instance.
    """
    return VGG(CFG["vgg16"], num_classes, in_channels, batch_norm=True)


def VGG19(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """Create VGG-19 model with batch normalization.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.

    Returns:
        VGG-19 model instance.
    """
    return VGG(CFG["vgg19"], num_classes, in_channels, batch_norm=True)
