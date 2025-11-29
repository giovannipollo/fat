"""
VGG models adapted for CIFAR-10/100 (32x32 images) and MNIST (28x28 images).

Reference:
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
"""

import torch
import torch.nn as nn
from typing import List, Union


# VGG configurations
# Numbers denote output channels, 'M' denotes max pooling
CFG = {
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
    """
    VGG model adapted for CIFAR-10/100 (32x32 images) and MNIST (28x28 images).

    The architecture is modified from the original ImageNet version:
    - Uses smaller fully connected layers (512 instead of 4096) for efficiency
    - Uses adaptive average pooling to handle different input sizes
    - Supports both RGB (3-channel) and grayscale (1-channel) inputs
    """

    def __init__(
        self,
        cfg: List[Union[int, str]],
        num_classes: int = 10,
        in_channels: int = 3,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels

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
        layers: List[nn.Module] = []
        current_channels = in_channels

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

    def _initialize_weights(self):
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
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def VGG11(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """VGG-11 model with batch normalization."""
    return VGG(CFG["vgg11"], num_classes, in_channels, batch_norm=True)


def VGG13(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """VGG-13 model with batch normalization."""
    return VGG(CFG["vgg13"], num_classes, in_channels, batch_norm=True)


def VGG16(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """VGG-16 model with batch normalization."""
    return VGG(CFG["vgg16"], num_classes, in_channels, batch_norm=True)


def VGG19(num_classes: int = 10, in_channels: int = 3) -> VGG:
    """VGG-19 model with batch normalization."""
    return VGG(CFG["vgg19"], num_classes, in_channels, batch_norm=True)
