import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise Separable Convolution Block (Depthwise conv + Pointwise conv)"""

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
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

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Module):
    """
    MobileNetV1 adapted for CIFAR-10/100 (32x32 images) and MNIST (28x28 images).

    The architecture is modified from the original ImageNet version:
    - Initial stride is 1 (vs 2 in ImageNet) to preserve resolution for 32x32 inputs
    - Fewer downsampling operations to prevent features from becoming too small
    - Supports both RGB (3-channel) and grayscale (1-channel) inputs
    """

    # Architecture config: (out_channels, stride)
    CFG = [
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
        super().__init__()

        self.in_channels = in_channels

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

    def _make_layers(self, in_planes: int):
        layers = []
        for out_planes, stride in self.CFG:
            layers.append(DepthwiseSeparableBlock(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = torch.nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
