# Adding a New Model

This guide explains how to add a custom neural network model to the framework.

## Overview

The framework uses a registry pattern for models. To add a new model:

1. Create a model class extending `nn.Module`
2. Implement the required interface (`__init__` and `forward`)
3. Register the model in the registry

## Step-by-Step Guide

### 1. Create the Model Class

Create a new file in the `models/` directory (e.g., `models/my_model.py`):

```python
"""My custom model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class MyModel(nn.Module):
    """My custom neural network model.
    
    Describe your model architecture, design choices, and use cases.
    
    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels (3=RGB, 1=grayscale).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        """Initialize the model.
        
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
        """
        super().__init__()
        
        # Define your layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # ... more layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Output logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
```

### 2. Required Interface

Every model must:

1. **Extend `nn.Module`**
2. **Accept these constructor arguments**:
   - `num_classes: int` - Number of output classes
   - `in_channels: int` - Number of input channels
3. **Return logits** (not probabilities) from `forward()`

### 3. Register the Model

Add your model to the registry in `models/__init__.py`:

```python
from .my_model import MyModel

MODELS: Dict[str, ModelType] = {
    # Existing models...
    "resnet20": ResNet20,
    "vgg16": VGG16,
    # Add your model here
    "my_model": MyModel,
}
```

### 4. Update Exports (Optional)

For clean imports, add to the imports at the top:

```python
from .my_model import MyModel
```

## Complete Example: Simple CNN

Here's a complete example of a simple CNN suitable for CIFAR-10:

```python
"""Simple CNN model for small images."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for image classification.
    
    A straightforward convolutional network with:
    - 3 convolutional blocks with batch normalization
    - Global average pooling
    - Single fully connected layer
    
    Suitable for 32x32 images (CIFAR-10, CIFAR-100).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        """Initialize SimpleCNN.
        
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
        """
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Output logits of shape (N, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

## Adding Model Variants

For models with multiple variants (like ResNet-18/34/50), use a factory pattern:

```python
"""MyNet model family."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


class MyNetBase(nn.Module):
    """Base class for MyNet variants."""

    def __init__(
        self,
        layers: List[int],
        num_classes: int = 10,
        in_channels: int = 3,
    ):
        super().__init__()
        self.in_planes = 64
        
        # Build network based on layer configuration
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1):
        # Implementation details...
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Factory functions for variants
def MyNetSmall(num_classes: int = 10, in_channels: int = 3) -> MyNetBase:
    """Small variant with [2, 2, 2] layers."""
    return MyNetBase([2, 2, 2], num_classes, in_channels)


def MyNetMedium(num_classes: int = 10, in_channels: int = 3) -> MyNetBase:
    """Medium variant with [3, 4, 3] layers."""
    return MyNetBase([3, 4, 3], num_classes, in_channels)


def MyNetLarge(num_classes: int = 10, in_channels: int = 3) -> MyNetBase:
    """Large variant with [4, 6, 4] layers."""
    return MyNetBase([4, 6, 4], num_classes, in_channels)
```

Register all variants:

```python
MODELS: Dict[str, ModelType] = {
    # ...
    "mynet_small": MyNetSmall,
    "mynet_medium": MyNetMedium,
    "mynet_large": MyNetLarge,
}
```

## Adding a Quantized Version

To add a quantized version using Brevitas:

```python
"""Quantized version of MyModel using Brevitas."""

from __future__ import annotations

import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat


class QuantMyModel(nn.Module):
    """Quantized version of MyModel.
    
    Uses Brevitas for quantization-aware training.
    
    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels.
        weight_bit_width: Bit width for weights.
        act_bit_width: Bit width for activations.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
    ):
        super().__init__()
        
        # Quantized input
        self.quant_inp = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )
        
        # Quantized conv layers
        self.conv1 = qnn.QuantConv2d(
            in_channels, 64,
            kernel_size=3, padding=1, bias=False,
            weight_bit_width=weight_bit_width,
            weight_quant=Int8WeightPerTensorFloat,
        )
        
        self.bn1 = nn.BatchNorm2d(64)
        
        self.relu1 = qnn.QuantReLU(
            act_quant=Int8ActPerTensorFloat,
            bit_width=act_bit_width,
            return_quant_tensor=True,
        )
        
        # ... more layers ...
        
        self.fc = qnn.QuantLinear(
            64, num_classes,
            bias=True,
            weight_bit_width=weight_bit_width,
            weight_quant=Int8WeightPerTensorFloat,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # ... more layers ...
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

Register in `QUANT_MODELS`:

```python
QUANT_MODELS: Dict[str, ModelType] = {
    # ...
    "quant_my_model": QuantMyModel,
}
```

## Using Your Model

Once registered, use your model in config files:

```yaml
model:
  name: "my_model"
  # num_classes and in_channels are auto-detected from dataset

dataset:
  name: "cifar10"
```

Or with explicit parameters:

```yaml
model:
  name: "my_model"
  num_classes: 100
  in_channels: 1
```

For quantized models:

```yaml
model:
  name: "quant_my_model"

quantization:
  weight_bit_width: 4
  act_bit_width: 4
```

## Best Practices

1. **Weight Initialization**: Always initialize weights properly for faster convergence.

2. **Batch Normalization**: Use batch norm after convolutions for stable training.

3. **Global Average Pooling**: Prefer `AdaptiveAvgPool2d` over fixed-size pooling for flexibility.

4. **Type Hints**: Use type hints for better code documentation and IDE support.

5. **Docstrings**: Document your model architecture and design choices.

6. **Test Your Model**: Verify the model works correctly:
   ```python
   from models import get_model
   
   config = {
       "model": {"name": "my_model", "num_classes": 10, "in_channels": 3}
   }
   model = get_model(config)
   
   # Test forward pass
   x = torch.randn(2, 3, 32, 32)
   y = model(x)
   print(f"Output shape: {y.shape}")  # Should be (2, 10)
   
   # Count parameters
   params = sum(p.numel() for p in model.parameters())
   print(f"Parameters: {params:,}")
   ```

## Tips

- **Start Simple**: Begin with a working model and iterate.
- **Check Dimensions**: Print tensor shapes during development to catch size mismatches.
- **Use Existing Models**: Look at `models/cnv.py` or `models/resnet_cifar/` for reference implementations.
- **Handle Different Image Sizes**: Use adaptive pooling to support various input resolutions.
