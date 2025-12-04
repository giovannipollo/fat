"""Model module initialization and registry.

Provides a unified interface for creating neural network models
through a registry pattern. Includes ResNet (CIFAR and ImageNet variants),
VGG, MobileNet, CNV architectures, and their quantized versions using Brevitas.

See Also:
    get_model: Main factory function.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type, Union

import torch.nn as nn

from .standard import (
    CNV,
    MobileNetV1,
    VGG11,
    VGG13,
    VGG16,
    VGG19,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from .quantized import (
    QuantCNV,
    QuantResNet20,
    QuantResNet32,
    QuantResNet44,
    QuantResNet56,
    QuantResNet110,
    QuantResNet18,
    QuantResNet34,
    QuantResNet50,
    QuantResNet101,
    QuantResNet152,
    QuantVGG11,
    QuantVGG13,
    QuantVGG16,
    QuantVGG19,
    QuantMobileNetV1,
)

ModelType = Union[Type[nn.Module], Callable[..., nn.Module]]
"""Type alias for model factories (class or callable returning nn.Module)."""

MODELS: Dict[str, ModelType] = {
    # CNV
    "cnv": CNV,
    # MobileNet
    "mobilenetv1": MobileNetV1,
    # ResNet (CIFAR-specific)
    "resnet20": ResNet20,
    "resnet32": ResNet32,
    "resnet44": ResNet44,
    "resnet56": ResNet56,
    "resnet110": ResNet110,
    # ResNet (ImageNet-style, adapted for small images)
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    # VGG
    "vgg11": VGG11,
    "vgg13": VGG13,
    "vgg16": VGG16,
    "vgg19": VGG19,
}
"""Registry mapping model names to their implementation classes/factories."""

QUANT_MODELS: Dict[str, ModelType] = {
    # Quantized CNV
    "quant_cnv": QuantCNV,
    # Quantized MobileNet
    "quant_mobilenetv1": QuantMobileNetV1,
    # Quantized ResNet (CIFAR-specific)
    "quant_resnet20": QuantResNet20,
    "quant_resnet32": QuantResNet32,
    "quant_resnet44": QuantResNet44,
    "quant_resnet56": QuantResNet56,
    "quant_resnet110": QuantResNet110,
    # Quantized ResNet (ImageNet-style)
    "quant_resnet18": QuantResNet18,
    "quant_resnet34": QuantResNet34,
    "quant_resnet50": QuantResNet50,
    "quant_resnet101": QuantResNet101,
    "quant_resnet152": QuantResNet152,
    # Quantized VGG
    "quant_vgg11": QuantVGG11,
    "quant_vgg13": QuantVGG13,
    "quant_vgg16": QuantVGG16,
    "quant_vgg19": QuantVGG19,
}
"""Registry mapping quantized model names to their implementation classes."""

# Merge both registries
MODELS.update(QUANT_MODELS)


def get_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create a model from configuration.
    
    Looks up the model name in the MODELS registry and instantiates
    it with the appropriate parameters. For quantized models (prefixed with
    'quant_'), also passes weight_bit_width and act_bit_width from the
    quantization config section.
    
    Args:
        config: Configuration dictionary containing model settings
            under config["model"] and optionally config["quantization"].
    
    Returns:
        Initialized nn.Module model.
        
    Raises:
        ValueError: If the model name is not found in the registry.
    
    Example:
        ```yaml
        model:
          name: "quant_resnet20"
          num_classes: 10
        
        quantization:
          weight_bit_width: 4
          act_bit_width: 4
        ```
    """
    model_name: str = config["model"]["name"].lower()
    num_classes: int = config["model"].get("num_classes", 10)
    in_channels: int = config["model"].get("in_channels", 3)
    
    if model_name not in MODELS:
        available: List[str] = list(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    # Check if this is a quantized model
    if model_name.startswith("quant_"):
        quant_config: Dict[str, Any] = config.get("quantization", {})
        weight_bit_width: int = quant_config.get("weight_bit_width", 8)
        act_bit_width: int = quant_config.get("act_bit_width", 8)
        
        return MODELS[model_name](
            num_classes=num_classes,
            in_channels=in_channels,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
        )
    
    return MODELS[model_name](num_classes=num_classes, in_channels=in_channels)
