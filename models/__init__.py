"""!
@file models/__init__.py
@brief Model module initialization and registry.

@details Provides a unified interface for creating neural network models
through a registry pattern. Includes ResNet (CIFAR and ImageNet variants),
VGG, and MobileNet architectures.

@see get_model for the main factory function
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type, Union

import torch.nn as nn

from .mobilenet import MobileNetV1
from .resnet_cifar import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
from .resnet_imagenet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG11, VGG13, VGG16, VGG19

## @var ModelType
#  @brief Type alias for model factories (class or callable returning nn.Module)
ModelType = Union[Type[nn.Module], Callable[..., nn.Module]]

## @var MODELS
#  @brief Registry mapping model names to their implementation classes/factories.
MODELS: Dict[str, ModelType] = {
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


def get_model(config: Dict[str, Any]) -> nn.Module:
    """!
    @brief Factory function to create a model from configuration.
    
    @details Looks up the model name in the MODELS registry and instantiates
    it with num_classes and in_channels from the configuration.
    
    @param config Configuration dictionary containing model settings
                  under config["model"]
    @return Initialized nn.Module model
    @throws ValueError If the model name is not found in the registry
    """
    model_name: str = config["model"]["name"].lower()
    num_classes: int = config["model"].get("num_classes", 10)
    in_channels: int = config["model"].get("in_channels", 3)
    
    if model_name not in MODELS:
        available: List[str] = list(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    return MODELS[model_name](num_classes=num_classes, in_channels=in_channels)
