"""Model module initialization and registry.

Provides a unified interface for creating neural network models
through a registry pattern. Includes ResNet (CIFAR and ImageNet variants),
MobileNet, CNV architectures, and their quantized versions using Brevitas.

Models with dataset-specific variants (e.g., quant_mobilenet) automatically
resolve to the appropriate implementation based on the configured dataset.
Explicit variant names (e.g., quant_mobilenet_cifar) are forbidden.

See Also:
    get_model: Main factory function.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type, Union

import torch.nn as nn

from .standard import (
    CNV,
    MobileNetV1,
    MobileNetV1Finn,
    MobileNetV1ImageNet,
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
    QuantMobileNetV1CIFAR,
    QuantMobileNetV1FinnCIFAR,
    QuantMobileNetV1ImageNet,
    QuantMobileNetV1FinnImageNet,
)

ModelType = Union[Type[nn.Module], Callable[..., nn.Module]]
"""Type alias for model factories (class or callable returning nn.Module)."""

MODELS: Dict[str, ModelType] = {
    # CNV
    "cnv": CNV,
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
}
"""Registry mapping model names to their implementation classes/factories."""

QUANT_MODELS: Dict[str, ModelType] = {
    # Quantized CNV
    "quant_cnv": QuantCNV,
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
}
"""Registry mapping quantized model names to their implementation classes."""

DATASET_VARIANTS: Dict[str, Dict[str, str]] = {
    "mobilenetv1": {
        "cifar10": "MobileNetV1",
        "cifar100": "MobileNetV1",
        "imagenet": "MobileNetV1ImageNet",
    },
    "mobilenetv1_finn": {
        "cifar10": "MobileNetV1Finn",
        "cifar100": "MobileNetV1Finn",
    },
    "quant_mobilenetv1_finn": {
        "cifar10": "QuantMobileNetV1FinnCIFAR",
        "cifar100": "QuantMobileNetV1FinnCIFAR",
        "imagenet": "QuantMobileNetV1FinnImageNet",
    },
    "quant_mobilenetv1": {
        "cifar10": "QuantMobileNetV1CIFAR",
        "cifar100": "QuantMobileNetV1CIFAR",
        "imagenet": "QuantMobileNetV1ImageNet",
    },
}
"""Mapping of model variants by dataset. Used to resolve base model names to implementation classes based on dataset."""

INTERNAL_REGISTRY: Dict[str, ModelType] = {
    "MobileNetV1": MobileNetV1,
    "MobileNetV1ImageNet": MobileNetV1ImageNet,
    "MobileNetV1Finn": MobileNetV1Finn,
    "QuantMobileNetV1CIFAR": QuantMobileNetV1CIFAR,
    "QuantMobileNetV1FinnCIFAR": QuantMobileNetV1FinnCIFAR,
    "QuantMobileNetV1ImageNet": QuantMobileNetV1ImageNet,
    "QuantMobileNetV1FinnImageNet": QuantMobileNetV1FinnImageNet,
}
"""Internal registry for implementation classes not directly accessible by name."""

# Merge both registries (except internal ones)
MODELS.update(QUANT_MODELS)


def get_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create a model from configuration.

    Looks up the model name in the MODELS registry and instantiates
    it with the appropriate parameters. For quantized models (prefixed with
    'quant_'), also passes weight_bit_width and act_bit_width from the
    quantization config section.

    For models with dataset-specific variants (e.g., quant_mobilenet), the
    variant is automatically resolved based on the configured dataset.

    Args:
        config: Configuration dictionary containing model settings
            under config["model"] and optionally config["quantization"].

    Returns:
        Initialized nn.Module model.

    Raises:
        ValueError: If the model name is not found in the registry or
            if a forbidden explicit variant name is used.

    Example:
        ```yaml
        model:
          name: "quant_resnet20"
          num_classes: 10

        quantization:
          in_weight_bit_width: 8
          weight_bit_width: 4
          act_bit_width: 4
        ```
    """
    model_name: str = config["model"]["name"].lower()
    if model_name is None:
        raise ValueError("Model name must be specified in config['model']['name']")
    num_classes: int = config["model"].get("num_classes", 10)
    in_channels: int = config["model"].get("in_channels", 3)
    dataset: str = config.get("dataset", {}).get("name")
    if dataset is None:
        raise ValueError("Dataset name must be specified in config['dataset']['name']")

    # Resolve model name if it has dataset variants
    model_class: ModelType

    if model_name in DATASET_VARIANTS:
        variants = DATASET_VARIANTS[model_name]
        if dataset not in variants:
            available = list(variants.keys())
            raise ValueError(
                f"Model '{model_name}' has no variant for dataset '{dataset}'. "
                f"Available variants for {model_name}: {available}"
            )
        class_name = variants[dataset]
        resolved_class: ModelType | None = INTERNAL_REGISTRY.get(class_name)
        if resolved_class is None:
            available = list(INTERNAL_REGISTRY.keys())
            raise ValueError(
                f"Implementation class '{class_name}' not found in internal registry. "
                f"Available: {available}"
            )
        model_class = resolved_class
    elif model_name in MODELS:
        model_class = MODELS[model_name]
    else:
        available = list(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    # Check if this is a quantized model
    if model_name.startswith("quant_"):
        quant_config: Dict[str, Any] = config.get("quantization", {})
        in_weight_bit_width: int = quant_config.get("in_weight_bit_width")
        weight_bit_width: int = quant_config.get("weight_bit_width")
        act_bit_width: int = quant_config.get("act_bit_width")
        if weight_bit_width is None or act_bit_width is None or in_weight_bit_width is None:
            raise ValueError(
                "Quantized models require 'weight_bit_width', 'in_weight_bit_width', and 'act_bit_width' "
                "in config['quantization']"
            )

        return model_class(
            num_classes=num_classes,
            in_channels=in_channels,
            weight_bit_width=weight_bit_width,
            in_weight_bit_width=in_weight_bit_width,
            act_bit_width=act_bit_width,
        )

    return model_class(num_classes=num_classes, in_channels=in_channels)
