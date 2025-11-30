"""Optimizer factory for creating optimizers from configuration.

Provides a factory pattern for instantiating PyTorch optimizers
(SGD, Adam, AdamW) based on YAML configuration dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import torch.optim as optim
from torch import nn


class OptimizerFactory:
    """Factory class for creating optimizers from configuration.
    
    Supports SGD, Adam, and AdamW optimizers with their
    respective hyperparameters extracted from config dictionaries.
    
    Supported Optimizers:
        - sgd: Stochastic Gradient Descent with momentum
        - adam: Adam optimizer
        - adamw: AdamW optimizer with decoupled weight decay
    
    Attributes:
        OPTIMIZERS: Registry of available optimizer classes.
    
    Example:
        ```python
        optimizer = OptimizerFactory.create(model, config)
        ```
    """
    
    OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }
    """Registry of available optimizer classes."""
    
    @classmethod
    def create(cls, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """Create an optimizer based on configuration.
        
        Args:
            model: The model whose parameters will be optimized.
            config: Configuration dictionary with optimizer settings
                under config["optimizer"].
        
        Returns:
            Configured optimizer instance.
            
        Raises:
            ValueError: If the optimizer name is unknown.
        """
        opt_config: Dict[str, Any] = config["optimizer"]
        opt_name: str = opt_config["name"].lower()
        
        if opt_name not in cls.OPTIMIZERS:
            available: List[str] = list(cls.OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer: {opt_name}. Available: {available}")
        
        # Common parameters
        params: Dict[str, Any] = {
            "params": model.parameters(),
            "lr": float(opt_config["learning_rate"]),
            "weight_decay": float(opt_config.get("weight_decay", 0)),
        }
        
        # SGD-specific parameters
        if opt_name == "sgd":
            params["momentum"] = float(opt_config.get("momentum", 0.9))
            params["nesterov"] = opt_config.get("nesterov", False)
        
        # Adam/AdamW-specific parameters
        if opt_name in ("adam", "adamw"):
            betas = opt_config.get("betas", (0.9, 0.999))
            params["betas"] = (float(betas[0]), float(betas[1]))
            params["eps"] = float(opt_config.get("eps", 1e-8))
        
        return cls.OPTIMIZERS[opt_name](**params)
    
    @classmethod
    def available_optimizers(cls) -> List[str]:
        """Get list of available optimizer names.
        
        Returns:
            List of supported optimizer name strings.
        """
        return list(cls.OPTIMIZERS.keys())
