"""!
@file utils/optimizer.py
@brief Optimizer factory for creating optimizers from configuration.

@details Provides a factory pattern for instantiating PyTorch optimizers
(SGD, Adam, AdamW) based on YAML configuration dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import torch.optim as optim
from torch import nn


class OptimizerFactory:
    """!
    @brief Factory class for creating optimizers from configuration.
    
    @details Supports SGD, Adam, and AdamW optimizers with their
    respective hyperparameters extracted from config dictionaries.
    
    @par Supported Optimizers
    - sgd: Stochastic Gradient Descent with momentum
    - adam: Adam optimizer
    - adamw: AdamW optimizer with decoupled weight decay
    """
    
    ## @var OPTIMIZERS
    #  @brief Registry of available optimizer classes
    OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }
    
    @classmethod
    def create(cls, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """!
        @brief Create an optimizer based on configuration.
        
        @param model The model whose parameters will be optimized
        @param config Configuration dictionary with optimizer settings
                      under config["optimizer"]
        @return Configured optimizer instance
        @throws ValueError If the optimizer name is unknown
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
        """!
        @brief Get list of available optimizer names.
        
        @return List of supported optimizer name strings
        """
        return list(cls.OPTIMIZERS.keys())
