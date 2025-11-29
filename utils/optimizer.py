"""Optimizer factory for creating optimizers from configuration."""

import torch.optim as optim
from torch import nn


class OptimizerFactory:
    """Factory class for creating optimizers from configuration."""
    
    OPTIMIZERS = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }
    
    @classmethod
    def create(cls, model: nn.Module, config: dict) -> optim.Optimizer:
        """
        Create an optimizer based on configuration.
        
        Args:
            model: The model whose parameters will be optimized
            config: Configuration dictionary with optimizer settings
            
        Returns:
            The configured optimizer
            
        Raises:
            ValueError: If the optimizer name is unknown
        """
        opt_config = config["optimizer"]
        opt_name = opt_config["name"].lower()
        
        if opt_name not in cls.OPTIMIZERS:
            available = list(cls.OPTIMIZERS.keys())
            raise ValueError(f"Unknown optimizer: {opt_name}. Available: {available}")
        
        # Common parameters
        params = {
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
    def available_optimizers(cls) -> list[str]:
        """Return list of available optimizer names."""
        return list(cls.OPTIMIZERS.keys())
