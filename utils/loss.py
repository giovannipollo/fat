"""!
@file utils/loss.py
@brief Loss function factory for creating loss functions from configuration.

@details Provides a factory pattern for instantiating PyTorch loss functions
based on YAML configuration dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import torch.nn as nn

from .losses import SqrHingeLoss


class LossFactory:
    """!
    @brief Factory class for creating loss functions from configuration.
    
    @details Supports common classification loss functions with their
    respective hyperparameters extracted from config dictionaries.
    
    @par Supported Loss Functions
    - cross_entropy: CrossEntropyLoss (default)
    - nll: Negative Log Likelihood Loss
    - mse: Mean Squared Error Loss
    - l1: L1 Loss (Mean Absolute Error)
    - smooth_l1: Smooth L1 Loss (Huber Loss)
    - bce: Binary Cross Entropy Loss
    - bce_with_logits: Binary Cross Entropy with Logits Loss
    - kl_div: Kullback-Leibler Divergence Loss
    - sqr_hinge: Squared Hinge Loss (for SVM-style training)
    """
    
    ## @var LOSSES
    #  @brief Registry of available loss function classes
    LOSSES: Dict[str, Type[nn.Module]] = {
        "cross_entropy": nn.CrossEntropyLoss,
        "nll": nn.NLLLoss,
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
        "smooth_l1": nn.SmoothL1Loss,
        "bce": nn.BCELoss,
        "bce_with_logits": nn.BCEWithLogitsLoss,
        "kl_div": nn.KLDivLoss,
        "sqr_hinge": SqrHingeLoss,
    }
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> nn.Module:
        """!
        @brief Create a loss function based on configuration.
        
        @param config Configuration dictionary with optional loss settings
                      under config["loss"]
        @return Configured loss function instance
        @throws ValueError If the loss name is unknown
        
        @par Example Config
        @code{.yaml}
        loss:
          name: "cross_entropy"
          label_smoothing: 0.1  # Optional, for cross_entropy
        @endcode
        """
        loss_config: Dict[str, Any] = config.get("loss", {})
        loss_name: str = loss_config.get("name", "cross_entropy").lower()
        
        if loss_name not in cls.LOSSES:
            available: List[str] = list(cls.LOSSES.keys())
            raise ValueError(f"Unknown loss function: {loss_name}. Available: {available}")
        
        # Build parameters based on loss type
        params: Dict[str, Any] = {}
        
        # CrossEntropyLoss parameters
        if loss_name == "cross_entropy":
            if "label_smoothing" in loss_config:
                params["label_smoothing"] = float(loss_config["label_smoothing"])
            if "weight" in loss_config:
                import torch
                params["weight"] = torch.tensor(loss_config["weight"])
            if "ignore_index" in loss_config:
                params["ignore_index"] = int(loss_config["ignore_index"])
        
        # NLLLoss parameters
        elif loss_name == "nll":
            if "weight" in loss_config:
                import torch
                params["weight"] = torch.tensor(loss_config["weight"])
            if "ignore_index" in loss_config:
                params["ignore_index"] = int(loss_config["ignore_index"])
        
        # SmoothL1Loss parameters
        elif loss_name == "smooth_l1":
            if "beta" in loss_config:
                params["beta"] = float(loss_config["beta"])
        
        # KLDivLoss parameters
        elif loss_name == "kl_div":
            params["reduction"] = loss_config.get("reduction", "batchmean")
            if "log_target" in loss_config:
                params["log_target"] = bool(loss_config["log_target"])
        
        # BCEWithLogitsLoss parameters
        elif loss_name == "bce_with_logits":
            if "pos_weight" in loss_config:
                import torch
                params["pos_weight"] = torch.tensor(loss_config["pos_weight"])
        
        return cls.LOSSES[loss_name](**params)
    
    @classmethod
    def available_losses(cls) -> List[str]:
        """!
        @brief Get list of available loss function names.
        
        @return List of supported loss function name strings
        """
        return list(cls.LOSSES.keys())
