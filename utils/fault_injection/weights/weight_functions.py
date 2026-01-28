"""Custom autograd function for weight fault injection.

Provides gradient handling for weight faults during training.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from torch import Tensor


class WeightFaultInjectionFunction(torch.autograd.Function):
    """Custom autograd function for weight fault injection.
    
    This function allows fault-aware training by controlling gradient
    flow through faulty weight positions.
    
    Forward pass: Applies faults to weights
    Backward pass: Options:
        1. Zero gradients at faulty positions (default)
        2. Clip gradients at faulty positions
        3. Pass gradients through normally
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        weights: Tensor,
        faulty_weights: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply weight faults.
        
        Args:
            ctx: Context for backward pass.
            weights: Original weight tensor.
            faulty_weights: Weight tensor with injected faults.
            mask: Boolean mask indicating fault positions.
            
        Returns:
            Faulty weight tensor.
        """
        ctx.save_for_backward(mask)
        return torch.where(mask, faulty_weights, weights)
    
    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> Tuple[Tensor, None, None]:
        """Backward pass with gradient handling.
        
        Zeros gradients at positions where faults were injected to
        prevent the optimizer from updating based on faulty outputs.
        
        Args:
            ctx: Context from forward pass.
            grad_output: Gradient from loss.
            
        Returns:
            Gradient for weights, None for faulty_weights, None for mask.
        """
        (mask,) = ctx.saved_tensors
        
        # Zero gradients at faulty positions
        grad_weights = grad_output.clone()
        grad_weights = torch.where(mask, torch.zeros_like(grad_weights), grad_weights)
        
        return grad_weights, None, None
