"""Custom autograd functions for fault injection.

Provides the FaultInjectionFunction for controlled gradient flow.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


class FaultInjectionFunction(torch.autograd.Function):
    """Custom autograd function for fault injection with zeroed faulty gradients.

    This function applies fault injection during the forward pass and zeros
    gradients at faulty positions during the backward pass.

    Example:
        ```python
        output = FaultInjectionFunction.apply(x, faulty_values, mask)
        ```
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        faulty_values: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass: apply fault injection.

        Args:
            ctx: Autograd context for saving tensors.
            x: Original activation tensor.
            faulty_values: Tensor of faulty values to inject.
            mask: Boolean mask indicating positions to inject (True = inject).

        Returns:
            Tensor with faults injected at masked positions.
        """
        # Save mask for backward pass
        ctx.save_for_backward(mask)

        # Apply fault injection: where mask is True, use faulty_values
        output = torch.where(mask, faulty_values, x)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, None]:
        """Backward pass: zero gradients at faulty positions.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradient from upstream layers.

        Returns:
            Tuple of gradients for each forward input:
            - grad_x: Gradient for original tensor x (zeroed at faulty positions)
            - None: No gradient for faulty_values
            - None: No gradient for mask
        """
        (mask,) = ctx.saved_tensors

        # Zero gradients at faulty positions
        grad_x = torch.where(mask, torch.zeros_like(grad_output), grad_output)

        return grad_x, None, None