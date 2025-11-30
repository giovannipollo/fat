"""Squared Hinge Loss implementation.

Implements squared hinge loss for SVM-style training.
Based on AMD/Xilinx Brevitas implementation.

See: https://github.com/Xilinx/brevitas
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
License: BSD-3-Clause
"""

from __future__ import annotations

import torch
from torch.autograd import Function
import torch.nn as nn


class _SquaredHingeLossFunction(Function):
    """Autograd function for squared hinge loss computation.

    Implements forward and backward pass for squared hinge loss.
    """

    @staticmethod
    def forward(ctx, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass for squared hinge loss.

        Args:
            ctx: Autograd context for saving tensors.
            predictions: Model predictions of shape (N, C).
            targets: Target labels of shape (N, C) with values -1 or +1.

        Returns:
            Scalar loss value.
        """
        ctx.save_for_backward(predictions, targets)
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for squared hinge loss.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradient from upstream.

        Returns:
            Tuple of gradients for predictions and targets.
        """
        predictions, targets = ctx.saved_tensors
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        grad_input = targets.mul(-2.0).mul(output)
        grad_input.mul_(output.ne(0).float())
        grad_input.div_(predictions.numel())
        return grad_input * grad_output, None


class SqrHingeLoss(nn.Module):
    """Squared Hinge Loss module.

    Computes the squared hinge loss, commonly used for
    multi-class classification with SVM-style margins. Automatically
    converts class indices to one-hot encoded targets with -1/+1 values.

    Formula:
        L = mean(max(0, 1 - y * f(x))^2)

    where y is +1 for the correct class and -1 for all other classes.

    Example:
        >>> criterion = SqrHingeLoss()
        >>> # targets can be class indices (like CrossEntropyLoss)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(self) -> None:
        """Initialize SqrHingeLoss module."""
        super().__init__()

    def _to_onehot(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert class indices to one-hot encoding with -1/+1 values.

        Args:
            target: Class indices of shape (N,).
            num_classes: Number of classes.

        Returns:
            One-hot encoded tensor of shape (N, C) with -1/+1 values.
        """
        # Create tensor filled with -1
        onehot = torch.full(
            (target.size(0), num_classes),
            -1.0,
            device=target.device,
            dtype=torch.float32,
        )
        # Set correct class to +1
        onehot.scatter_(1, target.unsqueeze(1), 1.0)
        return onehot

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute squared hinge loss.

        Args:
            input: Model predictions of shape (N, C).
            target: Target labels - either class indices (N,) or one-hot (N, C).

        Returns:
            Scalar loss value.
        """
        # If target is 1D (class indices), convert to one-hot with -1/+1
        if target.dim() == 1:
            num_classes = input.size(1)
            target = self._to_onehot(target, num_classes)

        return _SquaredHingeLossFunction.apply(input, target)
