"""Wrapper for activation fault injection layers.

Provides the _ActivationFaultInjectionWrapper class that combines a layer with its
activation fault injection counterpart.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .activation_layers import QuantActivationFaultInjectionLayer


class _ActivationFaultInjectionWrapper(nn.Module):
    """Wrapper that applies activation fault injection after a layer.

    This wrapper is used for non-Sequential layers where we can't
    simply insert a new layer after the target.

    Attributes:
        wrapped_layer: The original layer being wrapped.
        activation_injection_layer: The activation fault injection layer.
    """

    def __init__(
        self,
        wrapped_layer: nn.Module,
        activation_injection_layer: QuantActivationFaultInjectionLayer,
    ) -> None:
        """Initialize the wrapper.

        Args:
            wrapped_layer: Original layer to wrap.
            activation_injection_layer: Activation fault injection layer to apply after.
        """
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.activation_injection_layer = activation_injection_layer

    def forward(self, x: Any) -> Any:
        """Forward pass through wrapped layer then activation fault injection.

        Args:
            x: Input tensor.

        Returns:
            Output after wrapped layer and activation fault injection.
        """
        if self.wrapped_layer.__class__.__name__ == "QuantConv2d" or self.wrapped_layer.__class__.__name__ == "QuantLinear":
            # For Conv2d and Linear, apply injection before the layer if bit_width is present
            if hasattr(x, 'bit_width') and x.bit_width is not None:
                out = self.activation_injection_layer(x)
                out = self.wrapped_layer(out)
            else:
                out = self.wrapped_layer(x)
        else:
            out = self.wrapped_layer(x)
            if hasattr(out, 'bit_width') and out.bit_width is not None:
                out = self.activation_injection_layer(out)
        return out

    def __repr__(self) -> str:
        return (
            f"_ActivationFaultInjectionWrapper(\n"
            f"  (wrapped): {self.wrapped_layer}\n"
            f"  (activation_injection): {self.activation_injection_layer}\n"
            f")"
        )