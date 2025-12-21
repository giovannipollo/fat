"""Wrapper for fault injection layers.

Provides the _InjectionWrapper class that combines a layer with its
fault injection counterpart.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .layers import QuantFaultInjectionLayer


class _InjectionWrapper(nn.Module):
    """Wrapper that applies fault injection after a layer.

    This wrapper is used for non-Sequential layers where we can't
    simply insert a new layer after the target.

    Attributes:
        wrapped_layer: The original layer being wrapped.
        injection_layer: The fault injection layer.
    """

    def __init__(
        self,
        wrapped_layer: nn.Module,
        injection_layer: QuantFaultInjectionLayer,
    ) -> None:
        """Initialize the wrapper.

        Args:
            wrapped_layer: Original layer to wrap.
            injection_layer: Fault injection layer to apply after.
        """
        super().__init__()
        self.wrapped_layer = wrapped_layer
        self.injection_layer = injection_layer

    def forward(self, x: Any) -> Any:
        """Forward pass through wrapped layer then injection.

        Args:
            x: Input tensor.

        Returns:
            Output after wrapped layer and fault injection.
        """
        out = self.wrapped_layer(x)
        return self.injection_layer(out)

    def __repr__(self) -> str:
        return (
            f"_InjectionWrapper(\n"
            f"  (wrapped): {self.wrapped_layer}\n"
            f"  (injection): {self.injection_layer}\n"
            f")"
        )