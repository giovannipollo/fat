"""Activation fault injection layers for quantized models.

Provides layers that can be inserted into quantized models to inject
faults into activations. Supports both training (FAT) and evaluation modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

from brevitas.quant_tensor import QuantTensor

from .activation_functions import ActivationFaultInjectionFunction
from ..strategies import InjectionStrategy, get_strategy

if TYPE_CHECKING:
    from .statistics import FaultStatistics


class QuantActivationFaultInjectionLayer(nn.Module):
    """Activation fault injection layer for Brevitas QuantTensor outputs.

    This layer intercepts QuantTensor activations and injects faults
    based on the configured probability and injection strategy.

    The layer handles backpropagation by zeroing gradients at faulty positions
    to train models that are robust to errors.

    Attributes:
        layer_id: Unique identifier for this injection layer.
        probability: Injection probability (0-100).
        strategy: Injection strategy instance.
        verbose: Whether to print injection details.
        statistics: Optional statistics tracker.
        injection_enabled: Whether injection is currently active.

    Example:
        ```python
        layer = QuantActivationFaultInjectionLayer(
            layer_id=0,
            probability=5.0,
            strategy=RandomStrategy(),
        )
        output = layer(quant_tensor_input)
        ```
    """

    def __init__(
        self,
        layer_id: int = 0,
        probability: float = 0.0,
        strategy: Optional[InjectionStrategy] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the activation fault injection layer.

        Args:
            layer_id: Unique identifier for this layer.
            probability: Injection probability as percentage (0-100).
            strategy: Injection strategy to use.
            verbose: Print injection details.
        """
        super().__init__()

        self.layer_id = layer_id
        self.probability = probability
        self.verbose = verbose

        # Strategy for activation fault injection
        if strategy is None:
            strategy = get_strategy("random")
        self.strategy = strategy

        self.injection_enabled: bool = True

        # Statistics tracking (set by injector)
        self.statistics: Optional["FaultStatistics"] = None

    def forward(self, x: QuantTensor) -> QuantTensor:
        """Forward pass with optional activation fault injection.

        Args:
            x: Input QuantTensor from previous layer.

        Returns:
            QuantTensor with injected activation faults (if enabled and conditions met).
        """
        # Skip injection if disabled or probability is 0
        if not self.injection_enabled or self.probability <= 0.0:
            return x

        # Extract quantization parameters
        scale = x.scale
        zero_point = x.zero_point
        bit_width = x.bit_width
        signed = x.signed
        training = x.training
        shape = x.shape
        device = x.value.device

        # Get bit width as integer
        bit_width_int = int(bit_width.item())

        # Calculate exact number of elements to inject
        total_elements = torch.numel(x.value)
        num_to_inject = int(self.probability / 100.0 * total_elements)

        if num_to_inject > 0:
            # Randomly select positions to inject
            flat_indices = torch.randperm(total_elements, device=device)[:num_to_inject]

            # Create boolean mask
            condition_tensor = torch.zeros(
                total_elements, dtype=torch.bool, device=device
            )
            condition_tensor[flat_indices] = True
            condition_tensor = condition_tensor.view(shape)
        else:
            condition_tensor = torch.zeros(shape, dtype=torch.bool, device=device)

        # If no faults to inject (all False mask), return early
        if not condition_tensor.any():
            return x

        # Convert to integer representation
        if zero_point is not None:
            int_tensor = torch.round(x.value / scale) + zero_point
        else:
            int_tensor = torch.round(x.value / scale)
        int_tensor = int_tensor.to(torch.int32)

        # Apply injection strategy
        injected_int = self.strategy.inject(
            int_tensor=int_tensor,
            mask=condition_tensor,
            bit_width=bit_width_int,
            signed=signed,
            device=device,
        )

        # Convert back to float with scale
        if zero_point is not None:
            injected_float = (injected_int - zero_point).float() * scale
        else:
            injected_float = injected_int.float() * scale

        # Apply activation fault injection with zeroed gradients at faulty positions
        output_value = ActivationFaultInjectionFunction.apply(
            x.value,
            injected_float,
            condition_tensor,
        )

        # Track statistics if enabled
        if self.statistics is not None:
            self.statistics.record(
                clean_int=int_tensor,
                faulty_int=injected_int,
                mask=condition_tensor,
                layer_id=self.layer_id,
            )

        # Verbose output
        if self.verbose:
            num_injected = condition_tensor.sum().item()
            total = condition_tensor.numel()
            print(
                f"Layer {self.layer_id}: Injected {num_injected}/{total} "
                f"({100.0 * num_injected / total:.2f}%)"
            )

        # Reconstruct QuantTensor
        return QuantTensor(
            value=output_value,
            scale=scale,
            zero_point=zero_point,
            bit_width=bit_width,
            signed=torch.tensor(signed),
            training=torch.tensor(training),
        )

    def set_probability(self, probability: float) -> None:
        """Set the activation injection probability.

        Args:
            probability: New probability (0-100).
        """
        self.probability = probability

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable activation fault injection.

        Args:
            enabled: Whether to enable activation injection.
        """
        self.injection_enabled = enabled
