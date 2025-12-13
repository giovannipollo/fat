"""Fault injection layers for quantized models.

Provides layers that can be inserted into quantized models to inject
faults into activations. Supports both training (FAT) and evaluation modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch import Tensor

from brevitas.quant_tensor import QuantTensor

from .strategies import InjectionStrategy, get_strategy

if TYPE_CHECKING:
    from .statistics import FaultStatistics


class QuantFaultInjectionLayer(nn.Module):
    """Fault injection layer for Brevitas QuantTensor outputs.

    This layer intercepts QuantTensor activations and injects faults
    based on the configured probability and injection strategy.

    The layer handles backpropagation correctly by:
    1. Using torch.no_grad() for fault injection operations
    2. Implementing straight-through estimator (STE) for gradients
    3. Preserving the computation graph for non-injected values

    Attributes:
        layer_id: Unique identifier for this injection layer.
        probability: Injection probability (0-100).
        injection_layer: Target layer for "layer" mode.
        num_layers: Total number of injection layers in model.
        strategy: Injection strategy instance.
        mode: Injection mode ("full_model" or "layer").
        use_mode: Current usage mode ("train" or "eval").
        epoch: Current training epoch.
        counter: Iteration counter within epoch.
        epoch_interval: Re-generate mask every N epochs.
        step_interval: Probability of injection per step.
        condition_injector: Pre-computed injection schedule for steps.
        statistics: Optional statistics tracker.
        injection_enabled: Whether injection is currently active.

    Example:
        ```python
        layer = QuantFaultInjectionLayer(
            layer_id=0,
            probability=5.0,
            num_layers=10,
            strategy=RandomStrategy(),
        )
        output = layer(quant_tensor_input)
        ```
    """

    def __init__(
        self,
        layer_id: int = 0,
        probability: float = 0.0,
        injection_layer: int = 0,
        num_layers: int = 0,
        strategy: Optional[InjectionStrategy] = None,
        mode: str = "full_model",
        epoch_interval: int = 1,
        step_interval: float = 0.5,
        verbose: bool = False,
    ) -> None:
        """Initialize the fault injection layer.

        Args:
            layer_id: Unique identifier for this layer.
            probability: Injection probability as percentage (0-100).
            injection_layer: Target layer index for "layer" mode.
            num_layers: Total number of injection layers in model.
            strategy: Injection strategy to use.
            mode: Injection mode ("full_model" or "layer").
            epoch_interval: Re-generate fault mask every N epochs.
            step_interval: Probability of injection per batch step.
            verbose: Print injection details.
        """
        super().__init__()

        self.layer_id = layer_id
        self.probability = probability
        self.injection_layer = injection_layer
        self.num_layers = num_layers
        self.mode = mode
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self.verbose = verbose

        # Strategy for fault injection
        if strategy is None:
            strategy = get_strategy("random")
        self.strategy = strategy

        # Runtime state
        self.use_mode: str = "eval"
        self.epoch: int = 0
        self.counter: int = 0
        self.num_iterations: int = 0
        self.injection_enabled: bool = True

        # Pre-computed tensors for training mode
        self.condition_injector: Optional[Tensor] = None
        self._p_tensor: Optional[Tensor] = None
        self._condition_tensor: Optional[Tensor] = None
        self._condition_tensor_true: Optional[Tensor] = None
        self._condition_tensor_false: Optional[Tensor] = None
        self._rand_int_tensor: Optional[Tensor] = None

        # Statistics tracking (set by injector)
        self.statistics: Optional["FaultStatistics"] = None

    def forward(self, x: QuantTensor) -> QuantTensor:
        """Forward pass with optional fault injection.

        Args:
            x: Input QuantTensor from previous layer.

        Returns:
            QuantTensor with injected faults (if enabled and conditions met).
        """
        # Skip injection if disabled or probability is 0
        if not self.injection_enabled or self.probability <= 0.0:
            return x

        # Check if this layer should inject based on mode
        if not self._should_inject():
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

        # Generate or update fault mask based on mode
        condition_tensor = self._get_condition_tensor(shape, device)

        # If no faults to inject (all False mask), return early
        if condition_tensor is None or not condition_tensor.any():
            self._update_counter()
            return x

        # Convert to integer representation
        inv_scale = 1.0 / scale
        int_tensor = torch.round(x.value * inv_scale).to(torch.int32)

        # Apply injection strategy
        injected_int = self.strategy.inject(
            int_tensor=int_tensor,
            mask=condition_tensor,
            bit_width=bit_width_int,
            signed=signed,
            device=device,
        )

        # Convert back to float with scale
        injected_float = injected_int.float() * scale

        # Apply straight-through estimator for backpropagation:
        # Forward: use injected values
        # Backward: pass gradients through as if no injection happened
        # This is done by detaching the injected values and adding back the gradient path
        with torch.no_grad():
            # Compute the difference caused by injection
            injection_delta = injected_float - x.value

        # Apply injection while preserving gradient flow for non-injected values
        # For injected values, gradients are zeroed (fault is not differentiable)
        output_value = x.value + injection_delta

        # Track statistics if enabled
        if self.statistics is not None:
            self.statistics.record(
                clean_int=int_tensor,
                faulty_int=injected_int,
                mask=condition_tensor,
                layer_id=self.layer_id,
            )

        # Update counter for training mode
        self._update_counter()

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

    def _should_inject(self) -> bool:
        """Check if this layer should inject faults based on mode.

        Returns:
            True if this layer should perform injection.
        """
        if self.mode == "full_model":
            return True
        elif self.mode == "layer":
            return self.layer_id == self.injection_layer
        return False

    def _get_condition_tensor(
        self, shape: torch.Size, device: torch.device
    ) -> Optional[Tensor]:
        """Get or generate the condition tensor (fault mask).

        Args:
            shape: Shape of the activation tensor.
            device: Device to create tensor on.

        Returns:
            Boolean tensor indicating which values to inject, or None.
        """
        if self.use_mode == "train":
            return self._get_train_condition_tensor(shape, device)
        else:
            return self._get_eval_condition_tensor(shape, device)

    def _get_train_condition_tensor(
        self, shape: torch.Size, device: torch.device
    ) -> Optional[Tensor]:
        """Generate condition tensor for training mode.

        Implements the epoch_interval and step_interval logic from
        the original fat/src implementation.

        Args:
            shape: Shape of the activation tensor.
            device: Device to create tensor on.

        Returns:
            Boolean tensor for fault injection.
        """
        # Full model injection
        if self.injection_layer == self.num_layers:
            # Check if this is an epoch where we regenerate masks
            if self.epoch % self.epoch_interval == 0:
                if self.counter == 0:
                    # Generate new fault mask
                    self._p_tensor = torch.rand(shape, device=device)
                    self._condition_tensor = (
                        self._p_tensor < self.probability / 100.0
                    )
                    self._condition_tensor_true = self._condition_tensor.clone()
                    self._condition_tensor_false = torch.zeros(
                        shape, dtype=torch.bool, device=device
                    )
                else:
                    # Use pre-computed schedule
                    if (
                        self.condition_injector is not None
                        and self.counter < len(self.condition_injector)
                    ):
                        if self.condition_injector[self.counter]:
                            self._condition_tensor = self._condition_tensor_true
                        else:
                            self._condition_tensor = self._condition_tensor_false
            else:
                # Non-interval epoch: generate fresh mask each iteration
                if self.counter == 0:
                    self._p_tensor = torch.rand(shape, device=device)
                    self._condition_tensor = (
                        self._p_tensor < self.probability / 100.0
                    )

        # Single layer injection
        elif self.injection_layer == self.layer_id:
            self._p_tensor = torch.rand(shape, device=device)
            self._condition_tensor = self._p_tensor < self.probability / 100.0

        return self._condition_tensor

    def _get_eval_condition_tensor(
        self, shape: torch.Size, device: torch.device
    ) -> Optional[Tensor]:
        """Generate condition tensor for evaluation mode.

        In evaluation mode, we generate the mask once per forward pass
        (counter == 0) for consistent injection across all samples.

        Args:
            shape: Shape of the activation tensor.
            device: Device to create tensor on.

        Returns:
            Boolean tensor for fault injection.
        """
        # Only generate mask on first iteration
        if self.counter == 0:
            # Full model injection
            if self.injection_layer == self.num_layers:
                self._p_tensor = torch.rand(shape, device=device)
                self._condition_tensor = (
                    self._p_tensor < self.probability / 100.0
                )

            # Single layer injection
            elif self.layer_id == self.injection_layer:
                self._p_tensor = torch.rand(shape, device=device)
                self._condition_tensor = (
                    self._p_tensor < self.probability / 100.0
                )

        return self._condition_tensor

    def _update_counter(self) -> None:
        """Update the iteration counter based on mode and settings."""
        if self.use_mode == "train":
            if self.injection_layer == self.num_layers:
                self.counter += 1
            elif self.injection_layer == self.layer_id:
                self.counter += 1

    def reset_counter(self) -> None:
        """Reset the iteration counter to 0."""
        self.counter = 0
        self._condition_tensor = None
        self._p_tensor = None

    def set_probability(self, probability: float) -> None:
        """Set the injection probability.

        Args:
            probability: New probability (0-100).
        """
        self.probability = probability

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def set_injection_layer(self, injection_layer: int) -> None:
        """Set the target injection layer.

        Args:
            injection_layer: Layer index or num_layers for full model.
        """
        self.injection_layer = injection_layer

    def set_use_mode(self, mode: str) -> None:
        """Set the usage mode (train or eval).

        Args:
            mode: Either "train" or "eval".
        """
        self.use_mode = mode

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable fault injection.

        Args:
            enabled: Whether to enable injection.
        """
        self.injection_enabled = enabled

    def set_condition_injector(self, condition_injector: Tensor) -> None:
        """Set the pre-computed injection schedule for training.

        Args:
            condition_injector: Boolean tensor indicating injection per step.
        """
        self.condition_injector = condition_injector

    def set_num_iterations(self, num_iterations: int) -> None:
        """Set the number of iterations per epoch.

        Args:
            num_iterations: Number of iterations.
        """
        self.num_iterations = num_iterations


# Compatibility alias for existing fat/src code
ErrInjLayer = QuantFaultInjectionLayer
