"""Fault injection layers for quantized models.

Provides layers that can be inserted into quantized models to inject
faults into activations. Supports both training (FAT) and evaluation modes.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from brevitas.quant_tensor import QuantTensor

from .strategies import InjectionStrategy, get_strategy

if TYPE_CHECKING:
    from .statistics import FaultStatistics


class HardwareMaskGenerator:
    """Generates hardware-aware periodic fault masks.

    This class creates deterministic, periodic fault patterns that simulate
    how faults occur in real hardware accelerators (FPGAs/ASICs) with fixed
    parallelism.

    Instead of randomly selecting which activations to inject faults into,
    it creates a periodic pattern based on the hardware's parallelism factor
    (frequency_value). This is more realistic for hardware-in-the-loop
    validation and fault characterization.

    Attributes:
        frequency_value: Number of parallel processing units (e.g., MAC units).
        probability: Fault probability as percentage (0-100).
        seed: Random seed for reproducible mask patterns.

    Example:
        ```python
        generator = HardwareMaskGenerator(
            frequency_value=1024,
            probability=5.0,
            seed=42,
        )
        mask = generator.generate(shape=(32, 64, 8, 8), device=torch.device("cuda"))
        ```
    """

    def __init__(
        self,
        frequency_value: int = 1024,
        probability: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the hardware mask generator.

        Args:
            frequency_value: Hardware parallelism factor (number of parallel units).
            probability: Fault probability as percentage (0-100).
            seed: Random seed for reproducible patterns.
        """
        self.frequency_value = frequency_value
        self.probability = probability
        self.seed = seed

    def generate(
        self,
        shape: torch.Size,
        device: torch.device,
    ) -> Tensor:
        """Generate a hardware-aware periodic fault mask.

        Creates a boolean tensor where True indicates positions to inject faults.
        The pattern is periodic based on frequency_value and probability.

        Args:
            shape: Shape of the activation tensor (batch, channels, [height, width]).
            device: Device to create the mask on.

        Returns:
            Boolean tensor with shape matching input, indicating fault positions.
        """
        # Calculate number of fault positions in one period
        ones = math.ceil(self.frequency_value * self.probability / 100.0)
        zeros = self.frequency_value - ones

        # Create and shuffle the base pattern
        if self.seed is not None:
            random.seed(self.seed)

        frequency_mask_list = ([1] * ones) + ([0] * zeros)
        random.shuffle(frequency_mask_list)
        frequency_mask_np = np.array(frequency_mask_list, dtype=np.float32)

        # Create output mask
        mask_np = np.zeros(shape, dtype=np.float32)

        if len(shape) == 4:
            # 4D tensor: (batch, channels, height, width)
            mask_np = self._generate_4d_mask(shape, frequency_mask_np)
        elif len(shape) == 2:
            # 2D tensor: (batch, features)
            mask_np = self._generate_2d_mask(shape, frequency_mask_np)
        else:
            # Fallback: tile the pattern across all dimensions
            mask_np = self._generate_nd_mask(shape, frequency_mask_np)

        # Convert to tensor
        mask_tensor = torch.tensor(mask_np, device=device, dtype=torch.bool)
        return mask_tensor

    def _generate_4d_mask(
        self,
        shape: torch.Size,
        frequency_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate mask for 4D tensors (conv layers).

        Args:
            shape: (batch, channels, height, width).
            frequency_mask: Base periodic pattern.

        Returns:
            4D numpy mask array.
        """
        batch_size, channels, height, width = shape
        mask_np = np.zeros(shape, dtype=np.float32)

        if channels <= self.frequency_value:
            # Fewer channels than frequency: distribute pattern across spatial dims
            positions_needed = self.frequency_value // channels
            positions_filled = 0

            for b in range(batch_size):
                for h in range(height):
                    for w in range(width):
                        if positions_filled < positions_needed:
                            start_idx = positions_filled * channels
                            end_idx = start_idx + channels
                            if end_idx <= self.frequency_value:
                                if width == 1 and height == 1:
                                    # Special case: 1x1 spatial
                                    mask_np[b, :, h, w] = frequency_mask[:channels]
                                else:
                                    mask_np[b, :, h, w] = frequency_mask[start_idx:end_idx]

                                positions_filled += 1
                                if positions_filled == positions_needed:
                                    positions_filled = 0
        else:
            # More channels than frequency: tile pattern across channels
            repeats_needed = (channels + self.frequency_value) // self.frequency_value
            repeated_mask = np.tile(frequency_mask, repeats_needed)[:channels]
            for b in range(batch_size):
                for h in range(height):
                    for w in range(width):
                        mask_np[b, :, h, w] = repeated_mask

        return mask_np

    def _generate_2d_mask(
        self,
        shape: torch.Size,
        frequency_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate mask for 2D tensors (linear layers).

        Args:
            shape: (batch, features).
            frequency_mask: Base periodic pattern.

        Returns:
            2D numpy mask array.
        """
        batch_size, features = shape
        mask_np = np.zeros(shape, dtype=np.float32)

        if features <= self.frequency_value:
            # Fewer features than frequency: use first portion of pattern
            for b in range(batch_size):
                mask_np[b, :] = frequency_mask[:features]
        else:
            # More features than frequency: tile pattern
            repeats_needed = (features + self.frequency_value) // self.frequency_value
            repeated_mask = np.tile(frequency_mask, repeats_needed)[:features]
            for b in range(batch_size):
                mask_np[b, :] = repeated_mask

        return mask_np

    def _generate_nd_mask(
        self,
        shape: torch.Size,
        frequency_mask: np.ndarray,
    ) -> np.ndarray:
        """Generate mask for arbitrary N-dimensional tensors.

        Args:
            shape: Tensor shape.
            frequency_mask: Base periodic pattern.

        Returns:
            N-dimensional numpy mask array.
        """
        total_elements = int(np.prod(shape))
        repeats_needed = (total_elements + self.frequency_value) // self.frequency_value
        repeated_mask = np.tile(frequency_mask, repeats_needed)[:total_elements]
        return repeated_mask.reshape(shape)

    def set_probability(self, probability: float) -> None:
        """Update the fault probability.

        Args:
            probability: New probability (0-100).
        """
        self.probability = probability


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
        hw_mask: bool = False,
        frequency_value: int = 1024,
        seed: Optional[int] = None,
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
            hw_mask: Use hardware-aware periodic fault pattern.
            frequency_value: Hardware parallelism factor for hw_mask.
            seed: Random seed for reproducible fault patterns.
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
        self.hw_mask = hw_mask
        self.frequency_value = frequency_value
        self.seed = seed

        # Strategy for fault injection
        if strategy is None:
            strategy = get_strategy("random")
        self.strategy = strategy

        # Hardware mask generator (created lazily)
        self._hw_mask_generator: Optional[HardwareMaskGenerator] = None
        if hw_mask:
            self._hw_mask_generator = HardwareMaskGenerator(
                frequency_value=frequency_value,
                probability=probability,
                seed=seed,
            )

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
        # Use hardware mask if enabled
        if self.hw_mask and self._hw_mask_generator is not None:
            return self._get_hw_mask_condition_tensor(shape, device)

        if self.use_mode == "train":
            return self._get_train_condition_tensor(shape, device)
        else:
            return self._get_eval_condition_tensor(shape, device)

    def _get_hw_mask_condition_tensor(
        self, shape: torch.Size, device: torch.device
    ) -> Optional[Tensor]:
        """Generate condition tensor using hardware-aware periodic pattern.

        This method uses the HardwareMaskGenerator to create a deterministic,
        periodic fault pattern that simulates real hardware behavior.

        Args:
            shape: Shape of the activation tensor.
            device: Device to create tensor on.

        Returns:
            Boolean tensor for fault injection with periodic pattern.
        """
        # Only generate mask when needed (same logic as random mask)
        if self.use_mode == "train":
            # Training: check injection layer and counter
            if self.injection_layer == self.num_layers:
                if self.epoch % self.epoch_interval == 0:
                    if self.counter == 0:
                        self._condition_tensor = self._hw_mask_generator.generate(
                            shape, device
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
                    # Non-interval epoch
                    if self.counter == 0:
                        self._condition_tensor = self._hw_mask_generator.generate(
                            shape, device
                        )
            elif self.injection_layer == self.layer_id:
                self._condition_tensor = self._hw_mask_generator.generate(
                    shape, device
                )
        else:
            # Evaluation: generate once per evaluation run
            if self.counter == 0:
                if self.injection_layer == self.num_layers:
                    self._condition_tensor = self._hw_mask_generator.generate(
                        shape, device
                    )
                elif self.layer_id == self.injection_layer:
                    self._condition_tensor = self._hw_mask_generator.generate(
                        shape, device
                    )

        return self._condition_tensor

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
        # Also update hardware mask generator if present
        if self._hw_mask_generator is not None:
            self._hw_mask_generator.set_probability(probability)

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
