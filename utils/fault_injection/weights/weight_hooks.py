"""Forward hook for weight fault injection.

Provides WeightFaultInjectionHook that modifies layer weights
during forward passes.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..statistics import FaultStatistics


class WeightFaultInjectionHook:
    """Forward pre-hook for injecting faults into layer weights.
    
    This hook is registered on target layers (QuantConv2d, QuantLinear)
    and modifies the weight tensor before each forward pass.
    
    The hook:
    1. Extracts weight quantization parameters (scale, bit_width, etc.)
    2. Generates a random mask based on probability
    3. Converts weights to integer representation
    4. Applies the injection strategy
    5. Converts back to float and updates the layer's weight
    
    Attributes:
        layer_id: Unique identifier for this layer.
        probability: Injection probability (0-100).
        strategy: Fault injection strategy instance.
        verbose: Print injection details.
        statistics: Optional statistics tracker.
        enabled: Whether injection is currently active.
    """
    
    def __init__(
        self,
        layer_id: int,
        probability: float,
        strategy: Any,
        verbose: bool = False,
    ) -> None:
        """Initialize the hook.
        
        Args:
            layer_id: Unique identifier for this layer.
            probability: Injection probability (0-100).
            strategy: Injection strategy instance.
            verbose: Print injection details.
        """
        self.layer_id = layer_id
        self.probability = probability
        self.strategy = strategy
        self.verbose = verbose
        self.statistics: Optional[FaultStatistics] = None
        self.enabled = True
        
        # Cache original weight for restoration
        self._original_weight: Optional[Tensor] = None
        self._faulty_weight: Optional[Tensor] = None
    
    def __call__(
        self,
        module: nn.Module,
        input: Tuple[Tensor, ...],
    ) -> None:
        """Hook called before forward pass.
        
        Args:
            module: The layer being hooked.
            input: Input tensors to the layer (not used).
        """
        if not self.enabled or self.probability == 0.0:
            return
        
        # Check if module has weight
        if not hasattr(module, 'weight') or module.weight is None:
            return
        
        weight = module.weight
        
        # Check if weight is quantized (Brevitas)
        if not hasattr(weight, 'bit_width') or weight.bit_width is None:
            if self.verbose:
                print(f"Layer {self.layer_id}: Weight not quantized, skipping")
            return
        
        # Extract quantization parameters
        scale = weight.scale
        bit_width = int(weight.bit_width.item())
        signed = weight.signed
        zero_point = getattr(weight, 'zero_point', None)
        
        # Generate mask for fault injection
        num_elements = weight.value.numel()
        num_faults = max(1, int(num_elements * self.probability / 100.0))
        
        mask = torch.zeros(num_elements, dtype=torch.bool, device=weight.device)
        fault_indices = torch.randperm(num_elements, device=weight.device)[:num_faults]
        mask.view(-1)[fault_indices] = True
        mask = mask.view(weight.value.shape)
        
        if self.verbose:
            print(
                f"Layer {self.layer_id}: Injecting {num_faults}/{num_elements} "
                f"weight faults ({self.probability:.2f}%)"
            )
        
        # Convert to integer representation
        if zero_point is not None:
            int_weights = torch.round(weight.value / scale) + zero_point
        else:
            int_weights = torch.round(weight.value / scale)
        int_weights = int_weights.to(torch.int32)
        
        # Apply injection strategy
        faulty_int_weights = self.strategy.inject(
            int_tensor=int_weights,
            mask=mask,
            bit_width=bit_width,
            signed=signed,
            device=weight.device,
        )
        
        # Convert back to float
        if zero_point is not None:
            faulty_weights = (faulty_int_weights - zero_point).float() * scale
        else:
            faulty_weights = faulty_int_weights.float() * scale
        
        # Update statistics if tracking
        if self.statistics is not None:
            self.statistics.record_injection(
                layer_id=self.layer_id,
                original=weight.value,
                faulty=faulty_weights,
                mask=mask,
            )
        
        # CRITICAL: Modify the weight in-place
        # Store original for potential restoration
        if self._original_weight is None:
            self._original_weight = weight.value.clone()
        
        # Update weight data
        with torch.no_grad():
            weight.value.copy_(faulty_weights)
    
    def set_probability(self, probability: float) -> None:
        """Update injection probability.
        
        Args:
            probability: New probability (0-100).
        """
        self.probability = probability
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable injection.
        
        Args:
            enabled: Whether to enable injection.
        """
        self.enabled = enabled
    
    def restore_original_weights(self, module: nn.Module) -> None:
        """Restore original weights if cached.
        
        Args:
            module: The layer to restore.
        """
        if self._original_weight is not None and hasattr(module, 'weight'):
            with torch.no_grad():
                module.weight.value.copy_(self._original_weight)
            self._original_weight = None
