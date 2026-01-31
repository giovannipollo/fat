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
    
    The hook performs the following operations:
    1. Extracts weight quantization parameters (scale, bit_width, etc.)
    2. Generates a random mask based on probability
    3. Converts weights to integer representation
    4. Applies the injection strategy
    5. Converts back to float and updates the layer's weight
    6. Registers a gradient hook to zero gradients at faulty positions
    
    Gradient Handling:
    Unlike simple weight modification, this implementation includes proper
    gradient management for fault-aware training. During backward pass,
    gradients at positions where faults were injected are zeroed out.
    This prevents the optimizer from updating weights based on faulty
    outputs, making the training truly fault-aware.
    
    This behavior matches the activation fault injection approach and is
    critical for effective fault-aware training (FAT).
    
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
        
        # Fault mask for gradient zeroing (updated each forward pass)
        self._current_fault_mask: Optional[Tensor] = None
        
        # Gradient hook handle for cleanup
        self._grad_hook_handle: Optional[Any] = None
    
    def _create_gradient_hook(self) -> Any:
        """Create a gradient hook function that zeros gradients at faulty positions.
        
        This hook is called during the backward pass and modifies gradients
        before they reach the optimizer, similar to activation fault injection.
        
        Returns:
            Gradient hook function.
        """
        def gradient_hook(grad: Tensor) -> Tensor:
            """Zero gradients at positions where faults were injected.
            
            Args:
                grad: Original gradient tensor.
                
            Returns:
                Modified gradient with zeros at faulty positions.
            """
            if self._current_fault_mask is None:
                return grad
            
            # Zero gradients at faulty weight positions
            # This prevents the optimizer from updating based on faulty outputs
            grad_modified = torch.where(
                self._current_fault_mask,
                torch.zeros_like(grad),
                grad
            )
            
            return grad_modified
        
        return gradient_hook
    
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
        
        # Check if module has weight quantization (Brevitas)
        if not hasattr(module, 'weight_quant'):
            if self.verbose:
                print(f"Layer {self.layer_id}: No weight_quant attribute, skipping")
            return
        
        # Get quantized weight by calling the weight quantizer
        quant_weight = module.weight_quant(module.weight)
        
        # Check if weight is quantized (has bit_width)
        if not hasattr(quant_weight, 'bit_width') or quant_weight.bit_width is None:
            if self.verbose:
                print(f"Layer {self.layer_id}: Weight not quantized, skipping")
            return
        
        # Extract quantization parameters
        scale = quant_weight.scale
        bit_width = int(quant_weight.bit_width.item())
        signed = quant_weight.signed
        zero_point = getattr(quant_weight, 'zero_point', None)
        
        # Generate mask for fault injection
        num_elements = quant_weight.value.numel()
        num_faults = max(1, int(num_elements * self.probability / 100.0))
        
        mask = torch.zeros(num_elements, dtype=torch.bool, device=quant_weight.device)
        fault_indices = torch.randperm(num_elements, device=quant_weight.device)[:num_faults]
        mask.view(-1)[fault_indices] = True
        mask = mask.view(quant_weight.value.shape)
        
        if self.verbose:
            print(
                f"Layer {self.layer_id}: Injecting {num_faults}/{num_elements} "
                f"weight faults ({self.probability:.2f}%)"
            )
        
        # Convert to integer representation
        if zero_point is not None:
            int_weights = torch.round(quant_weight.value / scale) + zero_point
        else:
            int_weights = torch.round(quant_weight.value / scale)
        int_weights = int_weights.to(torch.int32)
        
        # Apply injection strategy
        faulty_int_weights = self.strategy.inject(
            int_tensor=int_weights,
            mask=mask,
            bit_width=bit_width,
            signed=signed,
            device=quant_weight.device,
        )
        
        # Update statistics if tracking (use integer tensors for accurate comparison)
        if self.statistics is not None:
            self.statistics.record(
                clean_int=int_weights,
                faulty_int=faulty_int_weights,
                mask=mask,
                layer_id=self.layer_id,
            )
        
        # Convert back to float
        if zero_point is not None:
            faulty_weights = (faulty_int_weights - zero_point).float() * scale
        else:
            faulty_weights = faulty_int_weights.float() * scale
        
        # Store the fault mask for gradient zeroing during backward pass
        self._current_fault_mask = mask
        
        # Register gradient hook on first call to zero gradients at faulty positions
        if self._grad_hook_handle is None:
            grad_hook_fn = self._create_gradient_hook()
            self._grad_hook_handle = module.weight.register_hook(grad_hook_fn)
        
        # Modify the weight in-place
        # Store original for potential restoration
        if self._original_weight is None:
            self._original_weight = module.weight.data.clone()
        
        # Update weight data (modify the underlying Parameter, not the QuantTensor)
        with torch.no_grad():
            module.weight.data.copy_(faulty_weights)
    
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
                module.weight.data.copy_(self._original_weight)
            self._original_weight = None
    
    def remove_hooks(self) -> None:
        """Remove gradient hooks for cleanup.
        
        Call this when removing the fault injector to clean up gradient hooks.
        """
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
        self._current_fault_mask = None
