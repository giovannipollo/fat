"""Model injector for weight fault injection.

Provides the WeightFaultInjector class that transforms models at runtime
to add fault injection to layer weights (parameters).
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

import torch
import torch.nn as nn

from .base_injector import BaseFaultInjector
from .config import FaultInjectionConfig
from .weights.weight_hooks import WeightFaultInjectionHook
from .statistics import FaultStatistics
from .strategies import get_strategy


class WeightFaultInjector(BaseFaultInjector):
    """Transforms models to add fault injection to weights.
    
    This class analyzes a model's structure and registers forward hooks
    on target layers (QuantConv2d, QuantLinear) that inject faults into
    the weight tensors during forward passes.
    
    The injector supports:
    - Runtime injection via forward hooks
    - Configurable target layers via YAML config
    - Parameter updates (probability, epoch, etc.)
    - Statistics tracking
    - Full removal of hooks
    
    Key Differences from Activation Injection:
    - Targets layer.weight parameters instead of outputs
    - Uses forward_pre_hook to modify weights before computation
    - Faults can be persistent or transient (configurable)
    - Works with weight quantization metadata
    
    Example:
        ```python
        injector = WeightFaultInjector()
        config = FaultInjectionConfig(
            enabled=True,
            target_type="weight",
            probability=2.0
        )
        
        # Inject hooks
        model = injector.inject(model, config)
        
        # Remove hooks
        model = injector.remove(model)
        ```
    
    Attributes:
        WEIGHT_TARGET_LAYERS: Set of layer class names that support weight injection.
    """
    
    WEIGHT_TARGET_LAYERS: Set[str] = {
        "QuantConv2d",
        "QuantLinear",
    }
    
    def __init__(self) -> None:
        """Initialize the weight fault injector."""
        self._layer_count: int = 0
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._injection_hooks: List[WeightFaultInjectionHook] = []
    
    def inject(
        self,
        model: nn.Module,
        config: FaultInjectionConfig,
    ) -> nn.Module:
        """Add weight fault injection hooks to a model.
        
        Walks the model graph and registers forward_pre_hooks on each
        target layer that modify the weight tensor before forward pass.
        
        Args:
            model: The model to transform.
            config: Fault injection configuration (must have target_type="weight").
            
        Returns:
            Transformed model with injection hooks.
            
        Raises:
            ValueError: If config.target_type is not "weight".
        """
        if not config.enabled:
            return model
        
        if config.target_type != "weight":
            raise ValueError(
                f"WeightFaultInjector requires target_type='weight', "
                f"got '{config.target_type}'"
            )
        
        # Reset state
        self._layer_count = 0
        self._hook_handles = []
        self._injection_hooks = []
        self._target_layers = set(config.target_layers)
        self.config = config
        
        if config.verbose:
            print("Injecting faults into weights of target layers")
        
        # Get injection strategy
        strategy = get_strategy(config.injection_type)
        
        # Register hooks on all target layers
        self._inject_recursive(model, config, strategy)
        
        if config.verbose:
            print(f"Registered {len(self._injection_hooks)} weight fault injection hooks")
        
        return model
    
    def _is_target_layer(self, module: nn.Module) -> bool:
        """Check if a module is a target for weight injection.
        
        Args:
            module: Module to check.
            
        Returns:
            True if module is a target layer type.
        """
        target_layers = getattr(self, "_target_layers", self.WEIGHT_TARGET_LAYERS)
        return module.__class__.__name__ in target_layers
    
    def _inject_recursive(
        self,
        module: nn.Module,
        config: FaultInjectionConfig,
        strategy: Any,
        name: str = "",
    ) -> None:
        """Recursively register hooks on target layers.
        
        Args:
            module: Current module to process.
            config: Fault injection configuration.
            strategy: Injection strategy instance.
            name: Name of this module in parent (for verbose logging).
        """
        for child_name, child in module.named_children():
            if self._is_target_layer(child):
                # Check if layer has quantized weights
                if not hasattr(child, 'weight'):
                    if config.verbose:
                        print(f"  Skipping {child_name} - no weight parameter")
                    continue
                
                # Create hook
                hook = WeightFaultInjectionHook(
                    layer_id=self._layer_count,
                    probability=config.probability,
                    strategy=strategy,
                    verbose=config.verbose,
                )
                
                # Register forward_pre_hook
                handle = child.register_forward_pre_hook(hook)
                
                self._injection_hooks.append(hook)
                self._hook_handles.append(handle)
                self._layer_count += 1
                
                if config.verbose:
                    print(f"  Registered hook on {child_name} (layer_id={hook.layer_id})")
            else:
                # Recurse into children
                self._inject_recursive(child, config, strategy, child_name)
    
    def remove(self, model: nn.Module) -> nn.Module:
        """Remove all weight fault injection hooks from a model.
        
        Args:
            model: Model with injection hooks.
            
        Returns:
            Model without injection hooks.
        """
        for handle in self._hook_handles:
            handle.remove()
        
        self._hook_handles = []
        self._injection_hooks = []
        self._layer_count = 0
        
        return model
    
    def update_probability(
        self,
        model: nn.Module,
        probability: float,
        layer_id: Optional[int] = None,
    ) -> None:
        """Update injection probability.
        
        Args:
            model: Model with injection hooks.
            probability: New probability (0-100).
            layer_id: Specific layer to update (None = all layers).
        """
        for hook in self._injection_hooks:
            if layer_id is None or hook.layer_id == layer_id:
                hook.set_probability(probability)
    
    def set_enabled(self, model: nn.Module, enabled: bool) -> None:
        """Enable or disable injection for all hooks.
        
        Args:
            model: Model with injection hooks.
            enabled: Whether to enable injection.
        """
        for hook in self._injection_hooks:
            hook.set_enabled(enabled)
    
    def set_statistics(self, model: nn.Module, statistics: FaultStatistics) -> None:
        """Set statistics tracker for all injection hooks.
        
        Args:
            model: Model with injection hooks.
            statistics: Statistics tracker instance.
        """
        for hook in self._injection_hooks:
            hook.statistics = statistics
    
    def get_num_layers(self, model: nn.Module) -> int:
        """Count the number of injection hooks in a model.
        
        Args:
            model: Model to count.
            
        Returns:
            Number of injection hooks.
        """
        return len(self._injection_hooks)
