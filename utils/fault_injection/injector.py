"""Model injector for fault injection.

Provides the FaultInjector class that transforms models at runtime
to add fault injection layers after target quantized layers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set

import torch.nn as nn

from .config import FaultInjectionConfig
from .layers import QuantFaultInjectionLayer
from .statistics import FaultStatistics
from .strategies import get_strategy
from .wrapper import _ActivationFaultInjectionWrapper


class FaultInjector:
    """Transforms models to add fault injection layers at runtime.

    This class analyzes a model's structure and inserts fault injection
    layers after target layers (QuantIdentity, QuantReLU, QuantConv2d, etc.)
    without modifying the original model definition.

    The injector supports:
    - Runtime insertion of fault injection layers
    - Configurable target layers via YAML config
    - Parameter updates (probability, epoch, etc.)
    - Statistics tracking
    - Full removal of injection layers

    Example:
        ```python
        injector = FaultInjector()
        config = FaultInjectionConfig(enabled=True, probability=5.0)

        # Inject layers
        model = injector.inject(model, config)

        # Remove injection layers
        model = injector.remove(model)
        ```

    Attributes:
        QUANT_TARGET_LAYERS: Set of layer class names to inject after (default).
            Can be overridden via config.target_layers.
    """

    # Layer types to inject after (Brevitas quantized layers)
    QUANT_TARGET_LAYERS: Set[str] = {
        "QuantIdentity",
        "QuantReLU",
        "QuantHardTanh",
        "QuantConv2d",
    }

    def __init__(self) -> None:
        """Initialize the fault injector."""
        self._layer_count: int = 0
        self._injection_layers: List[QuantFaultInjectionLayer] = []

    def inject(
        self,
        model: nn.Module,
        config: FaultInjectionConfig,
    ) -> nn.Module:
        """Add fault injection layers to a model.

        Walks the model graph and inserts QuantFaultInjectionLayer
        instances after each target layer.

        Args:
            model: The model to transform.
            config: Fault injection configuration.

        Returns:
            Transformed model with injection layers.

        Note:
            The model is modified in-place. The returned model is
            the same object as the input.
        """
        if not config.enabled:
            return model

        # Reset state
        self._layer_count = 0
        self._injection_layers = []
        self._target_layers = set(config.target_layers)
        self.config = config

        # Always inject into all layers (full model)
        if config.verbose:
            print("Injecting faults into all target layers in the model")

        # Get injection strategy
        strategy = get_strategy(config.injection_type)

        # Second pass: inject layers
        self._inject_recursive(
            model,
            config=config,
            strategy=strategy,
        )

        if config.verbose:
            print(f"Injected {len(self._injection_layers)} fault injection layers")

        return model

    def _is_target_layer(self, module: nn.Module) -> bool:
        """Check if a module is a target for injection.

        Args:
            module: Module to check.

        Returns:
            True if module is a target layer type.
        """
        target_layers = getattr(self, "_target_layers", self.QUANT_TARGET_LAYERS)
        return module.__class__.__name__ in target_layers

    def _create_injection_layer(
        self,
        config: FaultInjectionConfig,
        strategy: Any,
    ) -> QuantFaultInjectionLayer:
        """Create a new injection layer and track it.

        Args:
            config: Fault injection configuration.
            strategy: Injection strategy instance.

        Returns:
            New QuantFaultInjectionLayer instance.
        """
        layer = QuantFaultInjectionLayer(
            layer_id=self._layer_count,
            probability=config.probability,
            strategy=strategy,
            verbose=config.verbose,
        )
        self._injection_layers.append(layer)
        self._layer_count += 1
        return layer

    def _set_child(
        self,
        module: nn.Module,
        name: str,
        child: nn.Module,
    ) -> None:
        """Set a child module, handling different container types.

        Args:
            module: Parent module.
            name: Name of the child (or index string for ModuleList).
            child: New child module to set.
        """
        if isinstance(module, nn.ModuleList):
            module[int(name)] = child
        else:
            setattr(module, name, child)

    def _inject_recursive(
        self,
        module: nn.Module,
        config: FaultInjectionConfig,
        strategy: Any,
        name: str = "",
    ) -> None:
        """Recursively inject fault layers by wrapping target children.

        Uses a unified approach for all module types (Sequential, ModuleList,
        and regular modules) by wrapping target layers with _ActivationFaultInjectionWrapper.

        Args:
            module: Current module to process.
            config: Fault injection configuration.
            strategy: Injection strategy instance.
            name: Name of this module in parent (for verbose logging).
        """
        if config.verbose:
            name_str = f", name: {name}" if name else ""
            print(f"Processing module: {module.__class__.__name__}{name_str}")

        for child_name, child in list(module.named_children()):
            if config.verbose:
                print(f"  Child: {child_name} ({child.__class__.__name__})")

            if self._is_target_layer(child):
                if config.verbose:
                    print(f"    Found target layer, wrapping with injection")

                inj_layer = self._create_injection_layer(config, strategy)
                wrapper = _ActivationFaultInjectionWrapper(child, inj_layer)
                self._set_child(module, child_name, wrapper)

                if config.verbose:
                    print(f"    Created injection layer {inj_layer.layer_id}")
            else:
                # Recurse into non-target children
                self._inject_recursive(child, config, strategy, child_name)

    def remove(self, model: nn.Module) -> nn.Module:
        """Remove all fault injection layers from a model.

        Args:
            model: Model with injection layers.

        Returns:
            Model without injection layers.
        """
        self._remove_recursive(model)
        self._injection_layers = []
        return model

    def _remove_recursive(self, module: nn.Module) -> None:
        """Recursively unwrap all injection wrappers.

        Args:
            module: Module to process.
        """
        for name, child in list(module.named_children()):
            if isinstance(child, _ActivationFaultInjectionWrapper):
                # Unwrap the original layer
                self._set_child(module, name, child.wrapped_layer)
            else:
                # Recurse into children
                self._remove_recursive(child)

    def update_probability(
        self,
        model: nn.Module,
        probability: float,
        layer_id: Optional[int] = None,
    ) -> None:
        """Update injection probability.

        Args:
            model: Model with injection layers.
            probability: New probability (0-100).
            layer_id: Specific layer to update (None = all layers).
        """
        for layer in self._get_injection_layers(model):
            if layer_id is None or layer.layer_id == layer_id:
                layer.set_probability(probability)

    def set_enabled(self, model: nn.Module, enabled: bool) -> None:
        """Enable or disable injection for all layers.

        Args:
            model: Model with injection layers.
            enabled: Whether to enable injection.
        """
        for layer in self._get_injection_layers(model):
            layer.set_enabled(enabled)

    def set_statistics(self, model: nn.Module, statistics: FaultStatistics) -> None:
        """Set statistics tracker for all injection layers.

        Args:
            model: Model with injection layers.
            statistics: Statistics tracker instance.
        """
        for layer in self._get_injection_layers(model):
            layer.statistics = statistics

    def get_num_layers(self, model: nn.Module) -> int:
        """Count the number of injection layers in a model.

        Args:
            model: Model to count.

        Returns:
            Number of injection layers.
        """
        return len(self._get_injection_layers(model))

    def _get_injection_layers(self, model: nn.Module) -> List[QuantFaultInjectionLayer]:
        """Get all injection layer instances from a model.

        Args:
            model: Model to search.

        Returns:
            List of injection layer instances.
        """
        layers = []
        seen_ids = set()
        for module in model.modules():
            if isinstance(module, QuantFaultInjectionLayer):
                if id(module) not in seen_ids:
                    layers.append(module)
                    seen_ids.add(id(module))
            elif isinstance(module, _ActivationFaultInjectionWrapper):
                if id(module.activation_injection_layer) not in seen_ids:
                    layers.append(module.activation_injection_layer)
                    seen_ids.add(id(module.activation_injection_layer))
        return layers



