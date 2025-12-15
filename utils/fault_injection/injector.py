"""Model injector for fault injection.

Provides the FaultInjector class that transforms models at runtime
to add fault injection layers after target quantized layers.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor

from .config import FaultInjectionConfig
from .layers import QuantFaultInjectionLayer
from .statistics import FaultStatistics
from .strategies import get_strategy


class FaultInjector:
    """Transforms models to add fault injection layers at runtime.

    This class analyzes a model's structure and inserts fault injection
    layers after target layers (QuantIdentity, QuantReLU, etc.) without
    modifying the original model definition.

    The injector supports:
    - Runtime insertion of fault injection layers
    - Parameter updates (probability, epoch, etc.)
    - Statistics tracking
    - Full removal of injection layers

    Example:
        ```python
        injector = FaultInjector()
        config = FaultInjectionConfig(enabled=True, probability=5.0)

        # Inject layers
        model = injector.inject(model, config)

        # Update parameters per epoch
        injector.update_epoch(model, epoch=10)

        # Reset counters for new epoch
        injector.reset_counters(model)

        # Remove injection layers
        model = injector.remove(model)
        ```

    Attributes:
        QUANT_TARGET_LAYERS: Set of layer class names to inject after.
    """

    # Layer types to inject after (Brevitas quantized activation layers)
    QUANT_TARGET_LAYERS: Set[str] = {
        "QuantIdentity",
        "QuantReLU",
        "QuantHardTanh",
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

        # First pass: count target layers
        total_layers = self._count_target_layers(model)

        # Determine injection layer for "layer" mode
        if config.mode == "layer":
            if config.injection_layer == -1:
                # Random layer selection
                injection_layer = random.randint(0, total_layers - 1)
                if config.verbose:
                    print(f"Randomly selected layer {injection_layer} for injection")
            else:
                injection_layer = config.injection_layer
        else:
            # Full model: use total_layers as marker
            injection_layer = total_layers

        # Get injection strategy
        strategy = get_strategy(config.injection_type)

        # Second pass: inject layers
        self._inject_recursive(
            model,
            config=config,
            injection_layer=injection_layer,
            total_layers=total_layers,
            strategy=strategy,
        )

        if config.verbose:
            print(f"Injected {len(self._injection_layers)} fault injection layers")

        return model

    def _count_target_layers(self, module: nn.Module) -> int:
        """Count the number of target layers in the model.

        Args:
            module: Module to search.

        Returns:
            Number of target layers found.
        """
        count = 0
        for child in module.modules():
            if child.__class__.__name__ in self.QUANT_TARGET_LAYERS:
                count += 1
        return count

    def _inject_recursive(
        self,
        module: nn.Module,
        config: FaultInjectionConfig,
        injection_layer: int,
        total_layers: int,
        strategy: Any,
        parent: Optional[nn.Module] = None,
        name: str = "",
    ) -> None:
        """Recursively inject fault layers into a module.

        Args:
            module: Current module to process.
            config: Fault injection configuration.
            injection_layer: Target layer index or total_layers for full model.
            total_layers: Total number of injection layers.
            strategy: Injection strategy instance.
            parent: Parent module (for attribute replacement).
            name: Name of this module in parent.
        """
        # Handle Sequential modules specially
        if isinstance(module, nn.Sequential):
            self._inject_sequential(
                module,
                config=config,
                injection_layer=injection_layer,
                total_layers=total_layers,
                strategy=strategy,
            )
            return

        # Handle ModuleList
        if isinstance(module, nn.ModuleList):
            for i, child in enumerate(module):
                self._inject_recursive(
                    child,
                    config=config,
                    injection_layer=injection_layer,
                    total_layers=total_layers,
                    strategy=strategy,
                    parent=module,
                    name=str(i),
                )
            return

        # Process named children
        children_to_process = list(module.named_children())
        for child_name, child in children_to_process:
            # Check if this child is a target layer
            if child.__class__.__name__ in self.QUANT_TARGET_LAYERS:
                # Create injection layer
                inj_layer = QuantFaultInjectionLayer(
                    layer_id=self._layer_count,
                    probability=config.probability,
                    injection_layer=injection_layer,
                    num_layers=total_layers,
                    strategy=strategy,
                    mode=config.mode,
                    epoch_interval=config.epoch_interval,
                    step_interval=config.step_interval,
                    verbose=config.verbose,
                    hw_mask=config.hw_mask,
                    frequency_value=config.frequency_value,
                    seed=config.seed,
                    gradient_mode=config.gradient_mode,
                )
                self._injection_layers.append(inj_layer)

                # Wrap the target layer with injection
                wrapper = _InjectionWrapper(child, inj_layer)
                setattr(module, child_name, wrapper)

                self._layer_count += 1
            else:
                # Recurse into child
                self._inject_recursive(
                    child,
                    config=config,
                    injection_layer=injection_layer,
                    total_layers=total_layers,
                    strategy=strategy,
                    parent=module,
                    name=child_name,
                )

    def _inject_sequential(
        self,
        module: nn.Sequential,
        config: FaultInjectionConfig,
        injection_layer: int,
        total_layers: int,
        strategy: Any,
    ) -> None:
        """Inject into a Sequential module.

        Args:
            module: Sequential module to transform.
            config: Fault injection configuration.
            injection_layer: Target layer index or total_layers for full model.
            total_layers: Total number of injection layers.
            strategy: Injection strategy instance.
        """
        new_modules: List[Tuple[str, nn.Module]] = []

        for name, child in module.named_children():
            # First, recurse into the child if it has children
            if len(list(child.children())) > 0:
                self._inject_recursive(
                    child,
                    config=config,
                    injection_layer=injection_layer,
                    total_layers=total_layers,
                    strategy=strategy,
                )

            new_modules.append((name, child))

            # Check if we should inject after this layer
            if child.__class__.__name__ in self.QUANT_TARGET_LAYERS:
                inj_layer = QuantFaultInjectionLayer(
                    layer_id=self._layer_count,
                    probability=config.probability,
                    injection_layer=injection_layer,
                    num_layers=total_layers,
                    strategy=strategy,
                    mode=config.mode,
                    epoch_interval=config.epoch_interval,
                    step_interval=config.step_interval,
                    verbose=config.verbose,
                    hw_mask=config.hw_mask,
                    frequency_value=config.frequency_value,
                    seed=config.seed,
                    gradient_mode=config.gradient_mode,
                )
                self._injection_layers.append(inj_layer)
                new_modules.append((f"fault_inj_{self._layer_count}", inj_layer))
                self._layer_count += 1

        # Replace Sequential contents
        module._modules = OrderedDict(new_modules)

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
        """Recursively remove fault injection layers.

        Args:
            module: Module to process.
        """
        # Handle Sequential modules
        if isinstance(module, nn.Sequential):
            # Filter out injection layers
            new_modules = [
                (name, child)
                for name, child in module.named_children()
                if not isinstance(child, QuantFaultInjectionLayer)
            ]
            module._modules = OrderedDict(new_modules)

        # Handle wrapped layers
        for name, child in list(module.named_children()):
            if isinstance(child, _InjectionWrapper):
                # Unwrap the original layer
                setattr(module, name, child.wrapped_layer)
            elif isinstance(child, QuantFaultInjectionLayer):
                # Skip (handled in Sequential case)
                pass
            else:
                # Recurse
                self._remove_recursive(child)

    def update_epoch(self, model: nn.Module, epoch: int) -> None:
        """Update epoch for all injection layers.

        Args:
            model: Model with injection layers.
            epoch: Current epoch number.
        """
        for layer in self._get_injection_layers(model):
            layer.set_epoch(epoch)

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

    def reset_counters(self, model: nn.Module) -> None:
        """Reset iteration counters for all injection layers.

        Args:
            model: Model with injection layers.
        """
        for layer in self._get_injection_layers(model):
            layer.reset_counter()

    def set_mode(self, model: nn.Module, mode: str) -> None:
        """Set usage mode (train/eval) for all injection layers.

        Args:
            model: Model with injection layers.
            mode: Either "train" or "eval".
        """
        for layer in self._get_injection_layers(model):
            layer.set_use_mode(mode)

    def set_enabled(self, model: nn.Module, enabled: bool) -> None:
        """Enable or disable injection for all layers.

        Args:
            model: Model with injection layers.
            enabled: Whether to enable injection.
        """
        for layer in self._get_injection_layers(model):
            layer.set_enabled(enabled)

    def set_statistics(
        self, model: nn.Module, statistics: FaultStatistics
    ) -> None:
        """Set statistics tracker for all injection layers.

        Args:
            model: Model with injection layers.
            statistics: Statistics tracker instance.
        """
        for layer in self._get_injection_layers(model):
            layer.statistics = statistics

    def set_condition_injector(
        self, model: nn.Module, num_iterations: int, step_interval: float
    ) -> None:
        """Set up the condition injector for training mode.

        Pre-computes which iterations should have injection enabled
        based on the step_interval probability.

        Args:
            model: Model with injection layers.
            num_iterations: Number of iterations per epoch.
            step_interval: Probability of injection per step (0-1).
        """
        # Generate random schedule
        injector = torch.rand(num_iterations)
        condition_injector = injector < step_interval

        for layer in self._get_injection_layers(model):
            layer.set_condition_injector(condition_injector)
            layer.set_num_iterations(num_iterations)

    def get_num_layers(self, model: nn.Module) -> int:
        """Count the number of injection layers in a model.

        Args:
            model: Model to count.

        Returns:
            Number of injection layers.
        """
        return len(self._get_injection_layers(model))

    def _get_injection_layers(
        self, model: nn.Module
    ) -> List[QuantFaultInjectionLayer]:
        """Get all injection layer instances from a model.

        Args:
            model: Model to search.

        Returns:
            List of injection layer instances.
        """
        layers = []
        for module in model.modules():
            if isinstance(module, QuantFaultInjectionLayer):
                layers.append(module)
            elif isinstance(module, _InjectionWrapper):
                layers.append(module.injection_layer)
        return layers


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
