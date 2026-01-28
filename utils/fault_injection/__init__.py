"""Fault injection framework for quantized neural networks.

This module provides tools for injecting activation and weight faults into quantized
neural networks to enable fault-aware training (FAT) and fault resilience evaluation.

Main Components:
    - FaultInjectionConfig: Configuration dataclass for fault injection parameters (supports activation and weight)
    - ActivationFaultInjector: Runtime model transformer that adds activation fault injection layers
    - FaultInjector: Backward compatibility alias for ActivationFaultInjector (deprecated)
    - FaultStatistics: Statistics tracking for injection analysis
    - QuantActivationFaultInjectionLayer: Layer that injects activation faults into QuantTensor activations

Injection Strategies:
    - RandomStrategy: Adds random values to activations/weights
    - LSBFlipStrategy: Flips least significant bits
    - MSBFlipStrategy: Flips most significant bits
    - FullFlipStrategy: Flips all bits

Example:
    ```python
    from utils.fault_injection import (
        FaultInjectionConfig,
        ActivationFaultInjector,
        FaultStatistics,
    )

    # Create configuration
    config = FaultInjectionConfig(
        enabled=True,
        target_type="activation",
        probability=5.0,
        injection_type="random",
        apply_during="train",
    )

    # Inject activation fault layers into model
    injector = ActivationFaultInjector()
    model = injector.inject(model, config)

    # Optional: Track statistics
    stats = FaultStatistics()
    injector.set_statistics(model, stats)

    # Training loop
    for epoch in range(epochs):
        # ... train ...

    # Get statistics report
    stats.print_report()
    ```
"""

from __future__ import annotations

from .base_injector import BaseFaultInjector
from .config import FaultInjectionConfig
from .activations.activation_functions import ActivationFaultInjectionFunction
from .activation_injector import ActivationFaultInjector
from .activations.activation_layers import QuantActivationFaultInjectionLayer
from .statistics import FaultStatistics, LayerStatistics
from .strategies import (
    InjectionStrategy,
    RandomStrategy,
    LSBFlipStrategy,
    MSBFlipStrategy,
    FullFlipStrategy,
    get_strategy,
)

# Backward compatibility alias (deprecated)
FaultInjector = ActivationFaultInjector

__all__ = [
    # Configuration
    "FaultInjectionConfig",
    # Base classes
    "BaseFaultInjector",
    # Functions
    "ActivationFaultInjectionFunction",
    # Injectors
    "ActivationFaultInjector",
    "FaultInjector",  # Backward compatibility (deprecated)
    # Layers
    "QuantActivationFaultInjectionLayer",
    # Statistics
    "FaultStatistics",
    "LayerStatistics",
    # Strategies
    "InjectionStrategy",
    "RandomStrategy",
    "LSBFlipStrategy",
    "MSBFlipStrategy",
    "FullFlipStrategy",
    "get_strategy",
]
