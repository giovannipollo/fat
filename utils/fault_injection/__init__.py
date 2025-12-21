"""Fault injection framework for quantized neural networks.

This module provides tools for injecting faults into quantized neural networks
to enable fault-aware training (FAT) and fault resilience evaluation.

Main Components:
    - FaultInjectionConfig: Configuration dataclass for fault injection parameters
    - FaultInjector: Runtime model transformer that adds fault injection layers
    - FaultStatistics: Statistics tracking for injection analysis
    - QuantFaultInjectionLayer: Layer that injects faults into QuantTensor activations

Injection Strategies:
    - RandomStrategy: Adds random values to activations
    - LSBFlipStrategy: Flips least significant bits
    - MSBFlipStrategy: Flips most significant bits
    - FullFlipStrategy: Flips all bits

Example:
    ```python
    from utils.fault_injection import (
        FaultInjectionConfig,
        FaultInjector,
        FaultStatistics,
    )

    # Create configuration
    config = FaultInjectionConfig(
        enabled=True,
        probability=5.0,
        mode="full_model",
        injection_type="random",
        apply_during="train",
    )

    # Inject fault layers into model
    injector = FaultInjector()
    model = injector.inject(model, config)

    # Optional: Track statistics
    stats = FaultStatistics()
    injector.set_statistics(model, stats)

    # Training loop
    for epoch in range(epochs):
        injector.update_epoch(model, epoch)
        injector.reset_counters(model)
        # ... train ...

    # Get statistics report
    stats.print_report()
    ```
"""

from __future__ import annotations

from .config import FaultInjectionConfig
from .functions import FaultInjectionFunction
from .injector import FaultInjector
from .layers import QuantFaultInjectionLayer
from .statistics import FaultStatistics, LayerStatistics
from .strategies import (
    InjectionStrategy,
    RandomStrategy,
    LSBFlipStrategy,
    MSBFlipStrategy,
    FullFlipStrategy,
    get_strategy,
)

__all__ = [
    # Configuration
    "FaultInjectionConfig",
    # Functions
    "FaultInjectionFunction",
    # Injector
    "FaultInjector",
    # Layers
    "QuantFaultInjectionLayer",
    "ErrInjLayer",  # Alias for compatibility
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
