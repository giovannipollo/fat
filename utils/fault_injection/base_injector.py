"""Base interface for fault injectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn

from .config import FaultInjectionConfig
from .statistics import FaultStatistics


class BaseFaultInjector(ABC):
    """Abstract base class for fault injectors.

    Defines the common interface for activation and weight injectors.

    All injector implementations must inherit from this class and implement
    the abstract methods defined below.
    """

    @abstractmethod
    def inject(
        self,
        model: nn.Module,
        config: FaultInjectionConfig,
    ) -> nn.Module:
        """Add fault injection to a model.

        Args:
            model: The model to transform.
            config: Fault injection configuration.

        Returns:
            Transformed model with fault injection.
        """
        pass

    @abstractmethod
    def remove(self, model: nn.Module) -> nn.Module:
        """Remove fault injection from a model.

        Args:
            model: Model with fault injection.

        Returns:
            Model without fault injection.
        """
        pass

    @abstractmethod
    def update_probability(
        self,
        model: nn.Module,
        probability: float,
        layer_id: Optional[int] = None,
    ) -> None:
        """Update injection probability.

        Args:
            model: Model with fault injection.
            probability: New probability (0-100).
            layer_id: Specific layer to update (None = all layers).
        """
        pass

    @abstractmethod
    def set_enabled(self, model: nn.Module, enabled: bool) -> None:
        """Enable or disable injection.

        Args:
            model: Model with fault injection.
            enabled: Whether to enable injection.
        """
        pass

    @abstractmethod
    def set_statistics(
        self,
        model: nn.Module,
        statistics: FaultStatistics,
    ) -> None:
        """Set statistics tracker.

        Args:
            model: Model with fault injection.
            statistics: Statistics tracker instance.
        """
        pass

    @abstractmethod
    def get_num_layers(self, model: nn.Module) -> int:
        """Count injection layers.

        Args:
            model: Model to count.

        Returns:
            Number of injection layers/hooks.
        """
        pass
