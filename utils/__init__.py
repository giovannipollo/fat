"""Training utilities module for the PyTorch training framework.

Exports core utilities and modular components for training,
including device management, configuration loading, reproducibility,
and the main Trainer class.
"""

from .device import get_device
from .config import load_config
from .seed import set_seed
from .trainer import Trainer

# Modular components
from .optimizer import OptimizerFactory
from .scheduler import SchedulerFactory, WarmupScheduler
from .experiment import ExperimentManager
from .logging import MetricsLogger
from .loss import LossFactory

# Fault injection framework
from .fault_injection import (
    FaultInjectionConfig,
    ActivationFaultInjector,
    FaultStatistics,
    QuantActivationFaultInjectionLayer
)

__all__ = [
    # Core utilities
    "get_device",
    "load_config",
    "set_seed",
    "Trainer",
    # Modular components
    "OptimizerFactory",
    "SchedulerFactory",
    "WarmupScheduler",
    "ExperimentManager",
    "MetricsLogger",
    "LossFactory",
    # Fault injection
    "FaultInjectionConfig",
    "ActivationFaultInjector",
    "FaultStatistics",
    "QuantActivationFaultInjectionLayer",
]
