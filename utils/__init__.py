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

# Multi-phase training
from .phase_config import PhaseConfig
from .phase_manager import PhaseManager, create_phase_manager
from .config_validator import ConfigValidator, validate_config, ConfigValidationError

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
    # Multi-phase training
    "PhaseConfig",
    "PhaseManager",
    "create_phase_manager",
    # Configuration validation
    "ConfigValidator",
    "validate_config",
    "ConfigValidationError",
]
