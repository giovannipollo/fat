"""Configuration loading utilities.

Provides functions for loading YAML configuration files
used to specify training hyperparameters, model architecture, etc.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def validate_config_structure(config: Dict[str, Any]) -> None:
    """Validate that the config has the required structure.

    The 'phases' key is mandatory. Flat configs without 'phases' are not
    supported — they must be migrated to the phases format.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If 'phases' key is missing, not a list, or empty.
    """
    if "phases" not in config:
        raise ValueError(
            "Config must contain a 'phases' key with a list of training phases. "
            "Flat configs (with top-level 'optimizer', 'training', etc.) are no "
            "longer supported."
        )

    if not isinstance(config["phases"], list):
        raise ValueError("'phases' must be a list")

    if len(config["phases"]) == 0:
        raise ValueError("'phases' must contain at least one phase")

    ignored_keys = [
        "optimizer",
        "scheduler",
        "training",
        "loss",
        "activation_fault_injection",
        "weight_fault_injection",
    ]
    found = [k for k in ignored_keys if k in config]
    if found:
        warnings.warn(
            f"Top-level keys {found} are ignored when using 'phases'. "
            f"All training parameters must be defined inside each phase.",
            UserWarning,
        )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    validate_config_structure(config)

    return config
