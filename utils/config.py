"""Configuration loading utilities.

Provides functions for loading YAML configuration files
used to specify training hyperparameters, model architecture, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


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

    return config
