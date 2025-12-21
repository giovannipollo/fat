"""Fault injection strategies.

This module contains different strategies for injecting faults into quantized tensors.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import InjectionStrategy
from .full_flip import FullFlipStrategy
from .lsb_flip import LSBFlipStrategy
from .msb_flip import MSBFlipStrategy
from .random import RandomStrategy

# Registry of available strategies
_STRATEGIES: Dict[str, Type[InjectionStrategy]] = {
    "random": RandomStrategy,
    "lsb_flip": LSBFlipStrategy,
    "msb_flip": MSBFlipStrategy,
    "full_flip": FullFlipStrategy,
}


def get_strategy(name: str) -> InjectionStrategy:
    """Factory function to get strategy by name.

    Args:
        name: Strategy name ("random", "lsb_flip", "msb_flip", "full_flip").

    Returns:
        Instance of the requested strategy.

    Raises:
        ValueError: If strategy name is not recognized.

    Example:
        ```python
        strategy = get_strategy("random")
        ```
    """
    if name not in _STRATEGIES:
        available = list(_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: '{name}'. Available: {available}")

    return _STRATEGIES[name]()


def list_strategies() -> list[str]:
    """List all available strategy names.

    Returns:
        List of strategy names.
    """
    return list(_STRATEGIES.keys())


__all__ = [
    "InjectionStrategy",
    "RandomStrategy",
    "LSBFlipStrategy",
    "MSBFlipStrategy",
    "FullFlipStrategy",
    "get_strategy",
    "list_strategies",
]