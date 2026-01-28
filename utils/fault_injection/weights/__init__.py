"""Weight fault injection components.

This submodule provides classes and functions for injecting faults into
neural network weights during training and evaluation.
"""

from __future__ import annotations

from .weight_hooks import WeightFaultInjectionHook
from .weight_functions import WeightFaultInjectionFunction

__all__ = [
    "WeightFaultInjectionHook",
    "WeightFaultInjectionFunction",
]