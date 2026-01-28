"""Activation fault injection components.

This submodule provides classes and functions for injecting faults into
neural network activations during training and evaluation.
"""

from __future__ import annotations

from .activation_functions import ActivationFaultInjectionFunction
from .activation_layers import QuantActivationFaultInjectionLayer
from .activation_wrapper import _ActivationFaultInjectionWrapper

__all__ = [
    "ActivationFaultInjectionFunction",
    "QuantActivationFaultInjectionLayer",
    "_ActivationFaultInjectionWrapper",
]