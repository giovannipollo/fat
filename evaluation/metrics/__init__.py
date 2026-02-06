"""Evaluation metrics for fault injection experiments.

This module provides classes and utilities for calculating and managing
evaluation metrics including accuracy, degradation, and formatting utilities.
"""

from __future__ import annotations

from .accuracy import AccuracyMetrics, calculate_accuracy, aggregate_accuracies
from .degradation import DegradationMetrics, calculate_degradation
from .formatting import format_accuracy, format_degradation

__all__ = [
    # Classes
    "AccuracyMetrics",
    "DegradationMetrics",
    # Functions
    "calculate_accuracy",
    "aggregate_accuracies",
    "calculate_degradation",
    "format_accuracy",
    "format_degradation",
]