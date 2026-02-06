"""Formatting utilities for metrics display."""

from __future__ import annotations

from typing import Optional


def format_accuracy(
    accuracy: float, std: Optional[float] = None, decimals: int = 2
) -> str:
    """Format accuracy for display.

    Args:
        accuracy: Accuracy percentage.
        std: Standard deviation (optional).
        decimals: Number of decimal places.

    Returns:
        Formatted string like "85.42%" or "85.42% ± 1.23%".
    """
    if std is not None and std > 0:
        return f"{accuracy:.{decimals}f}% ± {std:.{decimals}f}%"
    return f"{accuracy:.{decimals}f}%"


def format_degradation(degradation: float, decimals: int = 2) -> str:
    """Format degradation for display.

    Args:
        degradation: Degradation value (can be negative or positive).
        decimals: Number of decimal places.

    Returns:
        Formatted string like "-4.22%" or "+0.00%".
    """
    return f"{degradation:+.{decimals}f}%"