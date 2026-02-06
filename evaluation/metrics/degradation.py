"""Degradation metrics for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DegradationMetrics:
    """Container for accuracy degradation metrics.

    Attributes:
        baseline_accuracy: Baseline (no-fault) accuracy.
        fault_accuracy: Accuracy with faults.
        absolute_degradation: Absolute degradation (baseline - fault).
        relative_degradation: Relative degradation percentage.
    """

    baseline_accuracy: float
    fault_accuracy: float
    absolute_degradation: float
    relative_degradation: float

    @classmethod
    def calculate(cls, baseline: float, fault: float) -> "DegradationMetrics":
        """Calculate degradation metrics from baseline and fault accuracy.

        Args:
            baseline: Baseline accuracy percentage.
            fault: Fault accuracy percentage.

        Returns:
            DegradationMetrics instance.
        """
        abs_degr, rel_degr = calculate_degradation(baseline, fault)

        return cls(
            baseline_accuracy=baseline,
            fault_accuracy=fault,
            absolute_degradation=abs_degr,
            relative_degradation=rel_degr,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "fault_accuracy": self.fault_accuracy,
            "absolute_degradation": self.absolute_degradation,
            "relative_degradation": self.relative_degradation,
        }


def calculate_degradation(baseline: float, fault: float) -> tuple[float, float]:
    """Calculate absolute and relative degradation.

    Args:
        baseline: Baseline accuracy percentage.
        fault: Fault accuracy percentage.

    Returns:
        Tuple of (absolute_degradation, relative_degradation) where:
            absolute_degradation: baseline - fault (can be negative if fault > baseline)
            relative_degradation: (baseline - fault) / baseline * 100 (percentage)
    """
    absolute = baseline - fault
    relative = (absolute / baseline * 100.0) if baseline > 0 else 0.0

    return absolute, relative