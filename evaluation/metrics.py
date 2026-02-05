"""Utility functions for calculating evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AccuracyMetrics:
    """Container for accuracy metrics.

    Attributes:
        accuracy: Accuracy percentage (0-100).
        correct: Number of correct predictions.
        total: Total number of samples.
        accuracies: List of accuracies from multiple runs.
        mean: Mean accuracy across runs.
        std: Standard deviation across runs.
    """

    accuracy: float
    correct: int
    total: int
    accuracies: List[float] = field(default_factory=list)
    mean: Optional[float] = None
    std: Optional[float] = None

    def __post_init__(self) -> None:
        """Calculate mean and std if not provided."""
        if not self.accuracies:
            self.accuracies = [self.accuracy]

        if self.mean is None:
            self.mean, self.std = aggregate_accuracies(self.accuracies)

    @classmethod
    def from_runs(cls, runs: List[Tuple[int, int]]) -> "AccuracyMetrics":
        """Create metrics from multiple evaluation runs.

        Args:
            runs: List of (correct, total) tuples from multiple runs.

        Returns:
            AccuracyMetrics instance with aggregated statistics.
        """
        accuracies = []
        total_correct = 0
        total_samples = 0

        for correct, total in runs:
            acc = calculate_accuracy(correct, total)
            accuracies.append(acc)
            total_correct += correct
            total_samples += total

        overall_acc = calculate_accuracy(total_correct, total_samples)
        mean, std = aggregate_accuracies(accuracies)

        return cls(
            accuracy=overall_acc,
            correct=total_correct,
            total=total_samples,
            accuracies=accuracies,
            mean=mean,
            std=std,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "accuracies": self.accuracies,
            "mean": self.mean,
            "std": self.std,
        }


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
    def calculate(
        cls, baseline: float, fault: float
    ) -> "DegradationMetrics":
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


def calculate_accuracy(correct: int, total: int) -> float:
    """Calculate accuracy percentage.

    Args:
        correct: Number of correct predictions.
        total: Total number of predictions.

    Returns:
        Accuracy as percentage (0-100).

    Raises:
        ValueError: If total is zero.
    """
    if total == 0:
        raise ValueError("Cannot calculate accuracy with zero total samples")

    return (correct / total) * 100.0


def aggregate_accuracies(
    accuracies: List[float],
) -> Tuple[float, Optional[float]]:
    """Calculate mean and std of accuracies.

    Args:
        accuracies: List of accuracy percentages.

    Returns:
        Tuple of (mean, std) where std is None if only one value.
    """
    import statistics

    mean = statistics.mean(accuracies)
    std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0

    return mean, std


def calculate_degradation(
    baseline: float, fault: float
) -> Tuple[float, float]:
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
