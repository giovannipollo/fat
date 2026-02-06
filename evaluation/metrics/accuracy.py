"""Accuracy metrics for evaluation."""

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