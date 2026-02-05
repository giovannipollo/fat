"""Abstract base class for evaluation reporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseReporter(ABC):
    """Abstract base class for evaluation reporters.

    Reporters handle formatting and saving evaluation results
    in different formats (console, JSON, CSV, etc.).

    Attributes:
        verbose: Enable verbose output.
        show_progress: Show progress indicators.
    """

    def __init__(self, verbose: bool = True, show_progress: bool = True, **kwargs):
        """Initialize reporter.

        Args:
            verbose: Enable verbose output.
            show_progress: Show progress bars/indicators.
            **kwargs: Additional arguments (e.g., save_path for JSONReporter).
        """
        self.verbose = verbose
        self.show_progress = show_progress

    @abstractmethod
    def report(self, results: Dict[str, Any]) -> None:
        """Report evaluation results.

        Args:
            results: Dictionary with evaluation results from runner.
        """
        pass

    def _format_percentage(
        self, value: float, decimals: int = 2, show_sign: bool = False
    ) -> str:
        """Format percentage value.

        Args:
            value: Percentage value.
            decimals: Number of decimal places.
            show_sign: Always show +/- sign.

        Returns:
            Formatted string.
        """
        fmt = f"{{:+.{decimals}f}}" if show_sign else f"{{:.{decimals}f}}"
        return fmt.format(value) + "%"

    def _format_metric(
        self, mean: float, std: Optional[float] = None, decimals: int = 2
    ) -> str:
        """Format metric with optional std deviation.

        Args:
            mean: Mean value.
            std: Standard deviation (optional).
            decimals: Number of decimal places.

        Returns:
            Formatted string like "85.42%" or "85.42% ± 1.23%".
        """
        mean_str = self._format_percentage(mean, decimals)
        if std is not None and std > 0:
            std_str = self._format_percentage(std, decimals)
            return f"{mean_str} ± {std_str}"
        return mean_str


def get_reporters(formats: list[str], **kwargs) -> list[BaseReporter]:
    """Factory function to create reporters for requested formats.

    Args:
        formats: List of format names ("console", "json", "csv").
        **kwargs: Additional arguments for reporters.

    Returns:
        List of reporter instances.

    Raises:
        ValueError: If format is unknown.
    """
    from .console import ConsoleReporter
    from .json_reporter import JSONReporter

    reporter_map = {
        "console": ConsoleReporter,
        "json": JSONReporter,
    }

    reporters = []
    for fmt in formats:
        if fmt not in reporter_map:
            raise ValueError(
                f"Unknown reporter format '{fmt}'. "
                f"Must be one of {list(reporter_map.keys())}"
            )
        reporters.append(reporter_map[fmt](**kwargs))

    return reporters
