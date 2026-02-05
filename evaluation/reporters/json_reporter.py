"""Reporter for JSON output."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from .base import BaseReporter


class JSONReporter(BaseReporter):
    """Reporter for JSON output.

    Saves evaluation results as JSON files with optional
    formatting and timestamp-based naming.

    Attributes:
        save_path: Path template for saving JSON.
        indent: JSON indentation (None for compact).
    """

    def __init__(
        self, save_path: Optional[str] = None, indent: int = 2, **kwargs
    ):
        """Initialize JSON reporter.

        Args:
            save_path: Path template for JSON file (supports {timestamp}, {name}).
            indent: JSON indentation (None for compact, 2-4 for readable).
            **kwargs: Additional arguments for BaseReporter.
        """
        super().__init__(**kwargs)
        self.save_path = save_path
        self.indent = indent

    def report(self, results: Dict[str, Any]) -> None:
        """Report evaluation results to JSON file.

        Args:
            results: Dictionary with evaluation results.
        """
        results_with_meta = self._add_metadata(results)

        save_path = self._get_save_path(results)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results_with_meta, f, indent=self.indent)

        if self.verbose:
            print(f"\nResults saved to: {save_path}")

    def _add_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to results.

        Args:
            results: Original results dictionary.

        Returns:
            Results with metadata added.
        """
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "reporter": "JSONReporter",
            },
            **results,
        }

    def _get_save_path(self, results: Dict[str, Any]) -> str:
        """Determine save path with template substitution.

        Args:
            results: Results dictionary.

        Returns:
            Resolved file path.
        """
        if self.save_path is None:
            exp_name = results.get("experiment_name", "evaluation")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"./results/{exp_name}_{timestamp}.json"

        path = self.save_path
        path = path.replace("{timestamp}", datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = path.replace("{name}", results.get("experiment_name", "evaluation"))

        if not path.endswith(".json"):
            path += ".json"

        return path
