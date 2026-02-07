"""Runner for single evaluation with fixed configuration."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseRunner
from ..metrics.formatting import format_accuracy


class SingleRunner(BaseRunner):
    """Runner for single fault injection evaluation.

    Executes:
        1. Baseline evaluation (if enabled)
        2. Fault evaluation with configured injections
    """

    def run(self) -> Dict[str, Any]:
        """Execute single evaluation.

        Returns:
            Dictionary with results:
            {
                "experiment_name": str,
                "baseline": AccuracyMetrics dict (if enabled),
                "fault": AccuracyMetrics dict,
                "statistics": Dict[injection_name, stats] (if enabled),
            }
        """
        results = self._create_result_dict()
        # Initialize results dictionary with base structure
        results["num_runs"] = self.config.runner.num_runs

        baseline_metrics = None

        # Optionally run baseline evaluation (no faults)
        if self.config.baseline.enabled:
            if self.config.output.verbose:
                print("\n" + "=" * 60)
                print("Running baseline evaluation (no faults)...")

            # Evaluate model performance without any faults
            baseline_metrics = self.evaluator.evaluate_baseline(injection_configs=self.config.injections)

            # Store baseline evaluation results in the results dictionary
            results["baseline"] = baseline_metrics.to_dict()

        if baseline_metrics is not None:
            if self.config.output.verbose:
                # Extract and handle baseline accuracy metrics for display
                baseline_mean = baseline_metrics.mean
                if baseline_mean is None:
                    baseline_mean = 0.0

                baseline_std = baseline_metrics.std
                if baseline_std is None:
                    baseline_std = 0.0

                formatted_accuracy = format_accuracy(
                    accuracy=baseline_mean, std=baseline_std
                )
                print(f"Baseline accuracy: {formatted_accuracy}")

        # Retrieve the list of fault injections that are enabled in the configuration
        enabled_injections = self.config.get_enabled_injections()

        if len(enabled_injections) == 0:
            if self.config.output.verbose:
                print("\n" + "=" * 60)
                print("Baseline-only evaluation - skipping fault injection")
            return results

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print("Running fault injection evaluation...")
            # Display details of each enabled injection
            for injection in enabled_injections:
                print(
                    f"  {injection.name}: {injection.injection_type} @ {injection.probability}% on {injection.target_type}"
                )

        # Run the evaluation with fault injections applied over multiple runs
        fault_metrics = self.evaluator.evaluate_with_faults(injection_configs=enabled_injections)

        # Store fault evaluation results
        results["fault"] = fault_metrics.to_dict()

        # Extract and handle fault accuracy metrics for display
        if self.config.output.verbose:
            fault_mean = fault_metrics.mean
            if fault_mean is None:
                fault_mean = 0.0

            fault_std = fault_metrics.std
            if fault_std is None:
                fault_std = 0.0

            formatted_accuracy = format_accuracy(fault_mean, fault_std)
            print(f"Fault accuracy: {formatted_accuracy}")

        # Check if any injection has statistics tracking enabled
        has_statistics_tracking = False
        for injection in enabled_injections:
            if injection.track_statistics:
                has_statistics_tracking = True
                break

        if has_statistics_tracking:
            # Collect detailed statistics for each injection that has tracking enabled
            stats_dict = {}
            for name, stats in self.evaluator.get_statistics().items():
                stats_dict[name] = stats.to_dict()
            results["statistics"] = stats_dict

        return results
