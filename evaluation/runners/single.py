"""Runner for single evaluation with fixed configuration."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseRunner
from ..metrics import DegradationMetrics
from ..metrics.formatting import format_accuracy, format_degradation


class SingleRunner(BaseRunner):
    """Runner for single fault injection evaluation.

    Executes:
        1. Baseline evaluation (if enabled)
        2. Fault evaluation with configured injections
        3. Calculate degradation metrics
    """

    def run(self) -> Dict[str, Any]:
        """Execute single evaluation.

        Returns:
            Dictionary with results:
            {
                "experiment_name": str,
                "baseline": AccuracyMetrics dict (if enabled),
                "fault": AccuracyMetrics dict,
                "degradation": DegradationMetrics dict (if baseline enabled),
                "statistics": Dict[injection_name, stats] (if enabled),
            }
        """
        results = self._create_result_dict()
        results["num_runs"] = self.config.runner.num_runs

        baseline_metrics = None
        if self.config.baseline.enabled:
            if self.config.output.verbose:
                print("\n" + "=" * 60)
                print("Running baseline evaluation (no faults)...")

            baseline_metrics = self.evaluator.evaluate_baseline()
            results["baseline"] = baseline_metrics.to_dict()

        if baseline_metrics is not None:
            if self.config.output.verbose:

                baseline_mean = baseline_metrics.mean
                if baseline_mean is None:
                    baseline_mean = 0.0

                baseline_std = baseline_metrics.std
                if baseline_std is None:
                    baseline_std = 0.0

                formatted_accuracy = format_accuracy(
                    accuracy=baseline_mean, baseline_std=baseline_std
                )
                print(f"Baseline accuracy: {formatted_accuracy}")

        enabled_injections = self.config.get_enabled_injections()

        if len(enabled_injections) == 0:
            if self.config.output.verbose:
                print("\n" + "=" * 60)
                print("Baseline-only evaluation - skipping fault injection")
            return results

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print("Running fault injection evaluation...")
            for injection in enabled_injections:
                print(
                    f"  {injection.name}: {injection.injection_type} @ {injection.probability}% on {injection.target_type}"
                )

        fault_metrics = self.evaluator.evaluate_with_faults(
            num_runs=self.config.runner.num_runs
        )
        results["fault"] = fault_metrics.to_dict()

        if self.config.output.verbose:

            fault_mean = fault_metrics.mean
            if fault_mean is None:
                fault_mean = 0.0

            fault_std = fault_metrics.std
            if fault_std is None:
                fault_std = 0.0

            formatted_accuracy = format_accuracy(fault_mean, fault_std)
            print(f"Fault accuracy: {formatted_accuracy}")

        if baseline_metrics is not None:
            baseline_mean = baseline_metrics.mean
            if baseline_mean is None:
                baseline_mean = 0.0

            fault_mean = fault_metrics.mean
            if fault_mean is None:
                fault_mean = 0.0

            degradation = DegradationMetrics.calculate(
                baseline=baseline_mean, fault=fault_mean
            )
            results["degradation"] = degradation.to_dict()

            if self.config.output.verbose:

                formatted_degradation = format_degradation(
                    degradation=degradation.absolute_degradation
                )
                print(f"Degradation: {formatted_degradation}")

        # Check if any injection has statistics tracking enabled
        has_statistics_tracking = False
        for injection in enabled_injections:
            if injection.track_statistics:
                has_statistics_tracking = True
                break

        if has_statistics_tracking:
            stats_dict = {}
            for name, stats in self.evaluator.get_statistics().items():
                stats_dict[name] = stats.to_dict()
            results["statistics"] = stats_dict

        return results
