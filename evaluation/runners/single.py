"""Runner for single evaluation with fixed configuration."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseRunner
from ..metrics import DegradationMetrics


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

            if self.config.output.verbose:
                from ..metrics import format_accuracy

                mean = (
                    baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
                )
                std = baseline_metrics.std if baseline_metrics.std is not None else 0.0
                print(f"Baseline accuracy: {format_accuracy(mean, std)}")

        enabled_injections = self.config.get_enabled_injections()

        if len(enabled_injections) == 0:
            if self.config.output.verbose:
                print("\n" + "=" * 60)
                print("Baseline-only evaluation - skipping fault injection")
            return results

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print("Running fault injection evaluation...")
            for inj in enabled_injections:
                print(
                    f"  {inj.name}: {inj.injection_type} @ {inj.probability}% on {inj.target_type}"
                )

        fault_metrics = self.evaluator.evaluate_with_faults(
            num_runs=self.config.runner.num_runs
        )
        results["fault"] = fault_metrics.to_dict()

        if self.config.output.verbose:
            from ..metrics import format_accuracy

            mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
            std = fault_metrics.std if fault_metrics.std is not None else 0.0
            print(f"Fault accuracy: {format_accuracy(mean, std)}")

        if baseline_metrics is not None:
            baseline_mean = (
                baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            )
            fault_mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
            degradation = DegradationMetrics.calculate(baseline_mean, fault_mean)
            results["degradation"] = degradation.to_dict()

            if self.config.output.verbose:
                from ..metrics import format_degradation

                print(
                    f"Degradation: {format_degradation(degradation.absolute_degradation)}"
                )

        if any(inj.track_statistics for inj in enabled_injections):
            stats_dict = {}
            for name, stats in self.evaluator.get_statistics().items():
                stats_dict[name] = stats.to_dict()
            results["statistics"] = stats_dict

        return results
