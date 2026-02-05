"""Runner for comparing multiple injection strategies."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseRunner
from ..config import InjectionConfig
from ..metrics import DegradationMetrics


class ComparisonRunner(BaseRunner):
    """Runner for comparing different injection strategies.

    Compares multiple injection configurations side-by-side:
        - Different injection types (random vs lsb_flip)
        - Different target types (activation vs weight)
        - Different injection combinations
    """

    def run(self) -> Dict[str, Any]:
        """Execute comparison evaluation.

        Returns:
            Dictionary with results:
            {
                "experiment_name": str,
                "baseline": AccuracyMetrics dict,
                "comparisons": List[Dict] with per-injection results,
            }
        """
        results = self._create_result_dict()

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print("Running baseline evaluation...")

        baseline_metrics = self.evaluator.evaluate_baseline()
        results["baseline"] = baseline_metrics.to_dict()

        if self.config.output.verbose:
            from ..metrics import format_accuracy

            mean = baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            std = baseline_metrics.std if baseline_metrics.std is not None else 0.0
            print(f"Baseline accuracy: {format_accuracy(mean, std)}")

        comparisons: List[Dict[str, Any]] = []
        enabled_injections = self.config.get_enabled_injections()

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print(f"Comparing {len(enabled_injections)} injection configurations...")

        for injection in enabled_injections:
            if self.config.output.verbose:
                print(f"\nEvaluating: {injection.name}")
                print(f"  Type: {injection.injection_type}")
                print(f"  Target: {injection.target_type}")
                print(f"  Probability: {injection.probability}%")

            fault_metrics = self.evaluator.evaluate_with_faults(
                injection_configs=[injection], num_runs=self.config.runner.num_runs
            )

            baseline_mean = (
                baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            )
            fault_mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
            degradation = DegradationMetrics.calculate(baseline_mean, fault_mean)

            comp_result = {
                "injection_name": injection.name,
                "injection_config": {
                    "target_type": injection.target_type,
                    "injection_type": injection.injection_type,
                    "probability": injection.probability,
                    "target_layers": injection.target_layers,
                },
                "fault_metrics": fault_metrics.to_dict(),
                "degradation": degradation.to_dict(),
            }

            comparisons.append(comp_result)

            if self.config.output.verbose:
                from ..metrics import format_accuracy, format_degradation

                mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
                std = fault_metrics.std if fault_metrics.std is not None else 0.0
                print(f"  Accuracy: {format_accuracy(mean, std)}")
                print(
                    f"  Degradation: {format_degradation(degradation.absolute_degradation)}"
                )

        results["comparisons"] = comparisons

        if self.config.output.verbose:
            baseline_mean = (
                baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            )
            self._print_comparison_table(baseline_mean, comparisons)

        return results

    def _print_comparison_table(
        self, baseline_acc: float, comparisons: List[Dict[str, Any]]
    ) -> None:
        """Print formatted comparison table."""
        print("\n" + "=" * 80)
        print("Comparison Summary:")
        print(
            f"{'Name':>20} | {'Type':>12} | {'Target':>12} | {'Prob':>6} | {'Acc':>10} | {'Degrad':>10}"
        )
        print("-" * 80)

        for comp in comparisons:
            name = comp["injection_name"][:20]
            config = comp["injection_config"]
            inj_type = config["injection_type"][:12]
            target = config["target_type"][:12]
            prob = config["probability"]
            acc = comp["fault_metrics"]["mean"]
            deg = comp["degradation"]["absolute_degradation"]

            mean = acc if acc is not None else 0.0
            print(
                f"{name:>20} | {inj_type:>12} | {target:>12} | {prob:>5.1f}% | {mean:>9.2f}% | {deg:>+9.2f}%"
            )
