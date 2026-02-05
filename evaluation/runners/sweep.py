"""Runner for probability sweep evaluations."""

from __future__ import annotations

from typing import Any, Dict, List

from tqdm import tqdm

from .base import BaseRunner
from ..metrics import DegradationMetrics


class SweepRunner(BaseRunner):
    """Runner for probability sweep evaluation.

    Sweeps fault injection probability across a range and measures
    accuracy degradation at each point.

    Supports sweeping:
        - Single injection probability
        - Multiple injections simultaneously (all sweep together)
    """

    def run(self) -> Dict[str, Any]:
        """Execute probability sweep.

        Returns:
            Dictionary with results:
            {
                "experiment_name": str,
                "baseline": AccuracyMetrics dict,
                "sweep_probabilities": List[float],
                "sweep_results": List[Dict] with per-probability results,
            }
        """
        results = self._create_result_dict()

        if not self.config.runner.probabilities:
            raise ValueError("Sweep runner requires probabilities list in config")

        probabilities = self.config.runner.probabilities
        results["sweep_probabilities"] = probabilities

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

        sweep_results: List[Dict[str, Any]] = []

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print(f"Running probability sweep: {probabilities}")
            print(f"Runs per probability: {self.config.runner.num_runs}")

        for prob in tqdm(
            probabilities,
            desc="Probability sweep",
            disable=not self.config.output.show_progress,
        ):
            if self.config.output.verbose:
                print(f"\nEvaluating probability: {prob}%")

            for injection in self.config.get_enabled_injections():
                self.evaluator.update_injection_probability(injection.name, prob)

            fault_metrics = self.evaluator.evaluate_with_faults(
                num_runs=self.config.runner.num_runs
            )

            baseline_mean = baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            fault_mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
            degradation = DegradationMetrics.calculate(baseline_mean, fault_mean)

            prob_result = {
                "probability": prob,
                "fault_metrics": fault_metrics.to_dict(),
                "degradation": degradation.to_dict(),
            }

            if any(inj.track_statistics for inj in self.config.get_enabled_injections()):
                stats_dict = {}
                for name, stats in self.evaluator.get_statistics().items():
                    stats_dict[name] = stats.to_dict()
                prob_result["statistics"] = stats_dict

            sweep_results.append(prob_result)

            if self.config.output.verbose:
                from ..metrics import format_accuracy, format_degradation

                mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
                std = fault_metrics.std if fault_metrics.std is not None else 0.0
                print(f"  Accuracy: {format_accuracy(mean, std)}")
                print(
                    f"  Degradation: {format_degradation(degradation.absolute_degradation)}"
                )

        results["sweep_results"] = sweep_results

        if self.config.output.verbose:
            baseline_mean = baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            self._print_sweep_summary(baseline_mean, sweep_results)

        return results

    def _print_sweep_summary(
        self, baseline_acc: float, sweep_results: List[Dict[str, Any]]
    ) -> None:
        """Print formatted sweep summary table."""
        print("\n" + "=" * 60)
        print("Sweep Summary:")
        print(f"{'Probability':>12} | {'Accuracy':>12} | {'Degradation':>12}")
        print("-" * 42)

        for result in sweep_results:
            prob = result["probability"]
            acc = result["fault_metrics"]["mean"]
            deg = result["degradation"]["absolute_degradation"]
            print(f"{prob:>11.1f}% | {acc:>11.2f}% | {deg:>+11.2f}%")
