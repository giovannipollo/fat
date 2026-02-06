"""Runner for probability sweep evaluations."""

from __future__ import annotations

from typing import Any, Dict, List

from tqdm import tqdm

from .base import BaseRunner
from ..metrics import DegradationMetrics
from ..metrics.formatting import format_accuracy, format_degradation


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
        # Initialize results dictionary with base structure

        if not self.config.runner.probabilities:
            raise ValueError("Sweep runner requires probabilities list in config")

        probabilities = self.config.runner.probabilities
        # Extract the list of probabilities to sweep over
        results["sweep_probabilities"] = probabilities
        # Store the sweep probabilities in results

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print("Running baseline evaluation...")

        baseline_metrics = self.evaluator.evaluate_baseline()
        # Evaluate model performance without any faults
        results["baseline"] = baseline_metrics.to_dict()
        # Store baseline evaluation results

        if self.config.output.verbose:
            # Extract and handle baseline accuracy for display
            baseline_mean = baseline_metrics.mean
            if baseline_mean is None:
                baseline_mean = 0.0
                
            baseline_std = baseline_metrics.std
            if baseline_std is None:
                baseline_std = 0.0
                
            print(f"Baseline accuracy: {format_accuracy(baseline_mean, baseline_std)}")

        sweep_results: List[Dict[str, Any]] = []
        # Initialize list to store evaluation results for each probability point

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print(f"Running probability sweep: {probabilities}")
            print(f"Runs per probability: {self.config.runner.num_runs}")

        # Sweep through each probability value in the configured list
        for prob in tqdm(
            probabilities,
            desc="Probability sweep",
            disable=not self.config.output.show_progress,
        ):
            if self.config.output.verbose:
                print(f"\nEvaluating probability: {prob}%")

            # Update all enabled injections to use the current sweep probability
            for injection in self.config.get_enabled_injections():
                self.evaluator.update_injection_probability(injection.name, prob)

            fault_metrics = self.evaluator.evaluate_with_faults(
                num_runs=self.config.runner.num_runs
            )
            # Run evaluation with faults applied at the current probability

            baseline_mean = baseline_metrics.mean
            if baseline_mean is None:
                baseline_mean = 0.0
                
            fault_mean = fault_metrics.mean
            if fault_mean is None:
                fault_mean = 0.0
                
            degradation = DegradationMetrics.calculate(baseline_mean, fault_mean)
            # Calculate accuracy degradation compared to baseline performance

            prob_result = {
                "probability": prob,
                "fault_metrics": fault_metrics.to_dict(),
                "degradation": degradation.to_dict(),
            }
            # Create a dictionary with results for this probability point

            # Check if any injection has statistics tracking enabled
            has_statistics_tracking = False
            for inj in self.config.get_enabled_injections():
                if inj.track_statistics:
                    has_statistics_tracking = True
                    break

            if has_statistics_tracking:
                # Collect detailed statistics for injections with tracking enabled
                stats_dict = {}
                for name, stats in self.evaluator.get_statistics().items():
                    stats_dict[name] = stats.to_dict()
                prob_result["statistics"] = stats_dict

            sweep_results.append(prob_result)
            # Add the result for this probability to the sweep results list

            if self.config.output.verbose:
                # Display accuracy and degradation results for this probability
                fault_mean = fault_metrics.mean
                if fault_mean is None:
                    fault_mean = 0.0
                    
                fault_std = fault_metrics.std
                if fault_std is None:
                    fault_std = 0.0
                    
                print(f"  Accuracy: {format_accuracy(fault_mean, fault_std)}")
                print(
                    f"  Degradation: {format_degradation(degradation.absolute_degradation)}"
                )

        results["sweep_results"] = sweep_results
        # Store the complete list of sweep results in the output dictionary

        if self.config.output.verbose:
            baseline_mean = baseline_metrics.mean
            if baseline_mean is None:
                baseline_mean = 0.0
                
            self._print_sweep_summary(baseline_mean, sweep_results)
            # Print a formatted summary table of all sweep results

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
