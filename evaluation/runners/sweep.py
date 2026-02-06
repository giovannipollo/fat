"""Runner for probability sweep evaluations."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseRunner
from ..metrics.formatting import format_accuracy


class SweepRunner(BaseRunner):
    """Runner for probability sweep evaluation.

    Sweeps fault injection probability across a range and measures
    accuracy at each point.

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
        # Initialize results dictionary with base structure
        results = self._create_result_dict()

        if not self.config.runner.probabilities:
            raise ValueError("Sweep runner requires probabilities list in config")

        # Extract the list of probabilities to sweep over
        probabilities = self.config.runner.probabilities
        
        # Store the sweep probabilities in results
        results["sweep_probabilities"] = probabilities

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print("Running baseline evaluation...")

        # Evaluate model performance without any faults
        baseline_metrics = self.evaluator.evaluate_baseline()
        
        # Store baseline evaluation results
        results["baseline"] = baseline_metrics.to_dict()

        if self.config.output.verbose:
            # Extract and handle baseline accuracy for display
            baseline_mean = baseline_metrics.mean
            if baseline_mean is None:
                baseline_mean = 0.0
                
            baseline_std = baseline_metrics.std
            if baseline_std is None:
                baseline_std = 0.0
                
            print(f"Baseline accuracy: {format_accuracy(baseline_mean, baseline_std)}")

        # Initialize list to store evaluation results for each probability point
        sweep_results: List[Dict[str, Any]] = []

        if self.config.output.verbose:
            print("\n" + "=" * 60)
            print(f"Running probability sweep: {probabilities}")
            print(f"Runs per probability: {self.config.runner.num_runs}")

        # Sweep through each probability value in the configured list
        for prob in probabilities:
            if self.config.output.verbose:
                print(f"\nEvaluating probability: {prob}%")

            # Update all enabled injections to use the current sweep probability
            for injection in self.config.get_enabled_injections():
                self.evaluator.update_injection_probability(injection.name, prob)

            # Run evaluation with faults applied at the current probability
            fault_metrics = self.evaluator.evaluate_with_faults(
                num_runs=self.config.runner.num_runs
            )

            baseline_mean = baseline_metrics.mean
            if baseline_mean is None:
                baseline_mean = 0.0
                
            fault_mean = fault_metrics.mean
            if fault_mean is None:
                fault_mean = 0.0
                
            # Create a dictionary with results for this probability point
            prob_result = {
                "probability": prob,
                "fault_metrics": fault_metrics.to_dict(),
            }

            # Check if any injection has statistics tracking enabled
            has_statistics_tracking = False
            for inj in self.config.get_enabled_injections():
                if inj.track_statistics:
                    has_statistics_tracking = True
                    break

            # Collect detailed statistics for injections with tracking enabled
            if has_statistics_tracking:
                stats_dict = {}
                for name, stats in self.evaluator.get_statistics().items():
                    stats_dict[name] = stats.to_dict()
                prob_result["statistics"] = stats_dict

            # Add the result for this probability to the sweep results list
            sweep_results.append(prob_result)

            # Display accuracy and degradation results for this probability
            if self.config.output.verbose:
                fault_mean = fault_metrics.mean
                if fault_mean is None:
                    fault_mean = 0.0
                    
                fault_std = fault_metrics.std
                if fault_std is None:
                    fault_std = 0.0
                    
                print(f"  Accuracy: {format_accuracy(fault_mean, fault_std)}")

        # Store the complete list of sweep results in the output dictionary
        results["sweep_results"] = sweep_results

        # Print a formatted summary table of all sweep results
        if self.config.output.verbose:
            self._print_sweep_summary(sweep_results)

        return results

    def _print_sweep_summary(
        self, sweep_results: List[Dict[str, Any]]
    ) -> None:
        """Print formatted sweep summary table."""
        print("\n" + "=" * 60)
        print("Sweep Summary:")
        print(f"{'Probability':>12} | {'Accuracy':>12}")
        print("-" * 27)

        for result in sweep_results:
            prob = result["probability"]
            acc = result["fault_metrics"]["mean"]
            print(f"{prob:>11.1f}% | {acc:>11.2f}%")
