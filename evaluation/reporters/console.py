"""Reporter for formatted console output."""

from __future__ import annotations

from typing import Any, Dict

from .base import BaseReporter


class ConsoleReporter(BaseReporter):
    """Reporter for formatted console output.

    Formats results as tables and text summaries for terminal display.
    Handles different runner types (single, sweep).
    """

    def report(self, results: Dict[str, Any]) -> None:
        """Report evaluation results to console.

        Args:
            results: Dictionary with evaluation results.
        """
        self._print_header(results)

        if "baseline" in results:
            self._print_baseline(results["baseline"])

        runner_type = results.get("runner_type", "single")

        if runner_type == "single":
            self._report_single(results)
        elif runner_type == "sweep":
            self._report_sweep(results)
        else:
            self._report_generic(results)

    def _print_header(self, results: Dict[str, Any]) -> None:
        """Print experiment header.

        Args:
            results: Results dictionary.
        """
        print("\n" + "=" * 80)
        print(f"Evaluation: {results.get('experiment_name', 'Unknown')}")
        if results.get("description"):
            print(f"Description: {results['description']}")
        print(f"Runner: {results.get('runner_type', 'unknown')}")
        if results.get("injections"):
            print(f"Injections: {', '.join(results['injections'])}")
        print("=" * 80)

    def _print_baseline(self, baseline: Dict[str, Any]) -> None:
        """Print baseline results.

        Args:
            baseline: Baseline metrics dictionary.
        """
        print("\nBaseline (No Faults):")
        mean = baseline.get("mean", baseline.get("accuracy"))
        std = baseline.get("std")
        mean_val = mean if mean is not None else 0.0
        print(f"  Accuracy: {self._format_metric(mean_val, std)}")

    def _report_single(self, results: Dict[str, Any]) -> None:
        """Report single evaluation results.

        Args:
            results: Results dictionary.
        """
        if "fault" not in results:
            if self.verbose:
                print("\nBaseline-only evaluation complete.")
            return

        print("\nFault Injection Results:")

        fault = results["fault"]
        mean = fault.get("mean", fault.get("accuracy"))
        std = fault.get("std")
        print(f"  Accuracy: {self._format_metric(mean, std)}")

        if "degradation" in results:
            deg = results["degradation"]
            abs_deg = deg["absolute_degradation"]
            rel_deg = deg["relative_degradation"]
            print(
                f"  Absolute degradation: {self._format_percentage(abs_deg, show_sign=True)}"
            )
            print(
                f"  Relative degradation: {self._format_percentage(rel_deg, decimals=1)}%"
            )

        if "statistics" in results:
            self._print_statistics(results["statistics"])

    def _report_sweep(self, results: Dict[str, Any]) -> None:
        """Report sweep results as table.

        Args:
            results: Results dictionary.
        """
        print("\nProbability Sweep Results:")

        sweep_results = results["sweep_results"]
        baseline_acc = results["baseline"]["mean"]

        print(f"\n{'Probability':>12} | {'Accuracy':>14}")
        print("-" * 30)

        for point in sweep_results:
            prob = point["probability"]
            acc_mean = point["fault_metrics"]["mean"]
            acc_std = point["fault_metrics"].get("std", 0)

            prob_str = f"{prob:.1f}%"
            acc_str = self._format_metric(acc_mean, acc_std if acc_std > 0 else None)

            print(f"{prob_str:>12} | {acc_str:>14}")

    def _report_generic(self, results: Dict[str, Any]) -> None:
        """Generic fallback reporter.

        Args:
            results: Results dictionary.
        """
        print("\nResults:")
        for key, value in results.items():
            if key not in [
                "experiment_name",
                "description",
                "runner_type",
                "injections",
            ]:
                print(f"  {key}: {value}")

    def _print_statistics(self, statistics: Dict[str, Any]) -> None:
        """Print fault injection statistics.

        Args:
            statistics: Statistics dictionary by injection name.
        """
        print("\nFault Injection Statistics:")
        for inj_name, stats in statistics.items():
            print(f"\n  {inj_name}:")
            if "total_faults" in stats:
                print(f"    Total faults: {stats['total_faults']}")
            if "avg_rmse" in stats:
                print(f"    Avg RMSE: {stats['avg_rmse']:.6f}")
            if "avg_cosine_sim" in stats:
                print(f"    Avg Cosine Similarity: {stats['avg_cosine_sim']:.6f}")
