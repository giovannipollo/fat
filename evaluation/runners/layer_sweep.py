"""Runner for layer-by-layer sweep evaluations."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseRunner
from ..metrics import DegradationMetrics


class LayerSweepRunner(BaseRunner):
    """Runner for layer-by-layer sweep evaluation.

    Evaluates each injection layer individually to understand
    which layers are most vulnerable to faults.

    Supports:
        - Sweep all layers one at a time
        - Specify which layer indices to evaluate
        - Compare resilience across layers
    """

    def run(self) -> Dict[str, Any]:
        """Execute layer-by-layer sweep.

        Returns:
            Dictionary with results:
            {
                "experiment_name": str,
                "baseline": AccuracyMetrics dict,
                "layer_results": Dict with per-layer results,
                "total_layers": int,
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

        enabled_injections = self.config.get_enabled_injections()

        layer_results: Dict[str, Any] = {}

        for injection in enabled_injections:
            if self.config.output.verbose:
                print("\n" + "=" * 60)
                print(f"Layer-by-layer sweep for: {injection.name}")
                print(f"Type: {injection.injection_type}")
                print(f"Target: {injection.target_type}")

            num_layers = self.evaluator.get_num_injection_layers(injection.name)
            results["total_layers"] = num_layers

            if self.config.output.verbose:
                print(f"\nTotal layers: {num_layers}")

            if injection.target_layer_indices is not None:
                layer_indices = injection.target_layer_indices
                if self.config.output.verbose:
                    print(f"Testing specified layers: {layer_indices}")
            else:
                layer_indices = list(range(num_layers))
                if self.config.output.verbose:
                    print(f"Testing all layers (0-{num_layers-1})")

            sweep_results: List[Dict[str, Any]] = []

            for layer_idx in layer_indices:
                if layer_idx < 0 or layer_idx >= num_layers:
                    print(
                        f"Warning: Layer index {layer_idx} out of range [0, {num_layers-1}], skipping"
                    )
                    continue

                if self.config.output.verbose:
                    print(f"\nEvaluating layer {layer_idx}...")

                self.evaluator.enable_injection_layers(injection.name, [layer_idx])

                fault_metrics = self.evaluator.evaluate_with_faults(
                    num_runs=self.config.runner.num_runs
                )

                baseline_mean = (
                    baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
                )
                fault_mean = fault_metrics.mean if fault_metrics.mean is not None else 0.0
                degradation = DegradationMetrics.calculate(baseline_mean, fault_mean)

                layer_result = {
                    "layer_index": layer_idx,
                    "fault_metrics": fault_metrics.to_dict(),
                    "degradation": degradation.to_dict(),
                }

                if any(inj.track_statistics for inj in self.config.get_enabled_injections()):
                    stats_dict = {}
                    for name, stats in self.evaluator.get_statistics().items():
                        stats_dict[name] = stats.to_dict()
                    layer_result["statistics"] = stats_dict

                sweep_results.append(layer_result)

                if self.config.output.verbose:
                    from ..metrics import format_accuracy, format_degradation

                    mean = (
                        fault_metrics.mean if fault_metrics.mean is not None else 0.0
                    )
                    std = (
                        fault_metrics.std if fault_metrics.std is not None else 0.0
                    )
                    print(f"  Accuracy: {format_accuracy(mean, std)}")
                    print(
                        f"  Degradation: {format_degradation(degradation.absolute_degradation)}"
                    )

            layer_results[injection.name] = sweep_results

        if self.config.output.verbose:
            baseline_mean = baseline_metrics.mean if baseline_metrics.mean is not None else 0.0
            self._print_layer_sweep_summary(baseline_mean, sweep_results)

        results["layer_results"] = layer_results

        return results

    def _print_layer_sweep_summary(
        self, baseline_acc: float, sweep_results: List[Dict[str, Any]]
    ) -> None:
        """Print formatted layer sweep summary table."""
        print("\n" + "=" * 80)
        print("Layer-by-Layer Sweep Summary:")
        print(
            f"{'Layer':>8} | {'Accuracy':>12} | {'Degradation':>12}"
        )
        print("-" * 38)

        for result in sweep_results:
            layer_idx = result["layer_index"]
            acc = result["fault_metrics"]["mean"]
            deg = result["degradation"]["absolute_degradation"]

            mean = acc if acc is not None else 0.0
            print(f"{layer_idx:>7} | {mean:>11.2f}% | {deg:>+11.2f}%")

        if sweep_results:
            most_resilient = min(
                sweep_results, key=lambda x: abs(x["degradation"]["absolute_degradation"])
            )
            least_resilient = max(
                sweep_results, key=lambda x: abs(x["degradation"]["absolute_degradation"])
            )

            baseline_mean = baseline_acc if baseline_acc is not None else 0.0
            print("\nMost resilient layer:")
            print(f"  Layer {most_resilient['layer_index']}: {most_resilient['degradation']['absolute_degradation']:+.2f}%")

            print("\nLeast resilient layer:")
            print(f"  Layer {least_resilient['layer_index']}: {least_resilient['degradation']['absolute_degradation']:+.2f}%")
