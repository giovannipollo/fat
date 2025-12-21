"""Statistics tracking for fault injection.

Provides classes to track and report fault injection statistics
including per-layer injection counts, RMSE, and cosine similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class LayerStatistics:
    """Statistics for a single fault injection layer.

    Tracks injection counts and error metrics for one layer.

    Attributes:
        layer_id: Unique identifier for this layer.
        total_activations: Total number of activations processed.
        injected_count: Number of activations with injected faults.
        rmse_sum: Sum of RMSE values across samples.
        cosine_similarity_sum: Sum of cosine similarity values.
        sample_count: Number of samples recorded.
    """

    layer_id: int
    total_activations: int = 0
    injected_count: int = 0
    rmse_sum: float = 0.0
    cosine_similarity_sum: float = 0.0
    sample_count: int = 0

    @property
    def injection_rate(self) -> float:
        """Actual injection rate as percentage.

        Returns:
            Percentage of activations that were injected.
        """
        if self.total_activations == 0:
            return 0.0
        return 100.0 * self.injected_count / self.total_activations

    @property
    def avg_rmse(self) -> float:
        """Average RMSE across samples.

        Returns:
            Average root mean squared error.
        """
        if self.sample_count == 0:
            return 0.0
        return self.rmse_sum / self.sample_count

    @property
    def avg_cosine_similarity(self) -> float:
        """Average cosine similarity across samples.

        Returns:
            Average cosine similarity (0-1 scale).
        """
        if self.sample_count == 0:
            return 0.0
        return self.cosine_similarity_sum / self.sample_count

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_activations = 0
        self.injected_count = 0
        self.rmse_sum = 0.0
        self.cosine_similarity_sum = 0.0
        self.sample_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary.

        Returns:
            Dictionary representation of statistics.
        """
        return {
            "layer_id": self.layer_id,
            "total_activations": self.total_activations,
            "injected_count": self.injected_count,
            "injection_rate": self.injection_rate,
            "avg_rmse": self.avg_rmse,
            "avg_cosine_similarity": self.avg_cosine_similarity,
            "sample_count": self.sample_count,
        }


class FaultStatistics:
    """Tracks fault injection statistics across all layers.

    Computes and aggregates:
    - Number of injected faults per layer
    - RMSE between clean and faulty outputs
    - Cosine similarity between clean and faulty outputs
    - Percentage of different values

    Example:
        ```python
        stats = FaultStatistics(num_layers=10)

        # During forward pass (called by FaultInjectionLayer)
        stats.record(clean_int, faulty_int, mask, layer_id=0)

        # After evaluation
        stats.print_report()
        stats.save_to_file("fault_stats.json")
        ```

    Attributes:
        num_layers: Number of fault injection layers.
        layer_stats: Dictionary mapping layer ID to LayerStatistics.
        enabled: Whether statistics tracking is active.
    """

    def __init__(self, num_layers: int) -> None:
        """Initialize statistics tracker.

        Args:
            num_layers: Number of fault injection layers to track.
        """
        self.num_layers = num_layers
        self.layer_stats: Dict[int, LayerStatistics] = {
            i: LayerStatistics(layer_id=i) for i in range(num_layers)
        }
        self.enabled = True

    def record(
        self,
        clean_int: Tensor,
        faulty_int: Tensor,
        mask: Tensor,
        layer_id: int,
    ) -> None:
        """Record statistics for a forward pass.

        Args:
            clean_int: Original quantized integer tensor.
            faulty_int: Faulty quantized integer tensor.
            mask: Boolean mask indicating injected positions.
            layer_id: Layer identifier.
        """
        if not self.enabled:
            return

        if layer_id not in self.layer_stats:
            self.layer_stats[layer_id] = LayerStatistics(layer_id=layer_id)

        stats = self.layer_stats[layer_id]

        # Count differences
        with torch.no_grad():
            different_mask = clean_int != faulty_int

            stats.total_activations += clean_int.numel()
            stats.injected_count += int(different_mask.sum().item())

            # Compute RMSE (only on different values)
            if different_mask.any():
                clean_diff = clean_int[different_mask].float()
                faulty_diff = faulty_int[different_mask].float()
                diff = clean_diff - faulty_diff
                rmse = torch.sqrt(torch.mean(diff**2)).item()
                stats.rmse_sum += rmse

                # Compute cosine similarity on flattened tensors
                clean_flat = clean_int.flatten().float()
                faulty_flat = faulty_int.flatten().float()

                # Avoid division by zero
                clean_norm = torch.norm(clean_flat)
                faulty_norm = torch.norm(faulty_flat)

                if clean_norm > 0 and faulty_norm > 0:
                    cos_sim = torch.dot(clean_flat, faulty_flat) / (
                        clean_norm * faulty_norm
                    )
                    stats.cosine_similarity_sum += cos_sim.item()
                else:
                    stats.cosine_similarity_sum += 1.0  # Identical if both zero

                stats.sample_count += 1

    def reset(self) -> None:
        """Reset all statistics."""
        for stats in self.layer_stats.values():
            stats.reset()

    def print_report(self) -> None:
        """Print statistics report to console."""
        print("\n" + "=" * 90)
        print("FAULT INJECTION STATISTICS")
        print("=" * 90)
        header = (
            f"{'Layer':<8} {'Injected':<12} {'Total':<14} "
            f"{'Rate (%)':<12} {'RMSE':<12} {'Cos Sim (%)':<12}"
        )
        print(header)
        print("-" * 90)

        total_injected = 0
        total_activations = 0

        for layer_id in sorted(self.layer_stats.keys()):
            stats = self.layer_stats[layer_id]
            total_injected += stats.injected_count
            total_activations += stats.total_activations

            row = (
                f"{layer_id:<8} "
                f"{stats.injected_count:<12} "
                f"{stats.total_activations:<14} "
                f"{stats.injection_rate:<12.4f} "
                f"{stats.avg_rmse:<12.4f} "
                f"{stats.avg_cosine_similarity * 100:<12.4f}"
            )
            print(row)

        print("-" * 90)

        if total_activations > 0:
            overall_rate = 100.0 * total_injected / total_activations
        else:
            overall_rate = 0.0

        total_row = (
            f"{'TOTAL':<8} "
            f"{total_injected:<12} "
            f"{total_activations:<14} "
            f"{overall_rate:<12.4f}"
        )
        print(total_row)
        print("=" * 90 + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with total counts and rates.
        """
        total_injected = sum(s.injected_count for s in self.layer_stats.values())
        total_activations = sum(s.total_activations for s in self.layer_stats.values())

        if total_activations > 0:
            overall_rate = 100.0 * total_injected / total_activations
        else:
            overall_rate = 0.0

        return {
            "total_injected": total_injected,
            "total_activations": total_activations,
            "overall_injection_rate": overall_rate,
            "num_layers": self.num_layers,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export all statistics as dictionary.

        Returns:
            Dictionary with per-layer and summary statistics.
        """
        return {
            "summary": self.get_summary(),
            "layers": {
                layer_id: stats.to_dict()
                for layer_id, stats in self.layer_stats.items()
            },
        }

    def save_to_file(self, path: str) -> None:
        """Save statistics to JSON file.

        Args:
            path: Path to output file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "FaultStatistics":
        """Load statistics from JSON file.

        Args:
            path: Path to input file.

        Returns:
            FaultStatistics instance with loaded data.
        """
        with open(path, "r") as f:
            data = json.load(f)

        num_layers = data["summary"]["num_layers"]
        stats = cls(num_layers=num_layers)

        for layer_id_str, layer_data in data["layers"].items():
            layer_id = int(layer_id_str)
            if layer_id in stats.layer_stats:
                ls = stats.layer_stats[layer_id]
                ls.total_activations = layer_data["total_activations"]
                ls.injected_count = layer_data["injected_count"]
                ls.sample_count = layer_data["sample_count"]
                # Note: rmse_sum and cosine_similarity_sum are not saved,
                # only averages are available after loading

        return stats

    @property
    def overall_injection_rate(self) -> float:
        """Overall injection rate across all layers.

        Returns:
            Percentage of total activations that were injected.
        """
        return self.get_summary()["overall_injection_rate"]
