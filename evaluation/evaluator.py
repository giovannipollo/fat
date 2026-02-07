"""Main evaluator for fault injection evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import EvaluationConfig, InjectionConfig
from .metrics import AccuracyMetrics
from utils.fault_injection import (
    ActivationFaultInjector,
    BaseFaultInjector,
    WeightFaultInjector,
    FaultStatistics,
)


class Evaluator:
    """Main evaluator for fault injection experiments.

    Orchestrates evaluation workflow:
        1. Load model and dataset
        2. Setup fault injectors
        3. Run baseline evaluation
        4. Run fault evaluations
        5. Collect and aggregate results

    Attributes:
        config: Evaluation configuration.
        model: Neural network model.
        device: Compute device.
        test_loader: Test dataset loader.
        injectors: Dict of fault injectors by injection name.
        statistics: Dict of fault statistics by injection name.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        model: nn.Module,
        test_loader: DataLoader[Any],
        device: torch.device,
    ):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration.
            model: Neural network model.
            test_loader: Test dataset loader.
            device: Compute device.
        """
        self.config = config
        self.model = model
        self.device = device
        self.test_loader = test_loader

        self.injectors: Dict[
            str, Union[ActivationFaultInjector, WeightFaultInjector]
        ] = {}
        self.statistics: Dict[str, FaultStatistics] = {}

        self.show_progress = config.output.show_progress
        self.verbose = config.output.verbose

        self.setup_injectors()

    def setup_injectors(self) -> None:
        """Setup fault injectors based on configuration."""
        enabled_injections = self.config.get_enabled_injections()

        for injection in enabled_injections:
            fault_config = injection.to_fault_injection_config()

            if injection.target_type == "activation":
                injector = ActivationFaultInjector()
            elif injection.target_type == "weight":
                injector = WeightFaultInjector()
            else:
                raise ValueError(f"Unknown target type: {injection.target_type}")

            self.injectors[injection.name] = injector
            self.model = injector.inject(model=self.model, config=fault_config)

            if injection.track_statistics:
                num_layers = injector.get_num_layers(self.model)
                stats = FaultStatistics(num_layers=num_layers)
                injector.set_statistics(self.model, stats)
                self.statistics[injection.name] = stats

    def evaluate_baseline(self) -> AccuracyMetrics:
        """Evaluate model without fault injection.

        Returns:
            AccuracyMetrics with baseline results.
        """
        self.enable_injectors(enabled=False)

        runs = []
        num_runs = self.config.baseline.num_runs

        for i in range(num_runs):
            correct, total = self._single_evaluation(
                desc=f"Baseline run {i + 1}/{num_runs}"
            )
            runs.append((correct, total))

        return AccuracyMetrics.from_runs(runs)

    def evaluate_with_faults(
        self,
        num_runs: int = 1,
        desc: str = "Fault evaluation",
    ) -> AccuracyMetrics:
        """Evaluate model with fault injection.

        Args:
            num_runs: Number of evaluation runs for averaging.
            desc: Description for progress bar.

        Returns:
            AccuracyMetrics with fault evaluation results.
        """

        self.enable_injectors(enabled=True)

        for stats in self.statistics.values():
            stats.reset()

        runs = []
        for i in range(num_runs):
            correct, total = self._single_evaluation(
                desc=f"{desc} run {i + 1}/{num_runs}",
            )
            runs.append((correct, total))

        return AccuracyMetrics.from_runs(runs)

    def _single_evaluation(self, desc: str = "Evaluating") -> Tuple[int, int]:
        """Run single forward pass through test set.

        Args:
            desc: Description for progress bar.

        Returns:
            Tuple of (correct_count, total_count).
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(
                self.test_loader, desc=desc, leave=False, disable=not self.show_progress
            ):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                _, predicted = torch.max(output, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

        return correct, total

    def enable_injectors(self, enabled: bool = True) -> None:
        """Enable/disable all injectors.

        Args:
            enabled: Whether to enable injection.
        """
        for injector in self.injectors.values():
            injector.set_enabled(self.model, enabled)

    def enable_injection_layers(
        self, injection_name: str, layer_indices: Optional[List[int]] = None
    ) -> None:
        """Enable/disable specific injection layers by index.

        Args:
            injection_name: Name of injection to configure.
            layer_indices: List of layer indices to enable (0-based). None = all layers.

        Raises:
            ValueError: If injection_name not found.
        """
        if injection_name not in self.injectors:
            raise ValueError(
                f"Unknown injection name: {injection_name}. "
                f"Available: {list(self.injectors.keys())}"
            )

        injector = self.injectors[injection_name]

        if layer_indices is None:
            injector.set_enabled(self.model, True)
        else:
            injector.set_enabled(self.model, False)
            for layer_id in layer_indices:
                if hasattr(injector, "set_layer_enabled"):
                    injector.set_layer_enabled(self.model, layer_id, True)

    def get_num_injection_layers(self, injection_name: str) -> int:
        """Get number of injection layers for a specific injection.

        Args:
            injection_name: Name of injection.

        Returns:
            Number of injection layers.

        Raises:
            ValueError: If injection_name not found.
        """
        if injection_name not in self.injectors:
            raise ValueError(
                f"Unknown injection name: {injection_name}. "
                f"Available: {list(self.injectors.keys())}"
            )

        injector = self.injectors[injection_name]
        return injector.get_num_layers(self.model)

    def update_injection_probability(
        self, injection_name: str, probability: float
    ) -> None:
        """Update probability for a specific injection.

        Args:
            injection_name: Name of injection to update.
            probability: New probability (0-100).

        Raises:
            ValueError: If injection_name not found.
        """
        if injection_name not in self.injectors:
            raise ValueError(
                f"Unknown injection name: {injection_name}. "
                f"Available: {list(self.injectors.keys())}"
            )

        injector: Union[ActivationFaultInjector, WeightFaultInjector] = self.injectors[
            injection_name
        ]
        injector.update_probability(self.model, probability)

    def get_statistics(self) -> Dict[str, FaultStatistics]:
        """Get statistics from all injectors.

        Returns:
            Dictionary of statistics by injection name.
        """
        return self.statistics

    def cleanup(self) -> None:
        """Cleanup and remove fault injection layers."""
        for injector in self.injectors.values():
            self.model = injector.remove(self.model)

        self.injectors = {}
        self.statistics = {}
