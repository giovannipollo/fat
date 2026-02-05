"""Evaluation framework for fault injection experiments.

This module provides a modular framework for evaluating fault resilience
of neural network models. It supports:

- YAML-based evaluation configuration
- Multiple injection strategies (activation, weight, or combined)
- Different evaluation patterns (single, sweep, comparison)
- Flexible output formats (console, JSON, CSV)

Main Components:
    - EvaluationConfig: Configuration dataclass with YAML support
    - Evaluator: Main evaluation orchestrator
    - AccuracyMetrics/DegradationMetrics: Metric calculation utilities
    - Runners: Single, Sweep, Comparison evaluation patterns
    - Reporters: Console, JSON output formatting

Example:
    ```python
    from evaluation import EvaluationConfig, Evaluator
    from evaluation.runners import get_runner
    from evaluation.reporters import get_reporters

    # Load configuration
    config = EvaluationConfig.from_yaml("configs/evaluation/sweep.yaml")

    # Create evaluator
    evaluator = Evaluator(config, model, test_loader, device)

    # Run evaluation
    runner = get_runner(config, evaluator)
    results = runner.run()

    # Report results
    reporters = get_reporters(formats=["console", "json"])
    for reporter in reporters:
        reporter.report(results)
    ```
"""

from __future__ import annotations

from .config import (
    EvaluationConfig,
    InjectionConfig,
    BaselineConfig,
    RunnerConfig,
    OutputConfig,
)
from .evaluator import Evaluator
from .metrics import AccuracyMetrics, DegradationMetrics

__all__ = [
    # Configuration
    "EvaluationConfig",
    "InjectionConfig",
    "BaselineConfig",
    "RunnerConfig",
    "OutputConfig",
    # Evaluator
    "Evaluator",
    # Metrics
    "AccuracyMetrics",
    "DegradationMetrics",
]
