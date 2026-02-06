"""Abstract base class for evaluation runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..evaluator import Evaluator
from ..config import EvaluationConfig


class BaseRunner(ABC):
    """Abstract base class for evaluation runners.

    Runners orchestrate different types of evaluations:
        - Single: One evaluation with fixed config
        - Sweep: Multiple evaluations across parameter ranges

    Attributes:
        config: Evaluation configuration.
        evaluator: Evaluator instance.
    """

    def __init__(self, config: EvaluationConfig, evaluator: Evaluator):
        """Initialize runner.

        Args:
            config: Evaluation configuration.
            evaluator: Evaluator instance.
        """
        self.config = config
        self.evaluator = evaluator

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute evaluation run.

        Returns:
            Dictionary with evaluation results.
        """
        pass

    def _create_result_dict(self) -> Dict[str, Any]:
        """Create base result dictionary with metadata.

        Returns:
            Dictionary with common metadata fields.
        """
        return {
            "experiment_name": self.config.name,
            "description": self.config.description,
            "runner_type": self.config.runner.type,
            "injections": [inj.name for inj in self.config.get_enabled_injections()],
            "seed": self.config.seed,
        }


def get_runner(config: EvaluationConfig, evaluator: Evaluator) -> BaseRunner:
    """Factory function to create appropriate runner.

    Args:
        config: Evaluation configuration.
        evaluator: Evaluator instance.

    Returns:
        Runner instance based on config.runner.type.

    Raises:
        ValueError: If runner type is unknown.
    """
    from .single import SingleRunner
    from .sweep import SweepRunner
    from .layer_sweep import LayerSweepRunner

    runners: Dict[str, type[BaseRunner]] = {
        "single": SingleRunner,
        "sweep": SweepRunner,
        "layer_sweep": LayerSweepRunner,
    }

    runner_type = config.runner.type
    if runner_type not in runners:
        raise ValueError(
            f"Unknown runner type '{runner_type}'. "
            f"Must be one of {list(runners.keys())}"
        )

    return runners[runner_type](config=config, evaluator=evaluator)
