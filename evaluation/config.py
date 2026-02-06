"""Configuration for fault injection evaluation.

Provides dataclasses for managing evaluation experiments with support for
YAML configuration loading and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from utils.fault_injection import FaultInjectionConfig


@dataclass
class InjectionConfig:
    """Configuration for a single fault injection in evaluation.

    Attributes:
        name: Identifier for this injection (e.g., "activation", "weight").
        enabled: Whether this injection is active.
        target_type: "activation" or "weight".
        probability: Injection probability (0-100).
        injection_type: "random", "lsb_flip", "msb_flip", "full_flip".
        target_layers: List of layer types to inject.
        target_layer_indices: List of specific layer indices to inject (0-based). None = all layers.
        track_statistics: Enable statistics tracking.
    """

    name: str
    enabled: bool = True
    target_type: str = "activation"
    probability: float = 5.0
    injection_type: str = "random"
    target_layers: List[str] = field(default_factory=list)
    target_layer_indices: Optional[List[int]] = None
    track_statistics: bool = True

    def to_fault_injection_config(self) -> FaultInjectionConfig:
        """Convert to FaultInjectionConfig.

        Returns:
            FaultInjectionConfig instance for use with injectors.
        """
        return FaultInjectionConfig(
            enabled=self.enabled,
            target_type=self.target_type,
            probability=self.probability,
            injection_type=self.injection_type,
            apply_during="eval",
            target_layers=self.target_layers,
            track_statistics=self.track_statistics,
            verbose=False,
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "InjectionConfig":
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing injection parameters.

        Returns:
            InjectionConfig instance.
        """
        return cls(
            name=config["name"],
            enabled=config.get("enabled", True),
            target_type=config.get("target_type", "activation"),
            probability=config.get("probability", 5.0),
            injection_type=config.get("injection_type", "random"),
            target_layers=config.get("target_layers", []),
            track_statistics=config.get("track_statistics", True),
        )


@dataclass
class BaselineConfig:
    """Configuration for baseline (no-fault) evaluation.

    Attributes:
        enabled: Whether to run baseline evaluation.
        num_runs: Number of baseline runs for averaging.
    """

    enabled: bool = True
    num_runs: int = 1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaselineConfig":
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing baseline parameters.

        Returns:
            BaselineConfig instance.
        """
        return cls(
            enabled=config.get("enabled", True),
            num_runs=config.get("num_runs", 1),
        )


@dataclass
class RunnerConfig:
    """Configuration for evaluation runner.

    Attributes:
        type: Runner type - "single", "sweep", "comparison".
        probabilities: List of probabilities for sweep.
        num_runs: Number of runs per configuration.
        seeds: Optional list of seeds for reproducibility.
    """

    type: str = "single"
    probabilities: List[float] = field(default_factory=list)
    num_runs: int = 1
    seeds: Optional[List[int]] = None

    def validate(self) -> None:
        """Validate runner configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        valid_types = ["single", "sweep", "comparison"]
        if self.type not in valid_types:
            raise ValueError(
                f"Runner type must be one of {valid_types}, got '{self.type}'"
            )

        if self.type == "sweep" and not self.probabilities:
            raise ValueError("Sweep runner requires probabilities list")

        for prob in self.probabilities:
            if prob < 0.0 or prob > 100.0:
                raise ValueError(f"Probability must be between 0 and 100, got {prob}")

        if self.num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {self.num_runs}")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RunnerConfig":
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing runner parameters.

        Returns:
            RunnerConfig instance.
        """
        return cls(
            type=config.get("type", "single"),
            probabilities=config.get("probabilities", []),
            num_runs=config.get("num_runs", 1),
            seeds=config.get("seeds"),
        )


@dataclass
class OutputConfig:
    """Configuration for evaluation output/reporting.

    Attributes:
        formats: List of output formats - "console", "json", "csv".
        save_path: Path template for saving results.
        verbose: Enable verbose output.
        show_progress: Show progress bars.
    """

    formats: List[str] = field(default_factory=lambda: ["console", "json"])
    save_path: Optional[str] = None
    verbose: bool = True
    show_progress: bool = True

    def get_save_path(self, timestamp: str, injection_name: str = "") -> str:
        """Get resolved save path with template substitution.

        Args:
            timestamp: Timestamp string for substitution.
            injection_name: Injection name for substitution.

        Returns:
            Resolved file path.
        """
        if self.save_path is None:
            return ""

        path = self.save_path
        path = path.replace("{timestamp}", timestamp)
        path = path.replace("{name}", injection_name)

        return path

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "OutputConfig":
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing output parameters.

        Returns:
            OutputConfig instance.
        """
        return cls(
            formats=config.get("formats", ["console", "json"]),
            save_path=config.get("save_path"),
            verbose=config.get("verbose", True),
            show_progress=config.get("show_progress", True),
        )


@dataclass
class EvaluationConfig:
    """Main configuration for fault injection evaluation.

    Attributes:
        name: Experiment name.
        description: Experiment description.
        checkpoint: Path to model checkpoint file.
        train_config: Path to training configuration YAML (optional, inferred from checkpoint if not provided).
        injections: List of injection configurations.
        baseline: Baseline evaluation configuration.
        runner: Runner configuration.
        output: Output configuration.
        seed: Random seed for reproducibility.
    """

    name: str = "evaluation"
    description: str = ""
    checkpoint: Optional[str] = None
    train_config: Optional[str] = None
    injections: List[InjectionConfig] = field(default_factory=list)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvaluationConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            EvaluationConfig instance.
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EvaluationConfig":
        """Create configuration from dictionary.

        Args:
            config: Dictionary containing evaluation parameters.

        Returns:
            EvaluationConfig instance.
        """
        return cls(
            name=config.get("name", "evaluation"),
            description=config.get("description", ""),
            checkpoint=config.get("checkpoint"),
            train_config=config.get("train_config"),
            injections=[
                InjectionConfig.from_dict(inj) for inj in config.get("injections", [])
            ],
            baseline=BaselineConfig.from_dict(config.get("baseline", {})),
            runner=RunnerConfig.from_dict(config.get("runner", {})),
            output=OutputConfig.from_dict(config.get("output", {})),
            seed=config.get("seed"),
        )

    def validate(self) -> None:
        """Validate evaluation configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.name:
            raise ValueError("Evaluation name cannot be empty")

        if not self.checkpoint:
            raise ValueError("Evaluation config must specify a checkpoint path")

        if not self.injections and not self.baseline.enabled:
            raise ValueError(
                "At least one injection configuration or baseline required"
            )

        for idx, injection in enumerate(self.injections):
            if not injection.name:
                raise ValueError(f"Injection {idx} must have a name")

        self.runner.validate()

    def get_enabled_injections(self) -> List[InjectionConfig]:
        """Get list of enabled injection configurations.

        Returns:
            List of enabled InjectionConfig instances.
        """
        enabled_injections = []
        for injection in self.injections:
            if injection.enabled:
                enabled_injections.append(injection)
        return enabled_injections

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        injections_list = []
        for injection in self.injections:
            injections_list.append(
                {
                    "name": injection.name,
                    "enabled": injection.enabled,
                    "target_type": injection.target_type,
                    "probability": injection.probability,
                    "injection_type": injection.injection_type,
                    "target_layers": injection.target_layers,
                    "track_statistics": injection.track_statistics,
                }
            )

        config_dict = {
            "name": self.name,
            "description": self.description,
            "checkpoint": self.checkpoint,
            "train_config": self.train_config,
            "injections": injections_list,
            "baseline": {
                "enabled": self.baseline.enabled,
                "num_runs": self.baseline.num_runs,
            },
            "runner": {
                "type": self.runner.type,
                "probabilities": self.runner.probabilities,
                "num_runs": self.runner.num_runs,
                "seeds": self.runner.seeds,
            },
            "output": {
                "formats": self.output.formats,
                "save_path": self.output.save_path,
                "verbose": self.output.verbose,
                "show_progress": self.output.show_progress,
            },
            "seed": self.seed,
        }
        return config_dict
