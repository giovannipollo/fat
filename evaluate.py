"""Fault injection evaluation script.

This script evaluates the fault resilience of trained models by running
inference with fault injection enabled and tracking accuracy degradation.

Usage:
    python evaluate.py --eval-config configs/evaluation/sweep.yaml

Examples:
    # Probability sweep
    python evaluate.py --eval-config configs/evaluation/sweep.yaml

    # Combined weight + activation injection
    python evaluate.py --eval-config configs/evaluation/combined.yaml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict

# Suppress pkg_resources deprecation warning from brevitas
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning
)

import torch

from datasets import get_dataset
from models import get_model
from utils import get_device, load_config, set_seed

from evaluation import EvaluationConfig, Evaluator
from evaluation.runners import get_runner
from evaluation.reporters import get_reporters

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="brevitas")
warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate model fault resilience",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required argument
    parser.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Path to evaluation configuration YAML",
    )

    return parser.parse_args()


def load_model_and_dataset(
    train_config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
):
    """Load model and dataset from training config.

    Args:
        train_config: Training configuration dict
        checkpoint_path: Path to model checkpoint
        device: Compute device

    Returns:
        Tuple of (model, test_loader)
    """
    # Load dataset
    dataset = get_dataset(train_config)
    _, _, test_loader = dataset.get_loaders()
    print(f"Dataset: {dataset.name} ({len(test_loader)} test batches)")

    # Create model
    if "num_classes" not in train_config["model"]:
        train_config["model"]["num_classes"] = dataset.num_classes
    if "in_channels" not in train_config["model"]:
        train_config["model"]["in_channels"] = dataset.in_channels

    model = get_model(train_config)
    print(f"Model: {train_config['model']['name']}")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "best_acc" in checkpoint:
            print(f"Checkpoint accuracy: {checkpoint['best_acc']:.2f}%")
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device)

    return model, test_loader


def main() -> None:
    """Main entry point for fault injection evaluation."""
    args = parse_args()

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load evaluation configuration
    print(f"Loading evaluation config: {args.eval_config}")
    eval_config = EvaluationConfig.from_yaml(args.eval_config)

    # Validate evaluation config
    eval_config.validate()

    # Get checkpoint path
    checkpoint_path = eval_config.checkpoint
    if not checkpoint_path:
        raise ValueError("Evaluation config must specify 'checkpoint' field")
    print(f"Checkpoint: {checkpoint_path}")

    # Determine training config path
    if eval_config.train_config:
        train_config_path = eval_config.train_config
    else:
        # Infer train_config from checkpoint directory
        checkpoint_dir = Path(checkpoint_path).parent.parent
        train_config_path = str(checkpoint_dir / "config.yaml")
        print(
            f"Training config inferred from checkpoint directory: {train_config_path}"
        )

    # Load training configuration (for model/dataset)
    train_config: Dict[str, Any] = load_config(train_config_path)

    # Set seed for reproducibility
    if eval_config.seed is not None:
        set_seed(eval_config.seed, deterministic=True)
        print(f"Random seed: {eval_config.seed}")

    # Load model and dataset
    model, test_loader = load_model_and_dataset(train_config, checkpoint_path, device)

    # Create evaluator
    evaluator = Evaluator(
        config=eval_config,
        model=model,
        test_loader=test_loader,
        device=device,
    )

    # Check if baseline-only evaluation
    enabled_injections = eval_config.get_enabled_injections()
    if len(enabled_injections) == 0:
        print("\n" + "=" * 80)
        print("BASELINE-ONLY EVALUATION (No Fault Injection)")
        print("=" * 80)
    else:
        # Setup fault injectors
        print("\nFault Injection Configuration:")
        for injection in enabled_injections:
            print(f"  {injection.name}:")
            print(f"    Target: {injection.target_type}")
            print(f"    Type: {injection.injection_type}")
            print(f"    Probability: {injection.probability}%")
            print(f"    Layers: {injection.target_layers}")

    # Get runner
    runner = get_runner(config=eval_config, evaluator=evaluator)

    # Run evaluation
    print("\n" + "=" * 80)
    print(f"Starting evaluation: {eval_config.name}")
    print("=" * 80)

    results = runner.run()

    # Report results
    reporters = get_reporters(
        formats=eval_config.output.formats,
        save_path=eval_config.output.save_path,
        verbose=eval_config.output.verbose,
        show_progress=eval_config.output.show_progress,
    )

    for reporter in reporters:
        reporter.report(results)

    # Cleanup
    evaluator.cleanup()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
