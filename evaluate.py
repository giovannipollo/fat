"""Fault injection evaluation script.

This script evaluates the fault resilience of trained models by running
inference with fault injection enabled and tracking accuracy degradation.

Usage:
    # YAML-based evaluation (recommended)
    python evaluate.py --config configs/train_config.yaml \\
                       --eval-config configs/evaluation/sweep.yaml \\
                       --checkpoint path/to/checkpoint.pth
    
    # Legacy CLI mode (backward compatible)
    python evaluate.py --config configs/train_config.yaml \\
                       --checkpoint checkpoints/best.pth \\
                       --probability 5.0 \\
                       --injection-type random

Examples:
    # Probability sweep
    python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml \\
                       --eval-config configs/evaluation/sweep.yaml \\
                       --checkpoint checkpoints/best.pth
    
    # Compare injection strategies
    python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml \\
                       --eval-config configs/evaluation/comparison.yaml \\
                       --checkpoint checkpoints/best.pth
    
    # Combined weight + activation injection
    python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml \\
                       --eval-config configs/evaluation/combined.yaml \\
                       --checkpoint checkpoints/best.pth
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import torch

from datasets import get_dataset
from models import get_model
from utils import get_device, load_config, set_seed

from evaluation import EvaluationConfig, Evaluator
from evaluation.runners import get_runner
from evaluation.reporters import get_reporters


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
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file (for model/dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Path to evaluation configuration YAML (recommended)",
    )
    
    # Legacy CLI options (backward compatibility)
    parser.add_argument(
        "--probability",
        type=float,
        default=5.0,
        help="Fault injection probability (0-100) [legacy mode]",
    )
    parser.add_argument(
        "--injection-type",
        type=str,
        choices=["random", "lsb_flip", "msb_flip", "full_flip"],
        default="random",
        help="Injection type [legacy mode]",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        choices=["activation", "weight"],
        default="activation",
        help="Injection target [legacy mode]",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated probabilities for sweep (e.g., '0,1,5,10') [legacy mode]",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def create_legacy_config(args: argparse.Namespace) -> EvaluationConfig:
    """Create evaluation config from legacy CLI arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        EvaluationConfig instance
    """
    from evaluation.config import InjectionConfig, RunnerConfig, OutputConfig
    
    # Create injection config
    injection = InjectionConfig(
        name=args.target_type,
        enabled=True,
        target_type=args.target_type,
        probability=args.probability,
        injection_type=args.injection_type,
        track_statistics=True,
    )
    
    # Determine runner type
    if args.sweep:
        runner_type = "sweep"
        probabilities = [float(p.strip()) for p in args.sweep.split(",")]
    else:
        runner_type = "single"
        probabilities = []
    
    # Create runner config
    runner = RunnerConfig(
        type=runner_type,
        probabilities=probabilities,
        num_runs=args.num_runs,
    )
    
    # Create output config
    output = OutputConfig(
        formats=["console", "json"],
        save_path=args.output,
        verbose=True,
        show_progress=not args.no_progress,
    )
    
    # Create evaluation config
    return EvaluationConfig(
        name=f"legacy_{args.target_type}_{args.injection_type}",
        description=f"Legacy CLI evaluation: {args.injection_type} on {args.target_type}",
        injections=[injection],
        runner=runner,
        output=output,
        seed=args.seed,
    )


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
    
    # Load training configuration (for model/dataset)
    train_config: Dict[str, Any] = load_config(args.config)
    
    # Load or create evaluation configuration
    if args.eval_config:
        # YAML-based evaluation
        print(f"Loading evaluation config: {args.eval_config}")
        eval_config = EvaluationConfig.from_yaml(args.eval_config)
        
        # Apply CLI overrides
        if args.num_runs > 1:
            eval_config.runner.num_runs = args.num_runs
        if args.output:
            eval_config.output.save_path = args.output
        if args.no_progress:
            eval_config.output.show_progress = False
        if args.seed is not None:
            eval_config.seed = args.seed
    else:
        # Legacy CLI mode
        print("Using legacy CLI mode (consider using --eval-config)")
        eval_config = create_legacy_config(args)
    
    # Validate evaluation config
    eval_config.validate()
    
    # Set seed for reproducibility
    if eval_config.seed is not None:
        set_seed(eval_config.seed, deterministic=True)
        print(f"Random seed: {eval_config.seed}")
    
    # Load model and dataset
    model, test_loader = load_model_and_dataset(
        train_config,
        args.checkpoint,
        device
    )
    
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
    runner = get_runner(eval_config, evaluator)
    
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
