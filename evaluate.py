"""Fault injection evaluation script.

This script evaluates the fault resilience of trained models by running
inference with fault injection enabled and tracking the accuracy degradation.

Usage:
    python evaluate.py --config configs/<config>.yaml --checkpoint path/to/checkpoint.pth

Example:
    # Evaluate with default fault injection settings from config
    python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml --checkpoint checkpoints/best.pth
    
    # Override fault injection probability
    python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml --checkpoint checkpoints/best.pth --probability 10.0
    
    # Sweep across multiple probabilities
    python evaluate.py --config configs/quant_cnv_w2a2_cifar10.yaml --checkpoint checkpoints/best.pth --sweep 0,1,5,10,20,50
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from utils import get_device, load_config, set_seed
from utils.fault_injection import (
    FaultInjectionConfig,
    FaultInjector,
    FaultStatistics,
)


def evaluate_with_faults(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    injector: FaultInjector,
    statistics: Optional[FaultStatistics] = None,
    show_progress: bool = True,
) -> Tuple[float, int, int]:
    """Evaluate model with fault injection enabled.
    
    Args:
        model: Model to evaluate (with fault injection layers).
        loader: DataLoader for evaluation.
        device: Compute device.
        injector: Fault injector instance.
        statistics: Optional statistics tracker.
        show_progress: Whether to show progress bar.
    
    Returns:
        Tuple of (accuracy, correct_count, total_count).
    """
    model.eval()
    correct = 0
    total = 0
    
    # Enable injection for evaluation
    injector.set_enabled(model, True)

    
    if statistics is not None:
        injector.set_statistics(model, statistics)
    
    # Create progress bar
    if show_progress:
        pbar = tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True)
    else:
        pbar = loader
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy, correct, total


def evaluate_baseline(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    injector: Optional[FaultInjector] = None,
    show_progress: bool = True,
) -> Tuple[float, int, int]:
    """Evaluate model without fault injection (baseline).
    
    Args:
        model: Model to evaluate.
        loader: DataLoader for evaluation.
        device: Compute device.
        injector: Optional fault injector to disable.
        show_progress: Whether to show progress bar.
    
    Returns:
        Tuple of (accuracy, correct_count, total_count).
    """
    model.eval()
    correct = 0
    total = 0
    
    # Disable injection for baseline
    if injector is not None:
        injector.set_enabled(model, False)
    
    # Create progress bar
    if show_progress:
        pbar = tqdm(loader, desc="Baseline", leave=False, dynamic_ncols=True)
    else:
        pbar = loader
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy, correct, total


def run_probability_sweep(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    injector: FaultInjector,
    probabilities: List[float],
    num_layers: int,
    num_runs: int = 1,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run evaluation sweep across multiple fault probabilities.
    
    Args:
        model: Model to evaluate (with fault injection layers).
        loader: DataLoader for evaluation.
        device: Compute device.
        injector: Fault injector instance.
        probabilities: List of probabilities to sweep.
        num_layers: Number of fault injection layers.
        num_runs: Number of runs per probability (for averaging).
        show_progress: Whether to show progress bar.
    
    Returns:
        Dictionary with sweep results.
    """
    results: Dict[str, Any] = {
        "probabilities": probabilities,
        "num_runs": num_runs,
        "results": [],
    }
    
    for prob in probabilities:
        print(f"\nEvaluating with probability {prob}%...")
        injector.update_probability(model, prob)
        
        run_accuracies = []
        for run in range(num_runs):
            if num_runs > 1:
                print(f"  Run {run + 1}/{num_runs}...")
            
            stats = FaultStatistics(num_layers=num_layers)
            accuracy, _, _ = evaluate_with_faults(
                model, loader, device, injector, stats, show_progress
            )
            run_accuracies.append(accuracy)
        
        avg_accuracy = sum(run_accuracies) / len(run_accuracies)
        std_accuracy = (
            (sum((a - avg_accuracy) ** 2 for a in run_accuracies) / len(run_accuracies))
            ** 0.5
            if num_runs > 1
            else 0.0
        )
        
        results["results"].append({
            "probability": prob,
            "accuracies": run_accuracies,
            "mean_accuracy": avg_accuracy,
            "std_accuracy": std_accuracy,
        })
        
        print(f"  Accuracy: {avg_accuracy:.2f}% (std: {std_accuracy:.2f}%)")
    
    return results


def main() -> None:
    """Main entry point for fault injection evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model fault resilience",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=None,
        help="Override fault injection probability (0-100)",
    )
    parser.add_argument(
        "--injection-type",
        type=str,
        choices=["random", "lsb_flip", "msb_flip", "full_flip"],
        default=None,
        help="Override injection type",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full_model", "layer"],
        default=None,
        help="Override injection mode",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Comma-separated list of probabilities to sweep (e.g., '0,1,5,10,20')",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs per probability (for averaging)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    args = parser.parse_args()

    # Load configuration
    config: Dict[str, Any] = load_config(args.config)
    
    # Set seed for reproducibility
    seed_config: Dict[str, Any] = config.get("seed", {})
    if seed_config.get("enabled", False):
        seed: int = seed_config.get("value", 42)
        set_seed(seed, deterministic=True)
        print(f"Random seed: {seed}")
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = get_dataset(config)
    _, _, test_loader = dataset.get_loaders()
    print(f"Dataset: {dataset.name} ({len(test_loader)} test batches)")
    
    # Create model
    if "num_classes" not in config["model"]:
        config["model"]["num_classes"] = dataset.num_classes
    if "in_channels" not in config["model"]:
        config["model"]["in_channels"] = dataset.in_channels
    
    model = get_model(config)
    print(f"Model: {config['model']['name']}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "best_acc" in checkpoint:
            print(f"Checkpoint accuracy: {checkpoint['best_acc']:.2f}%")
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Setup fault injection configuration
    fi_config = config.get("fault_injection", {})
    fi_config["enabled"] = True
    fi_config["apply_during"] = "eval"
    
    # Apply command-line overrides
    if args.probability is not None:
        fi_config["probability"] = args.probability
    if args.injection_type is not None:
        fi_config["injection_type"] = args.injection_type
    if args.mode is not None:
        fi_config["mode"] = args.mode
    
    # Set defaults if not specified
    fi_config.setdefault("probability", 5.0)
    fi_config.setdefault("mode", "full_model")
    fi_config.setdefault("injection_type", "random")
    fi_config.setdefault("track_statistics", True)
    
    fault_config = FaultInjectionConfig.from_dict(fi_config)
    
    # Inject fault layers
    injector = FaultInjector()
    model = injector.inject(model, fault_config)
    num_layers = injector.get_num_layers(model)
    
    print(f"\nFault Injection Configuration:")
    print(f"  Injection layers: {num_layers}")
    print(f"  Applu During: {fault_config.apply_during}")
    print(f"  Injection type: {fault_config.injection_type}")
    if args.sweep:
        print(f"  Probability sweep: {args.sweep}")
    else:
        print(f"  Probability: {fault_config.probability}%")
    
    # Evaluate baseline (no faults)
    print("\n" + "=" * 60)
    print("Evaluating baseline (no faults)...")
    baseline_acc, _, _ = evaluate_baseline(
        model, test_loader, device, injector, not args.no_progress
    )
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    # Results dictionary
    results: Dict[str, Any] = {
        "config_file": args.config,
        "checkpoint": args.checkpoint,
        "model": config["model"]["name"],
        "dataset": dataset.name,
        "num_injection_layers": num_layers,
        "fault_config": {
            "apply_during": fault_config.apply_during,
            "injection_type": fault_config.injection_type,
        },
        "baseline_accuracy": baseline_acc,
    }
    
    # Run evaluation
    print("\n" + "=" * 60)
    
    if args.sweep:
        # Probability sweep
        probabilities = [float(p.strip()) for p in args.sweep.split(",")]
        sweep_results = run_probability_sweep(
            model, test_loader, device, injector,
            probabilities, num_layers, args.num_runs, not args.no_progress
        )
        results["sweep_results"] = sweep_results
        
        # Print summary
        print("\n" + "=" * 60)
        print("Sweep Summary:")
        print(f"{'Probability':>12} | {'Accuracy':>10} | {'Degradation':>12}")
        print("-" * 40)
        for r in sweep_results["results"]:
            degradation = baseline_acc - r["mean_accuracy"]
            print(f"{r['probability']:>11.1f}% | {r['mean_accuracy']:>9.2f}% | {degradation:>+11.2f}%")
    else:
        # Single probability evaluation
        print(f"Evaluating with {fault_config.probability}% fault probability...")
        statistics = FaultStatistics(num_layers=num_layers)
        fault_acc, _, _ = evaluate_with_faults(
            model, test_loader, device, injector, statistics, not args.no_progress
        )
        
        results["fault_probability"] = fault_config.probability
        results["fault_accuracy"] = fault_acc
        results["accuracy_degradation"] = baseline_acc - fault_acc
        
        print(f"\nResults:")
        print(f"  Baseline accuracy:   {baseline_acc:.2f}%")
        print(f"  Fault accuracy:      {fault_acc:.2f}%")
        print(f"  Accuracy degradation: {baseline_acc - fault_acc:+.2f}%")
        
        # Print statistics
        if fault_config.track_statistics:
            print("\n" + "-" * 60)
            statistics.print_report()
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        # Default output path
        checkpoint_dir = os.path.dirname(args.checkpoint)
        output_path = os.path.join(
            checkpoint_dir or ".",
            f"fault_eval_{fault_config.injection_type}.json"
        )
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
