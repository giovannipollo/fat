"""Main training script for the PyTorch Training Framework.

This script provides the entry point for training deep learning models
on various datasets (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST) using different
architectures (ResNet, VGG, MobileNet).

Usage:
    python train.py --config configs/<config>.yaml
    torchrun --nproc_per_node=NUM_GPUS train.py --config configs/<config>.yaml
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Any, Dict

import torch
import torch.distributed as dist

from datasets import get_dataset
from models import get_model
from utils import get_device, load_config, Trainer, set_seed
from utils.config_validator import validate_config, ConfigValidationError

# Suppress Brevitas pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="brevitas")


def main() -> None:
    """Main entry point for training.

    Parses command-line arguments, loads configuration, initializes
    the dataset and model, and starts the training process.

    The function performs the following steps:

    1. Parse command-line arguments for config file path
    2. Load YAML configuration
    3. Set random seed for reproducibility (if enabled)
    4. Setup compute device (CUDA/MPS/CPU)
    5. Load and prepare dataset with data loaders
    6. Create model based on configuration
    7. Initialize trainer and start training loop
    """
    parser = argparse.ArgumentParser(description="Training Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config: Dict[str, Any] = load_config(args.config)

    # Setup distributed training if using torchrun
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    print(f"Distributed training: {is_distributed}")
    print("RANK:", os.environ.get("RANK"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Validate configuration (only on rank 0)
    if rank == 0:
        try:
            validate_config(config)
        except ConfigValidationError as e:
            print(f"Configuration error: {e}")
            print("Please fix the configuration and try again.")
            import sys
            sys.exit(1)

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Set seed for reproducibility
    seed_config: Dict[str, Any] = config.get("seed", {})
    if seed_config.get("enabled", False):
        seed: int = seed_config.get("value", 42)
        deterministic: bool = seed_config.get("deterministic", False)
        set_seed(seed, deterministic)
        if rank == 0:
            print(f"Random seed: {seed} (deterministic: {deterministic})")
    # Setup device
    device = get_device()
    if rank == 0:
        print(f"Using device: {device}")
        if is_distributed:
            print(f"DDP: {world_size} GPUs")

    # Load dataset
    dataset = get_dataset(config)
    train_loader, val_loader, test_loader = dataset.get_loaders()
    print(f"Rank {rank}: Lenght of train_loader: {len(train_loader)}")

    # Print dataset info
    if rank == 0:
        print(
            f"Dataset: {dataset.name} ({dataset.num_classes} classes, {dataset.in_channels} channels)"
        )
        print(f"Train batches: {len(train_loader)}")
        if val_loader is not None:
            print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

    # Create model (use dataset metadata if not specified in config)
    if "num_classes" not in config["model"]:
        config["model"]["num_classes"] = dataset.num_classes
    if "in_channels" not in config["model"]:
        config["model"]["in_channels"] = dataset.in_channels

    model = get_model(config)
    if rank == 0:
        print(f"Model: {config['model']['name']}")

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        val_loader=val_loader,
        local_rank=local_rank if is_distributed else None,
    )

    trainer.train()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
