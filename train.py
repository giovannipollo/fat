import argparse

from datasets import get_dataset
from models import get_model
from utils import get_device, load_config, Trainer, set_seed


def main():
    parser = argparse.ArgumentParser(description="Training Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    seed_config = config.get("seed", {})
    if seed_config.get("enabled", False):
        seed = seed_config.get("value", 42)
        deterministic = seed_config.get("deterministic", False)
        set_seed(seed, deterministic)
        print(f"Random seed: {seed} (deterministic: {deterministic})")

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    dataset = get_dataset(config)
    train_loader, val_loader, test_loader = dataset.get_loaders()

    # Print dataset info
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
    print(f"Model: {config['model']['name']}")

    # Create trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        val_loader=val_loader,
    )

    trainer.train()


if __name__ == "__main__":
    main()
