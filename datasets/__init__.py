"""Dataset module initialization and registry.

This module provides a unified interface for loading datasets
through a registry pattern. Supported datasets include CIFAR-10, CIFAR-100,
MNIST, Fashion-MNIST, and ImageNet.

See Also:
    BaseDataset: Abstract base class interface.
    get_dataset: Main factory function.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type, Tuple, Optional
from torch.utils.data import DataLoader

from .base import BaseDataset
from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .mnist import MNISTDataset
from .fashion_mnist import FashionMNISTDataset
from .imagenet import ImageNetDataset


DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "mnist": MNISTDataset,
    "fashion_mnist": FashionMNISTDataset,
    "imagenet": ImageNetDataset,
}
"""Registry mapping dataset names to their implementation classes."""


def register_dataset(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
    """Register a custom dataset class in the global registry.
    
    Can be used as a decorator or called directly to add
    custom dataset implementations to the DATASETS registry.
    
    Args:
        dataset_class: Dataset class extending BaseDataset.
    
    Returns:
        The registered dataset class (for decorator usage).
        
    Raises:
        TypeError: If dataset_class does not extend BaseDataset.
        ValueError: If dataset_class has no 'name' attribute.
    
    Example:
        ```python
        @register_dataset
        class MyCustomDataset(BaseDataset):
            name = "my_dataset"
            ...
        ```
    """
    if not issubclass(dataset_class, BaseDataset):
        raise TypeError(f"{dataset_class.__name__} must extend BaseDataset")
    
    if dataset_class.name is None:
        raise ValueError(f"{dataset_class.__name__} must define a 'name' class attribute")
    
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def get_dataset(config: Dict[str, Any]) -> BaseDataset:
    """Factory function to create a dataset instance from configuration.
    
    Looks up the dataset name in the registry and instantiates
    it with parameters from the configuration dictionary.
    
    Args:
        config: Configuration dictionary containing dataset settings
            under config["dataset"].
    
    Returns:
        Initialized dataset instance.
        
    Raises:
        ValueError: If the dataset name is not found in the registry.
    """
    dataset_name: str = config["dataset"]["name"].lower()
    
    if dataset_name not in DATASETS:
        available: List[str] = list(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    dataset_class: Type[BaseDataset] = DATASETS[dataset_name]
    dataset_config: Dict[str, Any] = config["dataset"]
    
    return dataset_class(
        root=dataset_config["root"],
        batch_size=config["training"]["batch_size"],
        num_workers=dataset_config["num_workers"],
        download=dataset_config.get("download", True),
        val_split=dataset_config.get("val_split", None),
        seed=dataset_config.get("seed", 42),
    )


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Convenience function to get data loaders directly from configuration.
    
    Creates a dataset instance and returns its data loaders.
    
    Args:
        config: Configuration dictionary with dataset settings.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        
    Note:
        val_loader may be None if val_split is not specified.
    """
    dataset: BaseDataset = get_dataset(config)
    return dataset.get_loaders()


# Export public API
__all__ = [
    "BaseDataset",
    "DATASETS",
    "register_dataset",
    "get_dataset",
    "get_dataloaders",
    "CIFAR10Dataset",
    "CIFAR100Dataset", 
    "MNISTDataset",
    "FashionMNISTDataset",
    "ImageNetDataset",
]
