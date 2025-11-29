"""!
@file datasets/__init__.py
@brief Dataset module initialization and registry.

@details This module provides a unified interface for loading datasets
through a registry pattern. Supported datasets include CIFAR-10, CIFAR-100,
MNIST, and Fashion-MNIST.

@see BaseDataset for the abstract base class interface
@see get_dataset for the main factory function
"""

from __future__ import annotations

from typing import Any, Dict, List, Type, Tuple, Optional
from torch.utils.data import DataLoader

from .base import BaseDataset
from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .mnist import MNISTDataset
from .fashion_mnist import FashionMNISTDataset


## @var DATASETS
#  @brief Registry mapping dataset names to their implementation classes.
DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "mnist": MNISTDataset,
    "fashion_mnist": FashionMNISTDataset,
}


def register_dataset(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
    """!
    @brief Register a custom dataset class in the global registry.
    
    @details Can be used as a decorator or called directly to add
    custom dataset implementations to the DATASETS registry.
    
    @code{.py}
    @register_dataset
    class MyCustomDataset(BaseDataset):
        name = "my_dataset"
        ...
    @endcode
    
    @param dataset_class Dataset class extending BaseDataset
    @return The registered dataset class (for decorator usage)
    @throws TypeError If dataset_class does not extend BaseDataset
    @throws ValueError If dataset_class has no 'name' attribute
    """
    if not issubclass(dataset_class, BaseDataset):
        raise TypeError(f"{dataset_class.__name__} must extend BaseDataset")
    
    if dataset_class.name is None:
        raise ValueError(f"{dataset_class.__name__} must define a 'name' class attribute")
    
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def get_dataset(config: Dict[str, Any]) -> BaseDataset:
    """!
    @brief Factory function to create a dataset instance from configuration.
    
    @details Looks up the dataset name in the registry and instantiates
    it with parameters from the configuration dictionary.
    
    @param config Configuration dictionary containing dataset settings
                  under config["dataset"]
    @return Initialized dataset instance
    @throws ValueError If the dataset name is not found in the registry
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
    """!
    @brief Convenience function to get data loaders directly from configuration.
    
    @details Creates a dataset instance and returns its data loaders.
    
    @param config Configuration dictionary with dataset settings
    @return Tuple of (train_loader, val_loader, test_loader)
    @note val_loader may be None if val_split is not specified
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
]
