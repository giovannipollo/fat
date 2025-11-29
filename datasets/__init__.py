from typing import Dict, Type, Tuple
from torch.utils.data import DataLoader

from .base import BaseDataset
from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .mnist import MNISTDataset
from .fashion_mnist import FashionMNISTDataset


# Registry of available datasets
DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "mnist": MNISTDataset,
    "fashion_mnist": FashionMNISTDataset,
}


def register_dataset(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
    """
    Register a custom dataset class.
    
    Can be used as a decorator:
        @register_dataset
        class MyCustomDataset(BaseDataset):
            name = "my_dataset"
            ...
    
    Or called directly:
        register_dataset(MyCustomDataset)
    
    Args:
        dataset_class: Dataset class extending BaseDataset
        
    Returns:
        The registered dataset class (for decorator usage)
    """
    if not issubclass(dataset_class, BaseDataset):
        raise TypeError(f"{dataset_class.__name__} must extend BaseDataset")
    
    if dataset_class.name is None:
        raise ValueError(f"{dataset_class.__name__} must define a 'name' class attribute")
    
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def get_dataset(config: dict) -> BaseDataset:
    """
    Get dataset instance based on configuration.
    
    Args:
        config: Configuration dictionary with dataset settings
        
    Returns:
        BaseDataset: Initialized dataset instance
    """
    dataset_name = config["dataset"]["name"].lower()
    
    if dataset_name not in DATASETS:
        available = list(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    
    dataset_class = DATASETS[dataset_name]
    dataset_config = config["dataset"]
    
    return dataset_class(
        root=dataset_config["root"],
        batch_size=config["training"]["batch_size"],
        num_workers=dataset_config["num_workers"],
        download=dataset_config.get("download", True),
        val_split=dataset_config.get("val_split", None),
        seed=dataset_config.get("seed", 42),
    )


def get_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders based on configuration.
    
    Args:
        config: Configuration dictionary with dataset settings
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    dataset = get_dataset(config)
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
