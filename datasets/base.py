"""Abstract base class for all dataset implementations.

Provides a unified interface for dataset loading, transformation,
and data loader creation. Subclasses must implement dataset-specific
metadata and the _load_dataset method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from typing import Any, List, Tuple, Optional, ClassVar, Union
import torch
import numpy as np
import random


def _worker_init_fn(worker_id: int) -> None:
    """Initialize data loader worker with deterministic seed.
    
    Ensures reproducibility when using num_workers > 0 by
    seeding each worker based on the main process's random state.
    
    Args:
        worker_id: The ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseDataset(ABC):
    """Abstract base class for all datasets.

    Subclasses must implement dataset-specific metadata and the
    _load_dataset method. The class provides automatic handling of:
    
    - Train/validation/test splits
    - Data transformations
    - DataLoader creation with reproducibility support
    
    Attributes:
        name: Dataset identifier string.
        num_classes: Number of output classes.
        in_channels: Number of input channels (1=grayscale, 3=RGB).
        image_size: Tuple of (height, width).
        mean: Tuple of channel means for normalization.
        std: Tuple of channel stds for normalization.
    
    Example:
        ```python
        class MyDataset(BaseDataset):
            name = "my_dataset"
            num_classes = 10
            in_channels = 3
            image_size = (32, 32)
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

            def _load_dataset(self, root, train, download, transform):
                return MyTorchDataset(root=root, train=train, transform=transform)
        ```
    """

    name: ClassVar[str]
    """Dataset identifier string (must be defined by subclasses)."""
    
    num_classes: ClassVar[int]
    """Number of classification classes."""
    
    in_channels: ClassVar[int]
    """Number of input image channels."""
    
    image_size: ClassVar[Tuple[int, int]]
    """Input image dimensions as (height, width)."""
    
    mean: ClassVar[Tuple[float, ...]]
    """Per-channel means for normalization."""
    
    std: ClassVar[Tuple[float, ...]]
    """Per-channel standard deviations for normalization."""

    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 256,
        num_workers: int = 16,
        download: bool = True,
        val_split: Optional[float] = None,
        seed: int = 42,
    ):
        """Initialize the dataset.
        
        Args:
            root: Root directory for dataset storage.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            download: Whether to download dataset if not present.
            val_split: Fraction of training data for validation (0.0-1.0), 
                None to disable.
            seed: Random seed for reproducible validation split.
        """
        self.root: str = root
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.download: bool = download
        self.val_split: Optional[float] = val_split
        self.seed: int = seed

        # Dataset instances (set by _load_all_datasets)
        self.train_dataset: Union[Dataset[Any], Subset[Any]]
        self.val_dataset: Optional[Union[Dataset[Any], Subset[Any]]] = None
        self.test_dataset: Dataset[Any]

        # Build transforms
        self.train_transform: transforms.Compose = self._build_train_transform()
        self.test_transform: transforms.Compose = self._build_test_transform()

        # Load datasets
        self._load_all_datasets()

    def _load_all_datasets(self) -> None:
        """Load train, validation, and test datasets.
        
        Handles validation split creation if val_split is specified.
        """
        # Load full training dataset
        full_train_dataset: Dataset[Any] = self._load_dataset(
            root=self.root,
            train=True,
            download=self.download,
            transform=self.train_transform,
        )

        # Handle validation split
        if self.val_split is not None and self.val_split > 0:
            # Calculate split sizes
            total_size: int = len(full_train_dataset)  # type: ignore[arg-type]
            val_size: int = int(total_size * self.val_split)
            train_size: int = total_size - val_size

            # Create reproducible split
            generator: torch.Generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, val_dataset_raw = random_split(
                full_train_dataset,
                [train_size, val_size],
                generator=generator,
            )

            # For validation, we need to apply test transforms
            # Create a wrapper that applies the correct transform
            self.val_dataset = self._create_val_dataset_with_transform(val_dataset_raw)
        else:
            self.train_dataset = full_train_dataset
            self.val_dataset = None

        # Load test dataset
        self.test_dataset = self._load_dataset(
            root=self.root,
            train=False,
            download=self.download,
            transform=self.test_transform,
        )

    def _create_val_dataset_with_transform(self, val_subset: Subset[Any]) -> Subset[Any]:
        """Create validation dataset with test transforms.
        
        Reloads training data with test transforms and selects
        the same indices as the validation split.
        
        Args:
            val_subset: The validation subset from random_split.
            
        Returns:
            Dataset with test transforms applied.
        """
        # Load training data again with test transforms
        train_with_test_transform: Dataset[Any] = self._load_dataset(
            root=self.root,
            train=True,
            download=False,  # Already downloaded
            transform=self.test_transform,
        )

        # Create a subset with the same indices
        return Subset(train_with_test_transform, val_subset.indices)

    def _build_train_transform(self) -> transforms.Compose:
        """Build training data transforms.
        
        Override this method to customize training augmentations.
        Default implementation applies ToTensor and normalization.
        
        Returns:
            Composed transforms for training data.
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _build_test_transform(self) -> transforms.Compose:
        """Build test/validation data transforms.
        
        Override this method to customize test transforms.
        Default implementation applies ToTensor and normalization.
        
        Returns:
            Composed transforms for test/validation data.
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    @abstractmethod
    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset[Any]:
        """Load the actual dataset (abstract method).
        
        Must be implemented by subclasses to return a PyTorch Dataset.
        
        Args:
            root: Root directory for dataset storage.
            train: Whether to load training or test set.
            download: Whether to download the dataset if not present.
            transform: Transforms to apply to the data.
            
        Returns:
            PyTorch Dataset instance.
        """
        pass

    def get_loaders(self) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]], DataLoader[Any]]:
        """Get train, validation, and test data loaders.

        Creates DataLoader instances with proper shuffling and
        worker initialization for reproducibility.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
            val_loader is None if val_split was not specified.
        """
        import os

        # Check if running in distributed mode
        is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        # Create a seeded generator for reproducible shuffling
        # This ensures the same batch order across runs with num_workers > 0
        generator: torch.Generator = torch.Generator().manual_seed(self.seed)

        # Use DistributedSampler for training in distributed mode
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                dataset=self.train_dataset
            )
            train_loader: DataLoader[Any] = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=_worker_init_fn,
            )
        else:
            train_loader: DataLoader[Any] = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                generator=generator,
                worker_init_fn=_worker_init_fn,
            )

        val_loader: Optional[DataLoader[Any]] = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=_worker_init_fn,
            )

        test_loader: DataLoader[Any] = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=_worker_init_fn,
        )

        return train_loader, val_loader, test_loader

    @property
    def has_validation(self) -> bool:
        """Check if validation set is available.
        
        Returns:
            True if validation dataset exists.
        """
        return self.val_dataset is not None

    def __repr__(self) -> str:
        val_info: str = f", val_split={self.val_split}" if self.val_split else ""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"num_classes={self.num_classes}, "
            f"in_channels={self.in_channels}, "
            f"image_size={self.image_size}"
            f"{val_info})"
        )
