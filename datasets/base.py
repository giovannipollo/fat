from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from typing import Tuple, Optional
import torch
import numpy as np
import random


def _worker_init_fn(worker_id: int):
    """
    Initialize worker with deterministic seed based on worker_id.
    
    This ensures reproducibility when using num_workers > 0.
    The base seed comes from the main process's random state.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseDataset(ABC):
    """
    Abstract base class for all datasets.

    Subclasses must implement:
        - name: Dataset identifier
        - num_classes: Number of output classes
        - in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        - image_size: Tuple of (height, width)
        - mean: Tuple of mean values for normalization
        - std: Tuple of std values for normalization
        - _load_dataset(): Method to load the actual dataset

    Example:
        class MyCustomDataset(BaseDataset):
            name = "my_dataset"
            num_classes = 10
            in_channels = 3
            image_size = (32, 32)
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

            def _load_dataset(self, root, train, download, transform):
                return MyTorchDataset(root=root, train=train, download=download, transform=transform)
    """

    # Dataset metadata (must be defined by subclasses)
    name: str = None
    num_classes: int = None
    in_channels: int = None
    image_size: Tuple[int, int] = None
    mean: Tuple[float, ...] = None
    std: Tuple[float, ...] = None

    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 256,
        num_workers: int = 16,
        download: bool = True,
        val_split: Optional[float] = None,
        seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory for dataset storage
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            download: Whether to download the dataset if not present
            val_split: Fraction of training data to use for validation (0.0-1.0).
                       If None, no validation split is created.
            seed: Random seed for reproducible validation split
        """
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.val_split = val_split
        self.seed = seed

        # Build transforms
        self.train_transform = self._build_train_transform()
        self.test_transform = self._build_test_transform()

        # Load datasets
        self._load_all_datasets()

    def _load_all_datasets(self):
        """Load train, validation, and test datasets."""
        # Load full training dataset
        full_train_dataset = self._load_dataset(
            root=self.root,
            train=True,
            download=self.download,
            transform=self.train_transform,
        )

        # Handle validation split
        if self.val_split is not None and self.val_split > 0:
            # Calculate split sizes
            total_size = len(full_train_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            # Create reproducible split
            generator = torch.Generator().manual_seed(self.seed)
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

    def _create_val_dataset_with_transform(self, val_subset) -> Dataset:
        """
        Create validation dataset with test transforms.

        This reloads the training data with test transforms and selects
        the same indices as the validation split.
        """
        # Load training data again with test transforms
        train_with_test_transform = self._load_dataset(
            root=self.root,
            train=True,
            download=False,  # Already downloaded
            transform=self.test_transform,
        )

        # Create a subset with the same indices
        return torch.utils.data.Subset(train_with_test_transform, val_subset.indices)

    def _build_train_transform(self) -> transforms.Compose:
        """
        Build training data transforms.
        Override this method to customize training augmentations.

        Returns:
            transforms.Compose: Composed transforms for training data
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _build_test_transform(self) -> transforms.Compose:
        """
        Build test/validation data transforms.
        Override this method to customize test transforms.

        Returns:
            transforms.Compose: Composed transforms for test data
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
    ) -> Dataset:
        """
        Load the actual dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to load training or test set
            download: Whether to download the dataset
            transform: Transforms to apply

        Returns:
            Dataset: PyTorch dataset instance
        """
        pass

    def get_loaders(self) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
        """
        Get train, validation, and test dataloaders.

        Returns:
            Tuple[DataLoader, Optional[DataLoader], DataLoader]:
                (train_loader, val_loader, test_loader)
                val_loader is None if val_split was not specified
        """
        # Create a seeded generator for reproducible shuffling
        # This ensures the same batch order across runs with num_workers > 0
        generator = torch.Generator().manual_seed(self.seed)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
            worker_init_fn=_worker_init_fn,
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=_worker_init_fn,
            )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=_worker_init_fn,
        )

        return train_loader, val_loader, test_loader

    @property
    def has_validation(self) -> bool:
        """Check if validation set is available."""
        return self.val_dataset is not None

    def __repr__(self) -> str:
        val_info = f", val_split={self.val_split}" if self.val_split else ""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"num_classes={self.num_classes}, "
            f"in_channels={self.in_channels}, "
            f"image_size={self.image_size}"
            f"{val_info})"
        )
