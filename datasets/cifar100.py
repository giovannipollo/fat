"""CIFAR-100 dataset implementation.

Provides the CIFAR-100 dataset with standard augmentations
(random crop with padding, horizontal flip) for training.

See: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class CIFAR100Dataset(BaseDataset):
    """CIFAR-100 dataset with standard augmentations.
    
    60,000 32x32 color images in 100 fine-grained classes,
    grouped into 20 superclasses. Training augmentations include 
    random crop and horizontal flip.
    
    Attributes:
        name: Dataset identifier ("cifar100").
        num_classes: Number of fine-grained classes (100).
        in_channels: Number of input channels (3 for RGB).
        image_size: Image dimensions (32, 32).
        mean: Per-channel normalization means.
        std: Per-channel normalization standard deviations.
    
    Note:
        100 fine-grained classes, 20 superclasses.
        500 training images per class, 100 test images per class.
    """

    name = "cifar100"
    """Dataset identifier."""
    
    num_classes = 100
    """Number of fine-grained classes."""
    
    in_channels = 3
    """Number of input channels (RGB)."""
    
    image_size = (32, 32)
    """Image dimensions."""
    
    mean = (0.5071, 0.4867, 0.4408)
    """Per-channel normalization means."""
    
    std = (0.2675, 0.2565, 0.2761)
    """Per-channel normalization stds."""

    def _build_train_transform(self) -> transforms.Compose:
        """Build training transforms with augmentation.
        
        Returns:
            Composed training transforms.
        """
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset[Any]:
        """Load CIFAR-100 dataset from torchvision.
        
        Args:
            root: Root directory for dataset storage.
            train: Whether to load training or test set.
            download: Whether to download if not present.
            transform: Transforms to apply.
            
        Returns:
            CIFAR-100 Dataset instance.
        """
        return torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
