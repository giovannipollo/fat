"""CIFAR-10 dataset implementation.

Provides the CIFAR-10 dataset with standard augmentations
(random crop with padding, horizontal flip) for training.

See: https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset with standard augmentations.
    
    60,000 32x32 color images in 10 classes, with 6,000 images per class.
    Training augmentations include random crop (32x32 with 4px padding) and
    horizontal flip.
    
    Attributes:
        name: Dataset identifier ("cifar10").
        num_classes: Number of classes (10).
        in_channels: Number of input channels (3 for RGB).
        image_size: Image dimensions (32, 32).
        mean: Per-channel normalization means.
        std: Per-channel normalization standard deviations.
    
    Note:
        Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
        50,000 training images, 10,000 test images.
    """

    name = "cifar10"
    """Dataset identifier."""
    
    num_classes = 10
    """Number of classification classes."""
    
    in_channels = 3
    """Number of input channels (RGB)."""
    
    image_size = (32, 32)
    """Image dimensions."""
    
    mean = (0.4914, 0.4822, 0.4465)
    """Per-channel normalization means."""
    
    std = (0.2470, 0.2435, 0.2616)
    """Per-channel normalization stds."""

    def _build_train_transform(self) -> transforms.Compose:
        """Build training transforms with augmentation.
        
        Applies random crop with 4px padding and horizontal flip
        before normalization.
        
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
        """Load CIFAR-10 dataset from torchvision.
        
        Args:
            root: Root directory for dataset storage.
            train: Whether to load training or test set.
            download: Whether to download if not present.
            transform: Transforms to apply.
            
        Returns:
            CIFAR-10 Dataset instance.
        """
        return torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
