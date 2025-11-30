"""Fashion-MNIST dataset implementation.

Provides the Fashion-MNIST dataset of clothing items
as a drop-in replacement for MNIST.

See: https://github.com/zalandoresearch/fashion-mnist
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class FashionMNISTDataset(BaseDataset):
    """Fashion-MNIST clothing dataset.
    
    70,000 28x28 grayscale images of clothing items in 10 categories.
    Training augmentation includes horizontal flip.
    
    Attributes:
        name: Dataset identifier ("fashion_mnist").
        num_classes: Number of clothing categories (10).
        in_channels: Number of input channels (1 for grayscale).
        image_size: Image dimensions (28, 28).
        mean: Normalization mean.
        std: Normalization standard deviation.
    
    Note:
        Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, 
        Sandal, Shirt, Sneaker, Bag, Ankle boot.
        60,000 training images, 10,000 test images.
    """

    name = "fashion_mnist"
    """Dataset identifier."""
    
    num_classes = 10
    """Number of clothing categories."""
    
    in_channels = 1
    """Number of input channels (grayscale)."""
    
    image_size = (28, 28)
    """Image dimensions."""
    
    mean = (0.2860,)
    """Normalization mean."""
    
    std = (0.3530,)
    """Normalization std."""

    def _build_train_transform(self) -> transforms.Compose:
        """Build training transforms with horizontal flip.
        
        Returns:
            Composed training transforms.
        """
        return transforms.Compose(
            [
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
        """Load Fashion-MNIST dataset from torchvision.
        
        Args:
            root: Root directory for dataset storage.
            train: Whether to load training or test set.
            download: Whether to download if not present.
            transform: Transforms to apply.
            
        Returns:
            Fashion-MNIST Dataset instance.
        """
        return torchvision.datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
