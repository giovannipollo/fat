"""MNIST handwritten digits dataset implementation.

Provides the classic MNIST dataset of handwritten digits 0-9.

See: http://yann.lecun.com/exdb/mnist/
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class MNISTDataset(BaseDataset):
    """MNIST handwritten digits dataset.
    
    70,000 28x28 grayscale images of handwritten digits (0-9).
    No augmentation is applied by default.
    
    Attributes:
        name: Dataset identifier ("mnist").
        num_classes: Number of digit classes (10).
        in_channels: Number of input channels (1 for grayscale).
        image_size: Image dimensions (28, 28).
        mean: Normalization mean.
        std: Normalization standard deviation.
    
    Note:
        Classes: digits 0-9.
        60,000 training images, 10,000 test images.
    """

    name = "mnist"
    """Dataset identifier."""
    
    num_classes = 10
    """Number of digit classes."""
    
    in_channels = 1
    """Number of input channels (grayscale)."""
    
    image_size = (28, 28)
    """Image dimensions."""
    
    mean = (0.1307,)
    """Normalization mean."""
    
    std = (0.3081,)
    """Normalization std."""

    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset[Any]:
        """Load MNIST dataset from torchvision.
        
        Args:
            root: Root directory for dataset storage.
            train: Whether to load training or test set.
            download: Whether to download if not present.
            transform: Transforms to apply.
            
        Returns:
            MNIST Dataset instance.
        """
        return torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
