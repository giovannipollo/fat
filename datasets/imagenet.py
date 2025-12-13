"""ImageNet-1K (ILSVRC2012) dataset implementation.

Provides the ImageNet-1K dataset with standard preprocessing
for training and validation.

Note:
    ImageNet must be manually downloaded from https://image-net.org/
    and extracted into the following structure:

    root/
    ├── train/
    │   ├── n01440764/
    │   │   ├── n01440764_10026.JPEG
    │   │   └── ...
    │   └── ...
    └── val/
        ├── n01440764/
        │   ├── ILSVRC2012_val_00000293.JPEG
        │   └── ...
        └── ...

See: https://image-net.org/challenges/LSVRC/2012/
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .base import BaseDataset


class ImageNetDataset(BaseDataset):
    """ImageNet-1K (ILSVRC2012) dataset.

    1,281,167 training images and 50,000 validation images across 1,000 classes.
    Standard preprocessing uses RandomResizedCrop(224) for training and
    Resize(256) + CenterCrop(224) for validation/test.

    Attributes:
        name: Dataset identifier ("imagenet").
        num_classes: Number of classes (1000).
        in_channels: Number of input channels (3 for RGB).
        image_size: Image dimensions (224, 224).
        mean: Per-channel normalization means (ImageNet statistics).
        std: Per-channel normalization standard deviations.

    Note:
        This dataset cannot be automatically downloaded. You must manually
        download ImageNet from https://image-net.org/ and organize it into
        train/ and val/ subdirectories.

    Example:
        ```python
        dataset = ImageNetDataset(
            root="/path/to/imagenet",
            batch_size=256,
            num_workers=8,
        )
        train_loader, val_loader, test_loader = dataset.get_loaders()
        ```
    """

    name: ClassVar[str] = "imagenet"
    """Dataset identifier."""

    num_classes: ClassVar[int] = 1000
    """Number of classification classes."""

    in_channels: ClassVar[int] = 3
    """Number of input channels (RGB)."""

    image_size: ClassVar[Tuple[int, int]] = (224, 224)
    """Image dimensions after preprocessing."""

    mean: ClassVar[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    """Per-channel normalization means (ImageNet statistics)."""

    std: ClassVar[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
    """Per-channel normalization stds (ImageNet statistics)."""

    def __init__(
        self,
        root: str = "./data/imagenet",
        batch_size: int = 256,
        num_workers: int = 8,
        download: bool = False,
        val_split: float | None = None,
        seed: int = 42,
    ):
        """Initialize the ImageNet dataset.

        Args:
            root: Root directory containing train/ and val/ subdirectories.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            download: Ignored (ImageNet cannot be auto-downloaded).
            val_split: Ignored (ImageNet has a dedicated validation set).
            seed: Random seed for reproducible data loading.

        Raises:
            FileNotFoundError: If train/ or val/ directories do not exist.
        """
        # Validate directory structure before calling parent __init__
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"ImageNet training directory not found: {train_dir}\n"
                f"Please download ImageNet from https://image-net.org/ and "
                f"extract it to {root}/"
            )

        if not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f"ImageNet validation directory not found: {val_dir}\n"
                f"Please download ImageNet from https://image-net.org/ and "
                f"extract it to {root}/"
            )

        # ImageNet has its own validation set, so we ignore val_split
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            download=False,  # Always False for ImageNet
            val_split=None,  # ImageNet has dedicated val set
            seed=seed,
        )

    def _build_train_transform(self) -> transforms.Compose:
        """Build training transforms with standard ImageNet preprocessing.

        Applies RandomResizedCrop(224) and RandomHorizontalFlip before
        normalization.

        Returns:
            Composed training transforms.
        """
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _build_test_transform(self) -> transforms.Compose:
        """Build validation/test transforms with standard ImageNet preprocessing.

        Applies Resize(256) and CenterCrop(224) before normalization.

        Returns:
            Composed validation/test transforms.
        """
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
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
        """Load ImageNet dataset using ImageFolder.

        Args:
            root: Root directory containing train/ and val/ subdirectories.
            train: Whether to load training or validation set.
            download: Ignored (ImageNet cannot be auto-downloaded).
            transform: Transforms to apply.

        Returns:
            ImageFolder Dataset instance.
        """
        if train:
            data_dir = os.path.join(root, "train")
        else:
            data_dir = os.path.join(root, "val")

        return ImageFolder(root=data_dir, transform=transform)
