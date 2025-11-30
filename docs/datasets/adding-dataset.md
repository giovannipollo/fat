# Adding a New Dataset

This guide explains how to add a custom dataset to the framework.

## Overview

The framework uses a registry pattern for datasets. To add a new dataset:

1. Create a dataset class extending `BaseDataset`
2. Define required class attributes
3. Implement the `_load_dataset` method
4. Register the dataset in the registry

## Step-by-Step Guide

### 1. Create the Dataset Class

Create a new file in the `datasets/` directory (e.g., `datasets/my_dataset.py`):

```python
"""My custom dataset implementation."""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class MyDataset(BaseDataset):
    """My custom dataset.
    
    Describe your dataset here: source, size, characteristics, etc.
    
    Attributes:
        name: Dataset identifier ("my_dataset").
        num_classes: Number of classes.
        in_channels: Number of input channels.
        image_size: Image dimensions.
        mean: Per-channel normalization means.
        std: Per-channel normalization standard deviations.
    """

    name = "my_dataset"
    """Dataset identifier (used in config files)."""
    
    num_classes = 10
    """Number of classification classes."""
    
    in_channels = 3
    """Number of input channels (3=RGB, 1=grayscale)."""
    
    image_size = (32, 32)
    """Image dimensions (height, width)."""
    
    mean = (0.5, 0.5, 0.5)
    """Per-channel normalization means."""
    
    std = (0.5, 0.5, 0.5)
    """Per-channel normalization stds."""

    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset[Any]:
        """Load the dataset.
        
        Args:
            root: Root directory for dataset storage.
            train: Whether to load training or test set.
            download: Whether to download if not present.
            transform: Transforms to apply.
            
        Returns:
            Dataset instance.
        """
        # Return your dataset here
        # Example using a torchvision dataset:
        return torchvision.datasets.MyDataset(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
```

### 2. Required Class Attributes

Every dataset must define these class-level attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier used in config files |
| `num_classes` | `int` | Number of output classes |
| `in_channels` | `int` | Input channels (1=grayscale, 3=RGB) |
| `image_size` | `Tuple[int, int]` | Image dimensions (height, width) |
| `mean` | `Tuple[float, ...]` | Per-channel means for normalization |
| `std` | `Tuple[float, ...]` | Per-channel stds for normalization |

### 3. Custom Transforms (Optional)

Override transform methods for custom augmentation:

```python
def _build_train_transform(self) -> transforms.Compose:
    """Build training transforms with augmentation."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
    ])

def _build_test_transform(self) -> transforms.Compose:
    """Build test/validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
    ])
```

### 4. Register the Dataset

Add your dataset to the registry in `datasets/__init__.py`:

```python
from .my_dataset import MyDataset

DATASETS: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10Dataset,
    "cifar100": CIFAR100Dataset,
    "mnist": MNISTDataset,
    "fashion_mnist": FashionMNISTDataset,
    "my_dataset": MyDataset,  # Add your dataset here
}
```

**Alternative: Use the decorator**

```python
from datasets import register_dataset, BaseDataset

@register_dataset
class MyDataset(BaseDataset):
    name = "my_dataset"
    # ... rest of implementation
```

### 5. Update Exports

Add your class to `__all__` in `datasets/__init__.py`:

```python
__all__ = [
    "BaseDataset",
    "DATASETS",
    # ...
    "MyDataset",  # Add here
]
```

## Complete Example

Here's a complete example adding a custom SVHN dataset:

```python
"""SVHN (Street View House Numbers) dataset implementation."""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class SVHNDataset(BaseDataset):
    """SVHN dataset for digit recognition.
    
    Street View House Numbers - real-world digit recognition
    from Google Street View images.
    """

    name = "svhn"
    num_classes = 10
    in_channels = 3
    image_size = (32, 32)
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)

    def _build_train_transform(self) -> transforms.Compose:
        """Training transforms with augmentation."""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset[Any]:
        """Load SVHN dataset."""
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(
            root=root,
            split=split,
            download=download,
            transform=transform,
        )
```

## Using Custom Data Sources

For datasets not in torchvision, create a custom `Dataset` class:

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    """Custom dataset loading images from a directory."""
    
    def __init__(self, root: str, train: bool, transform=None):
        self.root = root
        self.transform = transform
        split = "train" if train else "test"
        self.image_dir = os.path.join(root, split)
        self.images = os.listdir(self.image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self._get_label(self.images[idx])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def _get_label(self, filename: str) -> int:
        # Implement your label extraction logic
        return int(filename.split("_")[0])
```

Then use it in your `BaseDataset` subclass:

```python
class MyCustomDataset(BaseDataset):
    name = "my_custom"
    # ... attributes ...
    
    def _load_dataset(self, root, train, download, transform):
        return CustomImageDataset(root=root, train=train, transform=transform)
```

## Using Your Dataset

Once registered, use your dataset in config files:

```yaml
dataset:
  name: "my_dataset"
  root: "./data"
  download: true
  num_workers: 4
  val_split: 0.1
```

The framework automatically:

- Passes `num_classes` and `in_channels` to the model
- Handles train/validation/test splits
- Creates DataLoaders with proper configuration

## Tips

1. **Calculate normalization stats**: Compute mean and std from your training data:
   ```python
   # Compute mean and std for your dataset
   loader = DataLoader(dataset, batch_size=64)
   mean = 0.0
   std = 0.0
   for images, _ in loader:
       mean += images.mean([0, 2, 3])
       std += images.std([0, 2, 3])
   mean /= len(loader)
   std /= len(loader)
   ```

2. **Test your dataset**: Verify it loads correctly:
   ```python
   from datasets import get_dataset
   
   config = {
       "dataset": {"name": "my_dataset", "root": "./data", "num_workers": 0},
       "training": {"batch_size": 32}
   }
   dataset = get_dataset(config)
   train_loader, val_loader, test_loader = dataset.get_loaders()
   
   # Check a batch
   images, labels = next(iter(train_loader))
   print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
   ```

3. **Handle missing downloads**: If your dataset doesn't support automatic download, implement manual download instructions or raise a helpful error.
