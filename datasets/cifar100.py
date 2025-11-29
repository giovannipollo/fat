"""!
@file datasets/cifar100.py
@brief CIFAR-100 dataset implementation.

@details Provides the CIFAR-100 dataset with standard augmentations
(random crop with padding, horizontal flip) for training.

@see https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class CIFAR100Dataset(BaseDataset):
    """!
    @brief CIFAR-100 dataset with standard augmentations.
    
    @details 60,000 32x32 color images in 100 fine-grained classes,
    grouped into 20 superclasses. Training augmentations include 
    random crop and horizontal flip.
    
    @par Dataset Properties
    - 100 fine-grained classes, 20 superclasses
    - 500 training images per class, 100 test images per class
    - Image size: 32x32 RGB
    """

    ## @var name
    #  @brief Dataset identifier
    name = "cifar100"
    
    ## @var num_classes
    #  @brief Number of fine-grained classes
    num_classes = 100
    
    ## @var in_channels
    #  @brief Number of input channels (RGB)
    in_channels = 3
    
    ## @var image_size
    #  @brief Image dimensions
    image_size = (32, 32)
    
    ## @var mean
    #  @brief Per-channel normalization means
    mean = (0.5071, 0.4867, 0.4408)
    
    ## @var std
    #  @brief Per-channel normalization stds
    std = (0.2675, 0.2565, 0.2761)

    def _build_train_transform(self) -> transforms.Compose:
        """!
        @brief Build training transforms with augmentation.
        
        @return Composed training transforms
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
        """!
        @brief Load CIFAR-100 dataset from torchvision.
        
        @param root Root directory for dataset storage
        @param train Whether to load training or test set
        @param download Whether to download if not present
        @param transform Transforms to apply
        @return CIFAR-100 Dataset instance
        """
        return torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
