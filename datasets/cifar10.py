"""!
@file datasets/cifar10.py
@brief CIFAR-10 dataset implementation.

@details Provides the CIFAR-10 dataset with standard augmentations
(random crop with padding, horizontal flip) for training.

@see https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """!
    @brief CIFAR-10 dataset with standard augmentations.
    
    @details 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    Training augmentations include random crop (32x32 with 4px padding) and
    horizontal flip.
    
    @par Dataset Properties
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - 50,000 training images, 10,000 test images
    - Image size: 32x32 RGB
    """

    ## @var name
    #  @brief Dataset identifier
    name = "cifar10"
    
    ## @var num_classes
    #  @brief Number of classification classes
    num_classes = 10
    
    ## @var in_channels
    #  @brief Number of input channels (RGB)
    in_channels = 3
    
    ## @var image_size
    #  @brief Image dimensions
    image_size = (32, 32)
    
    ## @var mean
    #  @brief Per-channel normalization means
    mean = (0.4914, 0.4822, 0.4465)
    
    ## @var std
    #  @brief Per-channel normalization stds
    std = (0.2470, 0.2435, 0.2616)

    def _build_train_transform(self) -> transforms.Compose:
        """!
        @brief Build training transforms with augmentation.
        
        @details Applies random crop with 4px padding and horizontal flip
        before normalization.
        
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
        @brief Load CIFAR-10 dataset from torchvision.
        
        @param root Root directory for dataset storage
        @param train Whether to load training or test set
        @param download Whether to download if not present
        @param transform Transforms to apply
        @return CIFAR-10 Dataset instance
        """
        return torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
