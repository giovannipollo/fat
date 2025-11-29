"""!
@file datasets/fashion_mnist.py
@brief Fashion-MNIST dataset implementation.

@details Provides the Fashion-MNIST dataset of clothing items
as a drop-in replacement for MNIST.

@see https://github.com/zalandoresearch/fashion-mnist
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class FashionMNISTDataset(BaseDataset):
    """!
    @brief Fashion-MNIST clothing dataset.
    
    @details 70,000 28x28 grayscale images of clothing items in 10 categories.
    Training augmentation includes horizontal flip.
    
    @par Dataset Properties
    - 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, 
      Sandal, Shirt, Sneaker, Bag, Ankle boot
    - 60,000 training images, 10,000 test images
    - Image size: 28x28 grayscale
    """

    ## @var name
    #  @brief Dataset identifier
    name = "fashion_mnist"
    
    ## @var num_classes
    #  @brief Number of clothing categories
    num_classes = 10
    
    ## @var in_channels
    #  @brief Number of input channels (grayscale)
    in_channels = 1
    
    ## @var image_size
    #  @brief Image dimensions
    image_size = (28, 28)
    
    ## @var mean
    #  @brief Normalization mean
    mean = (0.2860,)
    
    ## @var std
    #  @brief Normalization std
    std = (0.3530,)

    def _build_train_transform(self) -> transforms.Compose:
        """!
        @brief Build training transforms with horizontal flip.
        
        @return Composed training transforms
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
        """!
        @brief Load Fashion-MNIST dataset from torchvision.
        
        @param root Root directory for dataset storage
        @param train Whether to load training or test set
        @param download Whether to download if not present
        @param transform Transforms to apply
        @return Fashion-MNIST Dataset instance
        """
        return torchvision.datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
