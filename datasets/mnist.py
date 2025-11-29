"""!
@file datasets/mnist.py
@brief MNIST handwritten digits dataset implementation.

@details Provides the classic MNIST dataset of handwritten digits 0-9.

@see http://yann.lecun.com/exdb/mnist/
"""

from __future__ import annotations

from typing import Any

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class MNISTDataset(BaseDataset):
    """!
    @brief MNIST handwritten digits dataset.
    
    @details 70,000 28x28 grayscale images of handwritten digits (0-9).
    No augmentation is applied by default.
    
    @par Dataset Properties
    - 10 classes: digits 0-9
    - 60,000 training images, 10,000 test images
    - Image size: 28x28 grayscale
    """

    ## @var name
    #  @brief Dataset identifier
    name = "mnist"
    
    ## @var num_classes
    #  @brief Number of digit classes
    num_classes = 10
    
    ## @var in_channels
    #  @brief Number of input channels (grayscale)
    in_channels = 1
    
    ## @var image_size
    #  @brief Image dimensions
    image_size = (28, 28)
    
    ## @var mean
    #  @brief Normalization mean
    mean = (0.1307,)
    
    ## @var std
    #  @brief Normalization std
    std = (0.3081,)

    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset[Any]:
        """!
        @brief Load MNIST dataset from torchvision.
        
        @param root Root directory for dataset storage
        @param train Whether to load training or test set
        @param download Whether to download if not present
        @param transform Transforms to apply
        @return MNIST Dataset instance
        """
        return torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
