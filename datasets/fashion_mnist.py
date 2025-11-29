import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class FashionMNISTDataset(BaseDataset):
    """FashionMNIST clothing dataset."""

    name = "fashion_mnist"
    num_classes = 10
    in_channels = 1
    image_size = (28, 28)
    mean = (0.2860,)
    std = (0.3530,)

    def _build_train_transform(self) -> transforms.Compose:
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
    ) -> Dataset:
        return torchvision.datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
