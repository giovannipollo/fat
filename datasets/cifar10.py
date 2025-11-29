import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset with standard augmentations."""

    name = "cifar10"
    num_classes = 10
    in_channels = 3
    image_size = (32, 32)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    def _build_train_transform(self) -> transforms.Compose:
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
    ) -> Dataset:
        return torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
