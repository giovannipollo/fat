import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class CIFAR100Dataset(BaseDataset):
    """CIFAR-100 dataset with standard augmentations."""

    name = "cifar100"
    num_classes = 100
    in_channels = 3
    image_size = (32, 32)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

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
        return torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
