import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .base import BaseDataset


class MNISTDataset(BaseDataset):
    """MNIST handwritten digits dataset."""

    name = "mnist"
    num_classes = 10
    in_channels = 1
    image_size = (28, 28)
    mean = (0.1307,)
    std = (0.3081,)

    def _load_dataset(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: transforms.Compose,
    ) -> Dataset:
        return torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )
