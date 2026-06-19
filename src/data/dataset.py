import medmnist
from torchvision import transforms
from src.config.config import DOWNLOAD

def get_mnist_dataset(
    split: str = "train",
    transform: transforms.Compose | None = None,
    image_size: int = 224,
) -> medmnist.ChestMNIST:
    return medmnist.ChestMNIST(split=split, transform=transform, download=DOWNLOAD, size=image_size)