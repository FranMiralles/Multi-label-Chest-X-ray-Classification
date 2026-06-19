from torch.utils.data import DataLoader
from torchvision import transforms

from src.config.config import DOWNLOAD
from src.data.dataset import get_mnist_dataset


def _make_loader(split: str, tf, batch_size: int = 64,
                 shuffle: bool = False, image_size: int = 224) -> DataLoader:
    dataset = get_mnist_dataset(split=split, transform=tf, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def get_dataloaders(
    train_tf: transforms.Compose | None = None,
    val_tf: transforms.Compose | None = None,
    batch_size: int = 64,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    train_loader = _make_loader('train', train_tf, batch_size, shuffle=True,
                                image_size=image_size)
    val_loader   = _make_loader('val',   val_tf,   batch_size, shuffle=False,
                                image_size=image_size)
    return train_loader, val_loader


def get_val_loader(val_tf: transforms.Compose | None = None,
                   batch_size: int = 64, image_size: int = 224) -> DataLoader:
    return _make_loader('val', val_tf, batch_size, image_size=image_size)


def get_test_loader(test_tf: transforms.Compose | None = None,
                    batch_size: int = 64, image_size: int = 224) -> DataLoader:
    return _make_loader('test', test_tf, batch_size, image_size=image_size)
