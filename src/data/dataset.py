import medmnist
from torch.utils.data import DataLoader
from torchvision import transforms

def get_mnist_dataset(split: str = "train", tf: transforms = transforms.Compose(), batch_size: int = 64, image_size: int = 224, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    dataset = medmnist.ChestMNIST(split=split, transform=tf, download=DOWNLOAD, size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)