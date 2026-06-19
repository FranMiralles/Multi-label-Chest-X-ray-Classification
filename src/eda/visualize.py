import os

import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO

from src.config.config import DATA_FLAG, DOWNLOAD, REPORTS_DIR
from src.data.dataset import get_mnist_dataset

import torch
from monai.transforms import (
    Compose, EnsureChannelFirst, Lambda,
    RandAdjustContrast, RandRotate, RandZoom, Resize,
    ScaleIntensityRangePercentiles, ToTensor,
)

info = INFO[DATA_FLAG]


def plot_sample_images(
    size: int = 28,
    num_images: int = 10,
    split: str = 'train',
    save: bool = True,
) -> None:
    
    dataset = get_mnist_dataset(split=split, transform=None, image_size=size)
    labels_map = info['label']
    rows = 2 if num_images > 5 else 1
    cols = (num_images + rows - 1) // rows

    fig = plt.figure(figsize=(15, 6))
    for i in range(num_images):
        img = dataset.imgs[i]
        target = dataset.labels[i]
        pathologies = [labels_map[str(j)] for j, v in enumerate(target) if v == 1]
        title_img = ", ".join(pathologies) if pathologies else "Healthy"

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(title_img, fontsize=8, wrap=True)
        ax.axis('off')

    fig.tight_layout()
    fig.suptitle(f"ChestMNIST — {split} split ({size}×{size} px)",
                 fontsize=13, y=1.02)

    if save:
        fpath = os.path.join(REPORTS_DIR, f'eda_samples_{split}_{size}px.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig)


def plot_augmented_samples(num_images: int = 10, save: bool = True) -> None:
    """Grid de muestras tras el pipeline de augmentación MONAI exploratorio."""

    augment_tf = Compose([
        Lambda(lambda x: np.array(x).astype(np.float32)),
        EnsureChannelFirst(channel_dim='no_channel'),
        # Normalización por percentil — más robusta que ImageNet para imágenes médicas
        ScaleIntensityRangePercentiles(
            lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
        RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
        RandRotate(prob=0.2, range_x=7.0 * (np.pi / 180.0),
                   padding_mode="constant", mode="bilinear"),
        RandZoom(prob=0.2, min_zoom=0.9, max_zoom=1.1,
                 padding_mode="constant", mode="bilinear"),
        Resize(spatial_size=(224, 224)),
        Lambda(lambda x: np.repeat(x, 3, axis=0) if x.shape[0] == 1 else x),
        ToTensor(),
    ])

    dataset = medmnist.ChestMNIST(split='train', transform=augment_tf,
                                  download=DOWNLOAD, size=224)
    labels_map = info['label']
    rows = 2 if num_images > 5 else 1
    cols = (num_images + rows - 1) // rows

    fig = plt.figure(figsize=(15, 6))
    for i in range(num_images):
        img, target = dataset[i]
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        elif img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        pathologies = [labels_map[str(j)] for j, v in enumerate(target) if v == 1]
        label_text = ", ".join(pathologies) if pathologies else "Healthy"

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(label_text, fontsize=8, wrap=True)
        ax.axis('off')

    fig.tight_layout()
    fig.suptitle("ChestMNIST — MONAI augmentation pipeline (224×224)",
                 fontsize=13, y=1.02)

    if save:
        fpath = os.path.join(REPORTS_DIR, 'eda_augmented_samples.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig)


def plot_disease_prevalence(save: bool = True) -> None:
    """Bar chart de prevalencia de cada patología sobre los tres splits."""
    all_labels = np.vstack([
        medmnist.ChestMNIST(split=s, transform=None, download=DOWNLOAD, size=28).labels
        for s in ('train', 'val', 'test')
    ])

    label_map = info['label']
    label_names = [label_map[k] for k in sorted(label_map, key=int)] + ['No Finding']
    percentages = np.append(
        all_labels.mean(axis=0) * 100,
        (all_labels.sum(axis=1) == 0).mean() * 100,
    )

    colors = plt.cm.tab20(np.linspace(0, 1, len(label_names)))
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(label_names, percentages, color=colors)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Disease prevalence + healthy cases — ChestMNIST (all splits)')

    for bar, value in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()

    if save:
        fpath = os.path.join(REPORTS_DIR, 'eda_disease_prevalence.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig)


def plot_pathologies_per_image(save: bool = True) -> None:
    """Histograma del número de patologías co-ocurrentes por imagen."""
    all_labels = np.vstack([
        medmnist.ChestMNIST(split=s, transform=None, download=DOWNLOAD, size=28).labels
        for s in ('train', 'val', 'test')
    ])

    num_pathologies = all_labels.sum(axis=1).astype(int)
    counts = np.bincount(num_pathologies)
    x = np.arange(len(counts))
    colors = plt.cm.viridis(np.linspace(0, 1, len(counts)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, counts, color=colors)
    ax.set_xlabel('Number of co-occurring pathologies per image')
    ax.set_ylabel('Number of images')
    ax.set_title('Multi-label co-occurrence distribution — ChestMNIST (all splits)')
    ax.set_xticks(x)

    for bar, value in zip(bars, counts):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    str(value), ha='center', va='bottom', fontsize=9)

    fig.tight_layout()

    if save:
        fpath = os.path.join(REPORTS_DIR, 'eda_pathologies_per_image.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    plt.show()
    plt.close(fig)
