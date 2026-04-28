"""
data.py
-------
Fashion-MNIST dataset loader with configurable input resolution.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 64,
    input_resolution: int = 28,
    val_fraction: float = 0.1,
    data_dir: str = "./data",
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, validation, and test DataLoaders for Fashion-MNIST.

    Args:
        batch_size:        Mini-batch size.
        input_resolution:  Spatial size to resize images to (default 28).
        val_fraction:      Fraction of training set used for validation.
        data_dir:          Root directory where the dataset is cached.
        num_workers:       Number of DataLoader worker processes.
        seed:              Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    transform_list = []

    if input_resolution != 28:
        transform_list.append(transforms.Resize((input_resolution, input_resolution)))

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST channel stats
    ]
    transform = transforms.Compose(transform_list)

    # Download training and test splits
    full_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Reproducible train/val split
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    # Pin memory only when CUDA is available to avoid overhead on CPU
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
