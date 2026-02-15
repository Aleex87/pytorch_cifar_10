from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_transforms():
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
    
    return transform

def get_datasets():
    train_path = Path("data/processed/train")
    test_path = Path("data/processed/test")

    transform = get_transforms()

    full_train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    return full_train_dataset, test_dataset


def get_dataloaders(batch_size=32, val_split=0.1):
    
    full_train_dataset, test_dataset = get_datasets()

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # random_split divides the dataset into two parts
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # makes split reproducible
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader