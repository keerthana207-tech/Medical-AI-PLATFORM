import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import PathMNIST
from backend.configs.config import settings

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(model_type: str = "cnn"):
    """
    Get image transformations for training and evaluation.
    """
    if model_type.lower() == "vit":
        train_transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE_VIT, settings.IMAGE_SIZE_VIT)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        eval_transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE_VIT, settings.IMAGE_SIZE_VIT)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:  # cnn
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
    return train_transform, eval_transform

def get_dataloader(
    model_type: str = "cnn",
    split: str = "train",
    batch_size: int = settings.DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    shuffle: bool = None,
    subset_size: int = None
):
    """
    Get DataLoader for PathMNIST dataset.
    """
    train_transform, eval_transform = get_transforms(model_type)
    transform = train_transform if split == "train" else eval_transform
    
    if shuffle is None:
        shuffle = True if split == "train" else False
        
    # MedMNIST handles downloading internally
    dataset = PathMNIST(
        split=split,
        transform=transform,
        download=True,
        root=str(settings.DATASETS_DIR)
    )
    
    if subset_size is not None and subset_size < len(dataset):
        # Create a representative subset deterministically
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(dataset), generator=g)[:subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader

if __name__ == "__main__":
    # Test loading
    print("Testing PathMNIST DataLoader...")
    cnn_loader = get_dataloader(model_type="cnn", split="train", batch_size=8)
    images, labels = next(iter(cnn_loader))
    print(f"CNN Train Batch - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    vit_loader = get_dataloader(model_type="vit", split="train", batch_size=8)
    vit_images, vit_labels = next(iter(vit_loader))
    print(f"ViT Train Batch - Images shape: {vit_images.shape}, Labels shape: {vit_labels.shape}")
