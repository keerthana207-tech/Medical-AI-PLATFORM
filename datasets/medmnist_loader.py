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

def download_dataset_if_not_exists():
    """
    Ensure the PathMNIST dataset file exists under settings.DATASETS_DIR.
    If not, download it using requests (more robust than torchvision's default downloader).
    """
    import hashlib
    import requests
    from medmnist.info import INFO
    
    dataset_name = "pathmnist"
    filename = f"{dataset_name}.npz"
    filepath = settings.DATASETS_DIR / filename
    
    # Create directories if they do not exist
    settings.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    info = INFO[dataset_name]
    url = info["url"]
    expected_md5 = info["MD5"]
    
    def check_md5(path, expected):
        md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest() == expected

    if filepath.exists():
        print(f"Checking MD5 for existing dataset file: {filepath}")
        try:
            if check_md5(filepath, expected_md5):
                print("Dataset file is valid. Skipping download.")
                return
            else:
                print("Existing dataset file is corrupted or outdated. Redownloading...")
                filepath.unlink()
        except Exception as e:
            print(f"Error checking existing file: {e}. Redownloading...")
            try:
                filepath.unlink()
            except Exception:
                pass
            
    print(f"Downloading PathMNIST dataset from {url} to {filepath}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Download with progress logging
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        # Print progress only at significant milestones
                        print(f"Download progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                            
        print("Download complete. Verifying integrity...")
        if check_md5(filepath, expected_md5):
            print("Verification successful!")
        else:
            raise RuntimeError("Downloaded file failed MD5 integrity check.")
            
    except Exception as e:
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception:
                pass
        print(f"Error downloading dataset: {e}")
        raise RuntimeError(f"Failed to download PathMNIST dataset: {e}")

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
        
    # Ensure dataset is downloaded using requests
    download_dataset_if_not_exists()
    
    dataset = PathMNIST(
        split=split,
        transform=transform,
        download=False,
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
