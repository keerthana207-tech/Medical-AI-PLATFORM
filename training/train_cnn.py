import os
import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.cnn import CustomCNN
from datasets.medmnist_loader import get_dataloader
from backend.configs.config import settings

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_cnn(
    epochs: int = settings.DEFAULT_EPOCHS,
    batch_size: int = settings.DEFAULT_BATCH_SIZE,
    lr: float = settings.DEFAULT_LR,
    patience: int = settings.EARLY_STOPPING_PATIENCE,
    resume: bool = False,
    num_workers: int = 0
):
    print("=" * 60)
    print("Training CNN Model on PathMNIST Dataset")
    print("=" * 60)
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Set Seed
    set_seed(settings.RANDOM_SEED)
    
    # 3. Load Dataloaders
    print("Loading data...")
    train_loader = get_dataloader(model_type="cnn", split="train", batch_size=batch_size, num_workers=num_workers)
    val_loader = get_dataloader(model_type="cnn", split="val", batch_size=batch_size, num_workers=num_workers)
    
    # 4. Instantiate Model
    model = CustomCNN(in_channels=3, num_classes=settings.NUM_CLASSES).to(device)
    
    # Calculate parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (num_params * 4) / (1024 * 1024)
    print(f"Model Name: {settings.CNN_MODEL_NAME}")
    print(f"Trainable Parameters: {num_params:,}")
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    # 5. Optimizer, Loss, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 6. Checkpoint Resuming
    start_epoch = 0
    best_val_loss = float('inf')
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    checkpoint_path = settings.CNN_CHECKPOINT_PATH
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            if 'history' in checkpoint:
                history = checkpoint['history']
            print(f"Successfully loaded checkpoint at epoch {start_epoch-1}. Best Val Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            
    # 7. Training Loop with Early Stopping
    epochs_no_improve = 0
    training_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # MedMNIST labels are often 2D (batch_size, 1), squeeze to 1D (batch_size)
            labels = labels.squeeze().long()
            if labels.dim() == 0:  # Handle single sample edge case
                labels = labels.unsqueeze(0)
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze().long()
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(epoch_val_loss)
        
        # Append history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Save Best Checkpoint and Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save weights
            print(f"--> Saving new best model to {checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'num_parameters': num_params,
                'model_size_mb': model_size_mb
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
    total_training_time = time.time() - training_start_time
    print(f"Training completed in {total_training_time/60:.2f} minutes.")
    
    # Save training metrics JSON
    metrics_path = settings.CNN_METRICS_PATH
    with open(metrics_path, 'w') as f:
        # We also write model info
        json.dump({
            "model_name": settings.CNN_MODEL_NAME,
            "training_time_seconds": total_training_time,
            "num_parameters": num_params,
            "model_size_mb": model_size_mb,
            "history": history
        }, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on PathMNIST")
    parser.add_argument("--epochs", type=int, default=settings.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=settings.DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=settings.DEFAULT_LR, help="Learning rate")
    parser.add_argument("--patience", type=int, default=settings.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    
    args = parser.parse_args()
    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        resume=args.resume,
        num_workers=args.workers
    )
