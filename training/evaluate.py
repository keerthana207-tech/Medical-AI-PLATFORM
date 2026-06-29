import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from models.cnn import CustomCNN
from models.vit import VisionTransformerWrapper
from datasets.medmnist_loader import get_dataloader
from backend.configs.config import settings

def load_model(model_type: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Helper to load model based on type and checkpoint path.
    """
    if model_type.lower() == "cnn":
        model = CustomCNN(in_channels=3, num_classes=settings.NUM_CLASSES)
    elif model_type.lower() == "vit":
        model = VisionTransformerWrapper(pretrained=False, num_classes=settings.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # The checkpoint could be a state dict directly or a dictionary containing 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model_type: str = "cnn", num_workers: int = 0):
    print("=" * 60)
    print(f"Evaluating {model_type.upper()} Model on Test Set")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = settings.CNN_CHECKPOINT_PATH if model_type.lower() == "cnn" else settings.VIT_CHECKPOINT_PATH
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        return
        
    # Load model and test data
    model = load_model(model_type, checkpoint_path, device)
    test_loader = get_dataloader(model_type=model_type, split="test", batch_size=settings.DEFAULT_BATCH_SIZE, num_workers=num_workers)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    inference_times = []
    
    # Run evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
                
            # Time inference for single batch (average it to image level)
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) / images.size(0))
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    avg_inference_time_ms = np.mean(inference_times) * 1000
    
    # Calculate performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate multiclass ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Could not calculate ROC-AUC (perhaps class sample mismatch): {e}")
        roc_auc = 0.0
        
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Avg Inference Time (per image): {avg_inference_time_ms:.2f} ms")
    
    # Load existing metrics history if available (to merge history curves with final metrics)
    metrics_path = settings.CNN_METRICS_PATH if model_type.lower() == "cnn" else settings.VIT_METRICS_PATH
    history = {}
    num_params = 0
    model_size_mb = 0.0
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                saved_data = json.load(f)
                history = saved_data.get("history", {})
                num_params = saved_data.get("num_parameters", 0)
                model_size_mb = saved_data.get("model_size_mb", 0.0)
        except Exception as e:
            print(f"Error loading existing metrics: {e}")
            
    # Compile all metrics
    evaluation_results = {
        "model_name": "Custom CNN" if model_type.lower() == "cnn" else f"Vision Transformer ({settings.VIT_MODEL_NAME})",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "inference_time_ms_avg": float(avg_inference_time_ms),
        "num_parameters": int(num_params),
        "model_size_mb": float(model_size_mb),
        "confusion_matrix": cm.tolist(),
        "history": history
    }
    
    # Save back to JSON
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    print(f"All evaluation results written back to: {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on PathMNIST test set")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "vit"], help="Model type to evaluate")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    
    args = parser.parse_args()
    evaluate_model(model_type=args.model, num_workers=args.workers)
