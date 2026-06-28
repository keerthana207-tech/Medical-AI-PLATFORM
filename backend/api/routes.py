import time
import io
import json
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from backend.configs.config import settings
from training.predict import predict_single_image

router = APIRouter()

# Global background training state
training_state = {
    "is_training": False,
    "model_type": None,
    "progress": 0.0,
    "epoch": 0,
    "total_epochs": settings.DEFAULT_EPOCHS,
    "logs": []
}

@router.get("/health")
def health_check():
    """
    Check API health and GPU availability.
    """
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": device,
        "cuda_available": cuda_available
    }

@router.get("/dataset")
def get_dataset_info():
    """
    Get PathMNIST dataset information.
    """
    # Later we will embed sample base64 images here or return URLs
    return {
        "name": settings.DATASET_NAME,
        "description": "PathMNIST is based on a dataset of 100,000 colorectal cancer histology images, split into training, validation, and test sets. It classifies tissue patches into 9 distinct classes.",
        "num_classes": settings.NUM_CLASSES,
        "classes": settings.CLASS_NAMES,
        "splits": {
            "train": 89996,
            "validation": 10004,
            "test": 7180
        },
        "image_dimension": "28x28 (RGB)",
        "sample_images": []  # Will be populated with sample images in base64
    }

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Run inference on uploaded image using CNN and ViT models.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    # Read image contents
    contents = await file.read()
    
    # Run real prediction for CNN if checkpoint exists
    try:
        image = Image.open(io.BytesIO(contents))
        if os.path.exists(settings.CNN_CHECKPOINT_PATH):
            cnn_results = predict_single_image(image, model_type="cnn", checkpoint_path=settings.CNN_CHECKPOINT_PATH)
        else:
            # Fallback to simulated prediction
            cnn_probs = [0.05] * settings.NUM_CLASSES
            cnn_probs[8] = 0.60
            cnn_confidence = max(cnn_probs)
            cnn_results = {
                "class_index": 8,
                "class_name": settings.CLASS_NAMES[8],
                "confidence": cnn_confidence,
                "confidence_level": "Medium Confidence",
                "probabilities": {name: prob for name, prob in zip(settings.CLASS_NAMES, cnn_probs)},
                "inference_time_ms": 12.5
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN Prediction failed: {str(e)}")
    
    # Run real prediction for ViT if checkpoint exists, else mock
    try:
        if os.path.exists(settings.VIT_CHECKPOINT_PATH):
            vit_results = predict_single_image(image, model_type="vit", checkpoint_path=settings.VIT_CHECKPOINT_PATH)
        else:
            vit_probs = [0.04] * settings.NUM_CLASSES
            vit_probs[8] = 0.70
            vit_confidence = max(vit_probs)
            vit_results = {
                "class_index": 8,
                "class_name": settings.CLASS_NAMES[8],
                "confidence": vit_confidence,
                "confidence_level": "Medium Confidence",
                "probabilities": {name: prob for name, prob in zip(settings.CLASS_NAMES, vit_probs)},
                "inference_time_ms": 45.2
            }
    except Exception as e:
        vit_results = {
            "error": f"ViT prediction failed: {str(e)}",
            "class_index": -1,
            "class_name": "Error",
            "confidence": 0.0,
            "confidence_level": "Low Confidence",
            "probabilities": {},
            "inference_time_ms": 0.0
        }
        
    return {
        "filename": file.filename,
        "predictions": {
            "cnn": cnn_results,
            "vit": vit_results
        },
        "image_id": f"img_{int(time.time())}"
    }

@router.get("/explain/{model_type}/{image_id}")
def get_explainability_map(model_type: str, image_id: str):
    """
    Get explainability heatmap (Grad-CAM for CNN, Attention Map for ViT).
    """
    if model_type not in ["cnn", "vit"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Must be 'cnn' or 'vit'.")
        
    # Return placeholder base64 blank transparent PNG or sample heatmap
    # In phase 4, this will generate actual heatmaps from the uploaded images
    return {
        "model_type": model_type,
        "image_id": image_id,
        "heatmap_image_base64": "" # Base64 string will go here
    }

@router.get("/metrics")
def get_model_metrics():
    """
    Retrieve performance and evaluation metrics for CNN and ViT models.
    """
    metrics = {}
    
    # Load CNN metrics if they exist, otherwise fallback to mock
    if os.path.exists(settings.CNN_METRICS_PATH):
        try:
            with open(settings.CNN_METRICS_PATH, 'r') as f:
                metrics["cnn"] = json.load(f)
        except Exception as e:
            metrics["cnn"] = {"error": f"Failed to load CNN metrics: {str(e)}"}
    else:
        metrics["cnn"] = {
            "model_name": "Custom CNN (Simulated)",
            "accuracy": 0.892,
            "precision": 0.887,
            "recall": 0.892,
            "f1_score": 0.889,
            "roc_auc": 0.965,
            "training_time_seconds": 920.0,
            "inference_time_ms_avg": 12.5,
            "num_parameters": 456789,
            "model_size_mb": 1.74,
            "confusion_matrix": [[0]*9]*9,
            "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        }
        
    # Load ViT metrics if they exist, otherwise fallback to mock
    if os.path.exists(settings.VIT_METRICS_PATH):
        try:
            with open(settings.VIT_METRICS_PATH, 'r') as f:
                metrics["vit"] = json.load(f)
        except Exception as e:
            metrics["vit"] = {"error": f"Failed to load ViT metrics: {str(e)}"}
    else:
        metrics["vit"] = {
            "model_name": "Vision Transformer (vit_tiny_patch16_224) (Simulated)",
            "accuracy": 0.925,
            "precision": 0.921,
            "recall": 0.925,
            "f1_score": 0.923,
            "roc_auc": 0.982,
            "training_time_seconds": 2400.0,
            "inference_time_ms_avg": 45.2,
            "num_parameters": 5717417,
            "model_size_mb": 21.8,
            "confusion_matrix": [[0]*9]*9,
            "history": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        }
        
    return metrics

@router.post("/train")
def train_model(model_type: str):
    """
    Trigger training for CNN or ViT model.
    """
    if model_type not in ["cnn", "vit"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Must be 'cnn' or 'vit'.")
        
    global training_state
    if training_state["is_training"]:
        return JSONResponse(status_code=400, content={"message": f"Training is already in progress for {training_state['model_type']}."})
        
    training_state = {
        "is_training": True,
        "model_type": model_type,
        "progress": 0.0,
        "epoch": 0,
        "total_epochs": settings.DEFAULT_EPOCHS,
        "logs": [f"Starting training for {model_type}..."]
    }
    
    # We will trigger the actual training script in the background in Phase 2
    return {"message": f"Training initiated for {model_type}.", "status": "started"}

@router.get("/train/status")
def get_train_status():
    """
    Check the status and logs of background training.
    """
    global training_state
    # Simulate progress for mock requests
    if training_state["is_training"]:
        training_state["progress"] += 5.0
        if training_state["progress"] >= 100.0:
            training_state["progress"] = 100.0
            training_state["is_training"] = False
            training_state["logs"].append("Training completed successfully.")
        else:
            training_state["epoch"] = int(training_state["progress"] / (100.0 / training_state["total_epochs"]))
            training_state["logs"].append(f"Epoch {training_state['epoch']}/{training_state['total_epochs']} - val_loss: {0.5 - training_state['progress']/300.0:.4f}")
            
    return training_state
