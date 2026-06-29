import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from models.cnn import CustomCNN
from models.vit import VisionTransformerWrapper
from datasets.medmnist_loader import IMAGENET_MEAN, IMAGENET_STD
from backend.configs.config import settings

def load_inference_model(model_type: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load model for inference only.
    """
    if model_type.lower() == "cnn":
        model = CustomCNN(in_channels=3, num_classes=settings.NUM_CLASSES)
    elif model_type.lower() == "vit":
        model = VisionTransformerWrapper(pretrained=False, num_classes=settings.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image, model_type: str) -> torch.Tensor:
    """
    Resize, convert to tensor, and normalize a PIL image based on model specifications.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    if model_type.lower() == "vit":
        transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE_VIT, settings.IMAGE_SIZE_VIT)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:  # cnn
        transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE_CNN, settings.IMAGE_SIZE_CNN)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    return transform(image).unsqueeze(0)

def predict_single_image(
    image: Image.Image,
    model_type: str = "cnn",
    checkpoint_path: str = None,
    device: str = "cpu"
) -> dict:
    """
    Predict labels for a single image.
    """
    device_obj = torch.device(device)
    
    if checkpoint_path is None:
        checkpoint_path = settings.CNN_CHECKPOINT_PATH if model_type.lower() == "cnn" else settings.VIT_CHECKPOINT_PATH
        
    # 1. Load model
    model = load_inference_model(model_type, checkpoint_path, device_obj)
    
    # 2. Preprocess image
    input_tensor = preprocess_image(image, model_type).to(device_obj)
    
    # 3. Inference
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
    end_time = time.perf_counter()
    
    inference_time_ms = (end_time - start_time) * 1000
    
    # 4. Get probabilities and predictions
    probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
    confidence = float(probabilities.max())
    class_idx = int(probabilities.argmax())
    class_name = settings.CLASS_NAMES[class_idx]
    
    # Simple confidence level interpretations
    if confidence >= settings.CONFIDENCE_HIGH:
        confidence_level = "High Confidence"
    elif confidence >= settings.CONFIDENCE_MEDIUM:
        confidence_level = "Medium Confidence"
    else:
        confidence_level = "Low Confidence"
        
    return {
        "class_index": class_idx,
        "class_name": class_name,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": {name: float(prob) for name, prob in zip(settings.CLASS_NAMES, probabilities)},
        "inference_time_ms": inference_time_ms
    }

if __name__ == "__main__":
    # Test execution
    print("Prediction utility script defined.")
