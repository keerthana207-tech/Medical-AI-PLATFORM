import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
METRICS_DIR = BASE_DIR / "metrics"

# Create directories if they do not exist
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

class Settings:
    # Base Paths
    BASE_DIR: Path = BASE_DIR
    DATASETS_DIR: Path = DATASETS_DIR
    SAVED_MODELS_DIR: Path = SAVED_MODELS_DIR
    METRICS_DIR: Path = METRICS_DIR

    # App Settings
    PROJECT_NAME: str = "Medical AI Model Comparison and Explainability Platform"
    API_V1_STR: str = "/api"
    
    # Dataset Settings
    DATASET_NAME: str = "pathmnist"
    IMAGE_SIZE_CNN: int = 28
    IMAGE_SIZE_VIT: int = 224  # timm ViT models default to 224x224
    NUM_CLASSES: int = 9
    CLASS_NAMES: list = [
        "adipose", "background", "debris", "lymphocytes", "mucus",
        "smooth muscle", "normal colon mucosa", "cancer-associated stroma",
        "colorectal adenocarcinoma epithelium"
    ]
    
    # Model Configurations
    CNN_MODEL_NAME: str = "CustomCNN"
    VIT_MODEL_NAME: str = "vit_tiny_patch16_224"  # timm model name
    
    # Training Hyperparameters
    RANDOM_SEED: int = 42
    DEFAULT_BATCH_SIZE: int = 128
    DEFAULT_EPOCHS: int = 30
    DEFAULT_LR: float = 1e-3
    EARLY_STOPPING_PATIENCE: int = 5
    
    # Confidence Thresholds
    CONFIDENCE_HIGH: float = 0.8
    CONFIDENCE_MEDIUM: float = 0.5
    
    # Paths
    CNN_CHECKPOINT_PATH: str = str(SAVED_MODELS_DIR / "cnn_best.pth")
    VIT_CHECKPOINT_PATH: str = str(SAVED_MODELS_DIR / "vit_best.pth")
    CNN_METRICS_PATH: str = str(METRICS_DIR / "cnn_metrics.json")
    VIT_METRICS_PATH: str = str(METRICS_DIR / "vit_metrics.json")
    PREDICTION_HISTORY_PATH: str = str(METRICS_DIR / "prediction_history.json")

settings = Settings()
