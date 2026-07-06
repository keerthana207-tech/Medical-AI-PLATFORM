import pytest
import torch
from models.cnn import CustomCNN
from models.vit import VisionTransformerWrapper
from backend.configs.config import settings

def test_cnn_dimensions():
    """
    Test CustomCNN forward pass shapes.
    """
    model = CustomCNN(in_channels=3, num_classes=settings.NUM_CLASSES)
    model.eval()
    
    # Batch size 4, 3 channels, 28x28 images
    dummy_input = torch.randn(4, 3, 28, 28)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (4, settings.NUM_CLASSES), f"Expected shape (4, {settings.NUM_CLASSES}), got {output.shape}"
    
def test_cnn_gradcam_layer():
    """
    Test CustomCNN last conv layer getter.
    """
    model = CustomCNN(in_channels=3, num_classes=settings.NUM_CLASSES)
    last_conv = model.get_last_conv_layer()
    assert isinstance(last_conv, torch.nn.Conv2d), "Expected Conv2d layer as target for Grad-CAM"
    
def test_vit_dimensions_and_gradcam_compatibility():
    """
    Test VisionTransformerWrapper forward pass and Grad-CAM target layer availability.
    """
    # Initialize wrapper without pretrained weights for fast offline unit testing
    model = VisionTransformerWrapper(pretrained=False, num_classes=settings.NUM_CLASSES)
    model.eval()
    
    # Batch size 2, 3 channels, 224x224 images (required by ViT)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Check predictions dimensions
    assert output.shape == (2, settings.NUM_CLASSES), f"Expected shape (2, {settings.NUM_CLASSES}), got {output.shape}"
    
    # Check that Grad-CAM target layer blocks exist
    assert hasattr(model.model, 'blocks'), "timm model is missing transformer blocks"
    assert len(model.model.blocks) > 0, "transformer blocks list is empty"
    assert hasattr(model.model.blocks[-1], 'norm1'), "timm model is missing norm1 block for Grad-CAM"
