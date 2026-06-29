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
    
def test_vit_dimensions_and_attention_hooks():
    """
    Test VisionTransformerWrapper forward pass and attention hooks.
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
    
    # Check captured attention weights
    attn_weights = model.get_last_self_attention()
    assert attn_weights is not None, "Hook failed to capture attention weights"
    
    # Expected shape: [B, num_heads, num_tokens, num_tokens]
    # For vit_tiny_patch16_224, num_heads = 3, num_tokens = 197
    assert attn_weights.dim() == 4, "Attention weights tensor must have 4 dimensions"
    assert attn_weights.shape[0] == 2, "Attention batch dimension does not match input"
    assert attn_weights.shape[1] == 3, "Expected 3 attention heads in vit_tiny"
    assert attn_weights.shape[2] == 197, "Expected 197 tokens in vit_tiny"
    assert attn_weights.shape[3] == 197, "Expected 197 tokens in vit_tiny"
