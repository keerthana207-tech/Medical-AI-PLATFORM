import io
import base64
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from models.vit import VisionTransformerWrapper

def generate_vit_attention_map(model: VisionTransformerWrapper, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Extracts self-attention weights from the last Transformer layer and generates a 2D map.
    """
    # Trigger forward pass to populate hook attention weights
    _ = model(input_tensor)
    
    # Retrieve captured attention weights: [B, num_heads, num_tokens, num_tokens]
    # For vit_tiny_patch16_224, shape will be [1, 3, 197, 197]
    attn_weights = model.get_last_self_attention()
    
    if attn_weights is None:
        raise ValueError("Attention weights were not captured by hooks. Make sure evaluation model is used.")
        
    # Extract attention weights for the CLS token (index 0) to all patch tokens (index 1 to end)
    # Shape: [num_heads, 196]
    cls_attention = attn_weights[0, :, 0, 1:]
    
    # Average across all heads
    # Shape: [196]
    mean_attention = cls_attention.mean(dim=0).cpu().data.numpy()
    
    # Normalize between 0 and 1
    mean_attention = mean_attention - np.min(mean_attention)
    if np.max(mean_attention) > 0:
        mean_attention = mean_attention / np.max(mean_attention)
        
    # Reshape to 14x14 grid (since 224x224 image divided by 16x16 patch size = 14x14 patches)
    attention_grid = mean_attention.reshape(14, 14)
    
    return attention_grid

def overlay_attention_map(original_image: Image.Image, attention_grid: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlays the 14x14 attention grid on top of the original image.
    """
    w, h = original_image.size
    
    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    # Show original
    ax.imshow(original_image)
    
    # Show attention map (colored overlay with 'inferno' or 'viridis' colormap)
    ax.imshow(attention_grid, cmap='inferno', alpha=alpha, extent=(0, w, h, 0))
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf).convert("RGB")

def get_attention_base64(model: VisionTransformerWrapper, input_tensor: torch.Tensor, original_image: Image.Image) -> str:
    """
    Computes attention maps and returns the overlaid image encoded in base64.
    """
    attention_grid = generate_vit_attention_map(model, input_tensor)
    overlaid = overlay_attention_map(original_image, attention_grid)
    
    # Encode to base64
    buffered = io.BytesIO()
    overlaid.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
