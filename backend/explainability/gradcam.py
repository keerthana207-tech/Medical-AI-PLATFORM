import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from models.cnn import CustomCNN

class GradCAM:
    """
    Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN.
    """
    def __init__(self, model: CustomCNN):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        # Last conv layer of our CustomCNN
        target_layer = self.model.get_last_conv_layer()
        
        def forward_hook(module, input_tensor, output_tensor):
            self.activations = output_tensor
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
        
    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generates Grad-CAM activation map for a specific class index.
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get activations and gradients
        gradients = self.gradients.cpu().data.numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().data.numpy()[0]  # [C, H, W]
        
        # Channel-wise mean gradients (importance weights)
        weights = np.mean(gradients, axis=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap

def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlays a heat map on top of the original image using a colormap.
    """
    # Resize original image to standard size if needed, but we'll work with original image dimensions
    w, h = original_image.size
    
    # Interpolate heatmap to original image size
    # We will use matplotlib to create a color overlay
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    
    # Plot original image
    ax.imshow(original_image)
    
    # Overlay heatmap
    ax.imshow(heatmap, cmap='jet', alpha=alpha, extent=(0, w, h, 0))
    
    # Save fig to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf).convert("RGB")

def get_gradcam_base64(model: CustomCNN, input_tensor: torch.Tensor, original_image: Image.Image, class_idx: int = None) -> str:
    """
    Executes Grad-CAM and returns the overlaid image encoded in base64.
    """
    cam = GradCAM(model)
    heatmap = cam.generate_heatmap(input_tensor, class_idx)
    overlaid = overlay_heatmap(original_image, heatmap)
    
    # Encode to base64
    buffered = io.BytesIO()
    overlaid.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
