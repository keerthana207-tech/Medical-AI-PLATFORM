import torch
import torch.nn as nn
import timm
from backend.configs.config import settings

class VisionTransformerWrapper(nn.Module):
    """
    Vision Transformer wrapper using timm models.
    Supports dynamic registration of hooks to capture self-attention weights from the last block.
    This enables explainability maps (Attention visualization/rollout) without modifying library code.
    """
    def __init__(self, model_name: str = settings.VIT_MODEL_NAME, pretrained: bool = True, num_classes: int = settings.NUM_CLASSES):
        super(VisionTransformerWrapper, self).__init__()
        # Load model from timm
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.attention_weights = None
        self._register_hooks()
        
    def _register_hooks(self):
        """
        Registers forward hook on the attention dropout module of the last Transformer block
        to capture the attention matrix (B, num_heads, num_tokens, num_tokens).
        """
        try:
            # Locate last block
            last_block = self.model.blocks[-1]
            # Locate attention dropout layer (where attention map resides before projection)
            attn_drop = last_block.attn.attn_drop
            
            def hook(module, input_tensor, output_tensor):
                # The first item in input_tensor is the attention matrix
                self.attention_weights = input_tensor[0]
                
            attn_drop.register_forward_hook(hook)
            print("Successfully registered forward hook on ViT attention layer.")
        except Exception as e:
            print(f"Warning: Could not register ViT attention hook: {e}")
            
    def get_last_self_attention(self) -> torch.Tensor:
        """
        Returns attention weights captured in the last forward pass.
        Shape: (B, num_heads, num_tokens, num_tokens)
        """
        return self.attention_weights
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

if __name__ == "__main__":
    # Test shape compatibility
    print("Testing VisionTransformerWrapper...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using pretrained=False for quick local testing without internet
    model = VisionTransformerWrapper(pretrained=False).to(device)
    model.eval()
    
    # Input size: batch of 1, 3 channels, 224x224 (required by vit_tiny_patch16_224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [1, 9])")
    
    attn_weights = model.get_last_self_attention()
    if attn_weights is not None:
        print(f"Captured Attention weights shape: {attn_weights.shape} (Expected: [1, 3, 197, 197])")
    else:
        print("Failed to capture attention weights.")
