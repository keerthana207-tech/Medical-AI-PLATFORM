import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    Lightweight, high-performance CNN optimized for 28x28 images (MedMNIST).
    Designed to train efficiently on CPU/GPU.
    Includes forward hooks compatibility for Grad-CAM explainability.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 9):
        super(CustomCNN, self).__init__()
        
        # Convolutional Block 1: 28x28 -> 14x14
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 2: 14x14 -> 7x7
        self.features2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional Block 3 (Last Conv Layer): 7x7 -> 7x7
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Global Average Pooling to reduce spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer. Used for Grad-CAM hooks.
        """
        return self.features3[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    # Validate shape compatibility
    model = CustomCNN(in_channels=3, num_classes=9)
    dummy_input = torch.randn(1, 3, 28, 28)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [1, 9])")
    
    # Verify last conv layer function
    last_conv = model.get_last_conv_layer()
    print(f"Last convolutional layer: {last_conv}")
