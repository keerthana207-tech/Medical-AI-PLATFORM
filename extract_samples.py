from medmnist import PathMNIST
import os
from datasets.medmnist_loader import download_dataset_if_not_exists

os.makedirs("sample_images", exist_ok=True)

# Ensure dataset is downloaded using requests
download_dataset_if_not_exists()

dataset = PathMNIST(
    split="test",
    download=False,
    root="datasets"
)


for i in range(10):
    img, label = dataset[i]

    # Convert NumPy array label to Python int
    label = label.item()

    img.save(f"sample_images/sample_{i}_class_{label}.png")

print("✅ Saved 10 images to sample_images/")