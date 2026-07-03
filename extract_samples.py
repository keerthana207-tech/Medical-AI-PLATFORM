from medmnist import PathMNIST
import os

os.makedirs("sample_images", exist_ok=True)

dataset = PathMNIST(
    split="test",
    download=True,
    root="datasets"
)

for i in range(10):
    img, label = dataset[i]

    # Convert NumPy array label to Python int
    label = label.item()

    img.save(f"sample_images/sample_{i}_class_{label}.png")

print("✅ Saved 10 images to sample_images/")