# Medical AI Model Comparison and Explainability Platform

A production-quality web application that classifies colorectal cancer histology tissue patches from the **PathMNIST** dataset, compares predictions from a custom **Convolutional Neural Network (CNN)** and a **Vision Transformer (ViT)**, and visualizes Explainable AI (XAI) overlays (**Grad-CAM** and **Self-Attention maps**).

---

## Technical Stack & Architecture

### Backend: FastAPI & PyTorch
- **API Server**: FastAPI asynchronous REST endpoints.
- **Inference Engines**: PyTorch (CPU-optimized architectures).
- **Explainability Layer**: 
  - **CNN**: Grad-CAM (computes activation gradients from the last Conv2d layer).
  - **Vision Transformer**: Self-Attention maps (registers a forward hook on the last block's `attn_drop` module to extract CLS token attention scores across patches).

### Frontend: React + Tailwind CSS v4
- **Dashboard**: Modern dark-themed dashboard with glassmorphism panels.
- **Visualization**: Recharts interactive line charts for loss/accuracy tracking and responsive side-by-side XAI comparison grids.
- **Dataloader Scaffolding**: Structured state management with drag-and-drop file inputs.

### Containerization: Docker & Docker Compose
- **Orchestration**: Orchestrates backend (Uvicorn port 8000) and frontend (Nginx serving on port 80) together.

---

## Directory Structure

```
backend/
├── api/
│   ├── routes.py          # API endpoints (inference, metrics, health, etc.)
│   └── __init__.py
├── configs/
│   └── config.py          # Global path configurations and thresholds
├── explainability/
│   ├── attention.py       # ViT self-attention map extraction
│   └── gradcam.py         # CNN Grad-CAM gradient overlays
├── main.py                # FastAPI entry point
├── utils/
│   └── __init__.py
└── __init__.py
models/
├── cnn.py                 # Custom CNN architecture (optimized for 28x28 images)
├── vit.py                 # VisionTransformerWrapper (timm vit_tiny_patch16_224)
└── __init__.py
training/
├── evaluate.py            # Computes Test split accuracy, recall, confusion matrices
├── predict.py             # Inference helper for single PIL images
├── train_cnn.py           # Training script for CNN (with early stopping, scheduler)
├── train_vit.py           # Fast training script for ViT (backbone freezing & subset support)
└── __init__.py
saved_models/
├── cnn_best.pth           # Trained CNN weights (82.48% test accuracy)
└── vit_best.pth           # Trained ViT weights (58.06% test accuracy, 3 epochs on 5k subset)
metrics/
├── cnn_metrics.json       # Training curves & test confusion matrix for CNN
└── vit_metrics.json       # Training curves & test confusion matrix for ViT
datasets/                  # Cached PathMNIST dataset downloads (Git ignored)
docker/                    # Dockerfiles and Nginx configurations
docker-compose.yml         # Container configuration
requirements.txt           # Python dependency file
.gitignore                 # Git ignore file
```

---

## Performance Metrics

| Model | Test Accuracy | Precision (Weighted) | ROC-AUC | Inference Time | Parameters | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | **82.48%** | **0.838** | **0.976** | **0.53 ms** | 111,369 | 1.36 MB |
| **Vision Transformer** | **58.06%** | **0.603** | **0.874** | **68.05 ms** | 1,737 (trainable) | 22.18 MB |

---

## How to Run Locally

### 1. Backend Server Setup
Ensure Python 3.11 is installed, then execute:

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (reloading enabled)
$env:PYTHONPATH = "."; python backend/main.py
```
*API documentation will be available at: http://localhost:8000/docs*

### 2. Frontend Setup
Navigate to the `frontend/` directory and install the packages:

```bash
# Setup portable Node (if not globally installed) or run directly:
cd frontend
npm install
npm run dev
```
*The React app will launch on: http://localhost:5173/*

### 3. Docker Compose Setup
To launch the complete containerized stack:

```bash
docker-compose up --build
```
*Access the frontend at http://localhost:80 and API at http://localhost:8000.*

---

## Remaining Tasks for Full Completion

Here are the remaining tasks that need to be finished before concluding the final-year project:

1. **Unit Testing**:
   - Write pytest unit tests under `tests/` to validate model forward shapes and API response codes.
   - Run: `pytest tests/`
2. **Production UI Polish**:
   - Configure minor animation tweaks and responsive adjustments on tablet and mobile viewports.
3. **Cloud Deployment Configuration**:
   - Prepare variables for deploying to Render, Railway, or Google Cloud Run.
4. **Final Presentation & Presentation Documentation**:
   - Write user manual guides and compile architecture graphs in the `docs/` folder.
