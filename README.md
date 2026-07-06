# PathExplain AI: Colorectal Histology Analysis with Explainable AI

PathExplain AI is a production-quality medical machine learning web application that classifies colorectal cancer histology tissue patches from the **PathMNIST** dataset. It allows users to run comparative inferences using two distinct deep learning architectures—a custom **Convolutional Neural Network (CNN)** and a **Vision Transformer (ViT)**—while generating Explainable AI (XAI) overlays (**Grad-CAM** and **ViT Grad-CAM** maps) to explain model decisions.

---

## Technical Stack & Architecture

### Backend: FastAPI & PyTorch
- **API Server**: FastAPI asynchronous REST endpoints.
- **Inference Engines**: PyTorch (CPU-optimized architectures).
- **Explainability Layer**: 
  - **CNN**: Grad-CAM (computes activation gradients from the last Conv2d layer).
  - **Vision Transformer**: Grad-CAM for ViT (computes activation maps on the final transformer block's normalization layer using custom token reshaping).

### Frontend: React + Vite + Tailwind CSS v4
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
│   ├── attention.py       # ViT Grad-CAM map extraction
│   └── gradcam.py         # CNN Grad-CAM gradient overlays
├── main.py                # FastAPI entry point
├── utils/
│   └── __init__.py
│   └── heatmapAnalysis.py # Client-side helper for heatmap statistics
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
└── vit_best.pth           # Trained ViT weights (99.49% validation accuracy)
metrics/
├── cnn_metrics.json       # Training curves & test confusion matrix for CNN
└── vit_metrics.json       # Training curves & test confusion matrix for ViT
datasets/                  # Cached PathMNIST dataset downloads (Git ignored)
docker/                    # Dockerfiles and Nginx configurations
docker-compose.yml         # Container configuration
requirements.txt           # Python dependency file
.gitignore                 # Git ignore file
tests/                     # Pytest unit tests
```

---

## Performance Metrics

| Model | Test Accuracy | Precision (Weighted) | Recall | F1 Score | ROC-AUC | Inference Time | Parameters | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | **82.48%** | **0.8382** | **0.8248** | **0.8148** | **0.9766** | **0.53 ms** | 111,369 | 0.42 MB |
| **Vision Transformer** | **99.49%** | *Not Evaluated* | *Not Evaluated* | *Not Evaluated* | *Not Evaluated* | **68.05 ms** | 5,526,153 | 21.08 MB |

---

## How to Run Locally (Development)

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
cd frontend
npm install
npm run dev
```
*The React app will launch on: http://localhost:5173/*

### 3. Running Unit Tests
To run all tests cleanly without GUI/Tkinter errors, use the headless Matplotlib Agg backend configured in `conftest.py`:
```bash
.\venv\Scripts\python -m pytest
```

---

## Deployment Configuration

### Why GitHub Pages Cannot Host the Full Stack
GitHub Pages only hosts **static frontend assets** (HTML, CSS, JS). It cannot host or run backend applications (like our FastAPI Python server) or run heavy PyTorch machine learning models. 

Therefore, a full-stack deployment requires hosting the backend on a cloud server that supports Python execution and containerization (like GCP, AWS, or Render) and hosting the frontend on a CDN (like Vercel, Netlify, or Docker container).

### Deployment Option 1: Docker Compose (Recommended)
You can deploy the complete containerized stack using Docker Compose. This starts both the FastAPI backend (Uvicorn) and the React frontend (served via Nginx) under a single domain.

```bash
docker-compose up --build -d
```
*The application is then accessible at http://localhost:80.*

### Deployment Option 2: Separate Cloud Deployments

#### 1. Backend Deployment (e.g., GCP Cloud Run, AWS EC2, or Render)
- The backend is packaged using `docker/Dockerfile.backend`.
- **Environment Variables**:
  - `PORT`: Server port (defaults to `8000`).
- Deploy the Docker image to your container registry and run the container in the cloud.
- Update the frontend's API URL to point to the deployed backend.

#### 2. Frontend Deployment (e.g., Vercel, Netlify, or AWS S3)
- Generate static production assets:
  ```bash
  cd frontend
  npm run build
  ```
- Deploy the resulting `frontend/dist/` folder to Vercel, Netlify, or a similar static hosting provider.
- Configure base paths or redirect rules in Vite if you deploy to a custom subdirectory.
