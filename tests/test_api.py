import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from backend.main import app
from backend.configs.config import settings

client = TestClient(app)

def test_welcome_endpoint():
    """
    Test welcome index page.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health_endpoint():
    """
    Test API health check status.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data

def test_dataset_endpoint():
    """
    Test dataset specifications endpoint.
    """
    response = client.get("/api/dataset")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == settings.DATASET_NAME
    assert data["num_classes"] == settings.NUM_CLASSES
    assert len(data["classes"]) == settings.NUM_CLASSES

def test_metrics_endpoint():
    """
    Test metrics loading endpoint.
    """
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "cnn" in data
    assert "vit" in data
    assert "model_name" in data["cnn"]
    assert "model_name" in data["vit"]

def test_predict_endpoint_with_mock_image():
    """
    Test prediction endpoint with a generated dummy image patch.
    """
    # Create a 28x28 dummy RGB image in memory
    img = Image.new('RGB', (28, 28), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Upload simulated file
    files = {'file': ('test_image.png', img_byte_arr, 'image/png')}
    response = client.post("/api/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check output keys
    assert "filename" in data
    assert "predictions" in data
    assert "explainability" in data
    assert "image_id" in data
    
    # Verify model prediction structures
    predictions = data["predictions"]
    assert "cnn" in predictions
    assert "vit" in predictions
    assert "class_name" in predictions["cnn"]
    assert "confidence" in predictions["cnn"]
    
    # Verify explainability base64 payloads
    explainability = data["explainability"]
    assert "original_image_base64" in explainability
    assert "cnn_gradcam_base64" in explainability
    assert "vit_attention_base64" in explainability
