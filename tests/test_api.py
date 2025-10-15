import pytest
from fastapi.testclient import TestClient
from app.main import app
import io
from PIL import Image

# Create test client
client = TestClient(app)


def create_test_image():
    """Create a simple test image in memory"""
    img = Image.new('RGB', (640, 480), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert data["model"] == "YOLOv8"


def test_detect_objects_with_valid_image():
    """Test object detection with a valid image"""
    img_bytes = create_test_image()
    
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "request_id" in data
    assert "detections" in data
    assert "count" in data
    assert "processing_time" in data
    assert isinstance(data["detections"], list)
    assert data["count"] == len(data["detections"])


def test_detect_with_confidence_parameter():
    """Test detection with custom confidence threshold"""
    img_bytes = create_test_image()
    
    response = client.post(
        "/api/v1/detect?confidence=0.7",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200


def test_detect_with_invalid_confidence():
    """Test that invalid confidence values are rejected"""
    img_bytes = create_test_image()
    
    # Confidence too low
    response = client.post(
        "/api/v1/detect?confidence=0.05",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 422  # Validation error
    
    # Confidence too high
    img_bytes = create_test_image()
    response = client.post(
        "/api/v1/detect?confidence=1.5",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 422


def test_detect_with_non_image_file():
    """Test that non-image files are rejected"""
    text_file = io.BytesIO(b"This is not an image")
    
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.txt", text_file, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"].lower()


def test_annotated_detection():
    """Test annotated image endpoint"""
    img_bytes = create_test_image()
    
    response = client.post(
        "/api/v1/detect/annotated",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    
    assert "model_type" in data
    assert "classes" in data
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) > 0