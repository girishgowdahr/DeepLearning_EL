import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.db import init_db

@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    init_db()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ForestGuard" in data["app"]

def test_api_models_status(client):
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "cnn" in data
    assert "yolo" in data
    from backend.models.model_registry import registry
    assert data["cnn"]["available"] == registry.cnn.available
    assert data["yolo"]["available"] == registry.yolo.available

def test_api_settings_roundtrip(client):
    # Read current settings
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    assert "yolo_conf_fire" in data
    assert "cnn_threshold" in data

    # Update settings
    new_val = 0.42
    update_res = client.put("/api/settings", json={"yolo_conf_fire": new_val})
    assert update_res.status_code == 200
    assert update_res.json()["settings"]["yolo_conf_fire"] == new_val

def test_api_camera_status(client):
    response = client.get("/api/camera/status")
    assert response.status_code == 200
    data = response.json()
    assert "active" in data
    assert "connected" in data

def test_api_alerts_test_endpoint(client):
    response = client.post("/api/alerts/test")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["alert"]["is_test"] == 1

def test_api_detect_image(client):
    # Create synthetic test frame (green forest canopy)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (34, 139, 34)
    _, buf = cv2.imencode(".jpg", img)

    files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
    response = client.post("/api/detect/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "decision" in data
    assert "annotated_image_b64" in data
