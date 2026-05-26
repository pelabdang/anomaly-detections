"""Tests for the FastAPI application."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.models.isolation_forest import IsolationForestDetector


@pytest.fixture
def trained_model(tmp_path):
    """Create and save a trained model for testing."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (100, 10))

    detector = IsolationForestDetector(n_estimators=50, random_state=42)
    detector.fit(X_train)

    model_path = tmp_path / "model.joblib"
    detector.save(model_path)
    return model_path


@pytest.fixture
def client(trained_model):
    """Create test client with loaded model."""
    app = create_app(model_path=trained_model)
    return TestClient(app)


@pytest.fixture
def client_no_model():
    """Create test client without model."""
    app = create_app(model_path="/nonexistent/path")
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_with_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self, client_no_model):
        response = client_no_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    def test_predict_normal(self, client):
        response = client.post(
            "/predict",
            json={"features": [0.1] * 10, "timestamp": "2024-01-01T00:00:00"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data
        assert isinstance(data["is_anomaly"], bool)

    def test_predict_anomaly(self, client):
        response = client.post(
            "/predict",
            json={"features": [100.0] * 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_anomaly"] is True

    def test_predict_no_model(self, client_no_model):
        response = client_no_model.post(
            "/predict",
            json={"features": [0.1] * 10},
        )
        assert response.status_code == 503


class TestBatchPredictEndpoint:
    def test_batch_predict(self, client):
        readings = [
            {"features": [0.1] * 10, "timestamp": f"t{i}"}
            for i in range(5)
        ]
        response = client.post(
            "/predict/batch",
            json={"readings": readings},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 5
        assert "total_anomalies" in data
        assert "anomaly_ratio" in data


class TestModelInfoEndpoint:
    def test_model_info_loaded(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "IsolationForest"
        assert data["status"] == "loaded"

    def test_model_info_not_loaded(self, client_no_model):
        response = client_no_model.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_loaded"
