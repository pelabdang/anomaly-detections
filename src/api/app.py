"""FastAPI service for real-time anomaly detection.

Provides REST endpoints for:
- Health checks
- Single-sample anomaly prediction
- Batch anomaly prediction
- Model information
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.isolation_forest import IsolationForestDetector

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================


class SensorReading(BaseModel):
    """Single sensor reading with features."""

    features: list[float] = Field(
        ...,
        description="List of extracted feature values",
        min_length=1,
    )
    timestamp: str | None = Field(None, description="Optional timestamp")


class BatchSensorReading(BaseModel):
    """Batch of sensor readings."""

    readings: list[SensorReading] = Field(
        ...,
        description="List of sensor readings",
        min_length=1,
    )


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    is_anomaly: bool
    anomaly_score: float
    timestamp: str | None = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictionResponse]
    total_anomalies: int
    anomaly_ratio: float


class ModelInfo(BaseModel):
    """Model information response."""

    model_type: str
    parameters: dict[str, Any]
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ============================================================================
# Application
# ============================================================================


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create FastAPI application with loaded model.

    Parameters
    ----------
    model_path : str or Path or None
        Path to saved model. If None, uses default path.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    # State container - load model eagerly
    state = {"model": None, "model_loaded": False}

    if model_path is None:
        model_path = Path("models/isolation_forest/model.joblib")

    resolved_path = Path(model_path)
    if resolved_path.exists():
        try:
            state["model"] = IsolationForestDetector.load(resolved_path)
            state["model_loaded"] = True
            logger.info(f"Model loaded from {resolved_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
    else:
        logger.warning(
            f"Model file not found at {resolved_path}. "
            f"API will start without a model."
        )

    app = FastAPI(
        title="Sensor Anomaly Detection API",
        description=(
            "Real-time anomaly detection for industrial sensor data "
            "using Isolation Forest and LSTM Autoencoder models."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=state["model_loaded"],
            version="1.0.0",
        )

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(reading: SensorReading):
        """Predict anomaly for a single sensor reading."""
        if not state["model_loaded"]:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train a model first.",
            )

        features = np.array(reading.features).reshape(1, -1)

        try:
            label = state["model"].predict(features)[0]
            score = state["model"].score_samples(features)[0]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Prediction failed: {str(e)}",
            )

        return PredictionResponse(
            is_anomaly=bool(label == -1),
            anomaly_score=float(score),
            timestamp=reading.timestamp,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(batch: BatchSensorReading):
        """Predict anomalies for a batch of sensor readings."""
        if not state["model_loaded"]:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train a model first.",
            )

        features = np.array([r.features for r in batch.readings])

        try:
            labels = state["model"].predict(features)
            scores = state["model"].score_samples(features)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Batch prediction failed: {str(e)}",
            )

        predictions = [
            PredictionResponse(
                is_anomaly=bool(label == -1),
                anomaly_score=float(score),
                timestamp=reading.timestamp,
            )
            for label, score, reading in zip(labels, scores, batch.readings)
        ]

        n_anomalies = int(np.sum(labels == -1))
        return BatchPredictionResponse(
            predictions=predictions,
            total_anomalies=n_anomalies,
            anomaly_ratio=n_anomalies / len(labels),
        )

    @app.get("/model/info", response_model=ModelInfo)
    async def model_info():
        """Get information about the loaded model."""
        if not state["model_loaded"]:
            return ModelInfo(
                model_type="none",
                parameters={},
                status="not_loaded",
            )

        return ModelInfo(
            model_type="IsolationForest",
            parameters=state["model"].get_params(),
            status="loaded",
        )

    return app


# Default application instance
app = create_app()
