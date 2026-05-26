"""Tests for anomaly detection models."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderDetector


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def setup_method(self):
        """Create test data."""
        rng = np.random.default_rng(42)
        # Normal data
        self.X_train = rng.normal(0, 1, (200, 10))
        # Test data with some anomalies
        self.X_test = np.vstack([
            rng.normal(0, 1, (80, 10)),
            rng.normal(10, 1, (20, 10)),  # anomalies
        ])

    def test_fit_predict(self):
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        detector.fit(self.X_train)
        predictions = detector.predict(self.X_test)

        assert predictions.shape == (100,)
        assert set(np.unique(predictions)).issubset({-1, 1})

    def test_score_samples(self):
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        detector.fit(self.X_train)
        scores = detector.score_samples(self.X_test)

        assert scores.shape == (100,)
        # Anomalous points (last 20) should generally have lower scores
        normal_mean = np.mean(scores[:80])
        anomaly_mean = np.mean(scores[80:])
        assert anomaly_mean < normal_mean

    def test_detect(self):
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        detector.fit(self.X_train)
        results = detector.detect(self.X_test)

        assert "labels" in results
        assert "scores" in results
        assert "anomaly_ratio" in results
        assert "n_anomalies" in results
        assert 0 <= results["anomaly_ratio"] <= 1

    def test_not_fitted_raises(self):
        detector = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.predict(self.X_test)

    def test_save_load(self):
        detector = IsolationForestDetector(n_estimators=50, random_state=42)
        detector.fit(self.X_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            detector.save(path)

            loaded = IsolationForestDetector.load(path)
            original_scores = detector.score_samples(self.X_test)
            loaded_scores = loaded.score_samples(self.X_test)

            np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_get_params(self):
        detector = IsolationForestDetector(n_estimators=100, contamination=0.1)
        params = detector.get_params()

        assert params["model_type"] == "IsolationForest"
        assert params["n_estimators"] == 100
        assert params["contamination"] == 0.1


class TestLSTMAutoencoder:
    """Tests for LSTMAutoencoder model architecture."""

    def test_forward_pass(self):
        import torch

        model = LSTMAutoencoder(n_features=5, hidden_size=32, num_layers=1)
        x = torch.randn(4, 10, 5)  # batch=4, seq_len=10, features=5
        output = model(x)

        assert output.shape == x.shape

    def test_different_sequence_lengths(self):
        import torch

        model = LSTMAutoencoder(n_features=3, hidden_size=16, num_layers=2)

        for seq_len in [5, 10, 20, 50]:
            x = torch.randn(2, seq_len, 3)
            output = model(x)
            assert output.shape == x.shape


class TestLSTMAutoencoderDetector:
    """Tests for LSTMAutoencoderDetector."""

    def setup_method(self):
        """Create test sequences."""
        rng = np.random.default_rng(42)
        # Normal training sequences
        self.X_train = rng.normal(0, 0.1, (50, 10, 5))
        # Test sequences with anomalies
        self.X_test = np.concatenate([
            rng.normal(0, 0.1, (30, 10, 5)),
            rng.normal(5, 1.0, (10, 10, 5)),  # anomalies
        ])

    def test_fit_predict(self):
        detector = LSTMAutoencoderDetector(
            n_features=5,
            hidden_size=16,
            num_layers=1,
            epochs=5,
            batch_size=16,
            device="cpu",
        )
        detector.fit(self.X_train)
        predictions = detector.predict(self.X_test)

        assert predictions.shape == (40,)
        assert set(np.unique(predictions)).issubset({-1, 1})

    def test_detect(self):
        detector = LSTMAutoencoderDetector(
            n_features=5,
            hidden_size=16,
            num_layers=1,
            epochs=5,
            batch_size=16,
            device="cpu",
        )
        detector.fit(self.X_train)
        results = detector.detect(self.X_test)

        assert "labels" in results
        assert "errors" in results
        assert "threshold" in results
        assert "anomaly_ratio" in results
        assert results["threshold"] > 0

    def test_not_fitted_raises(self):
        detector = LSTMAutoencoderDetector(n_features=5, device="cpu")
        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.predict(self.X_test)

    def test_save_load(self):
        detector = LSTMAutoencoderDetector(
            n_features=5,
            hidden_size=16,
            num_layers=1,
            epochs=3,
            batch_size=16,
            device="cpu",
        )
        detector.fit(self.X_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            detector.save(tmpdir)
            loaded = LSTMAutoencoderDetector.load(tmpdir, device="cpu")

            original_results = detector.detect(self.X_test)
            loaded_results = loaded.detect(self.X_test)

            np.testing.assert_array_almost_equal(
                original_results["errors"], loaded_results["errors"]
            )

    def test_get_params(self):
        detector = LSTMAutoencoderDetector(
            n_features=10, hidden_size=64, num_layers=2, device="cpu"
        )
        params = detector.get_params()

        assert params["model_type"] == "LSTMAutoencoder"
        assert params["n_features"] == 10
        assert params["hidden_size"] == 64
