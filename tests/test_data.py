"""Tests for the data ingestion module."""

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import NASABearingDataLoader


class TestNASABearingDataLoader:
    """Tests for NASABearingDataLoader."""

    def test_init_valid_test_set(self):
        loader = NASABearingDataLoader(data_dir="data/raw", test_set="2nd_test")
        assert loader.test_set == "2nd_test"

    def test_init_invalid_test_set(self):
        with pytest.raises(ValueError, match="test_set must be one of"):
            NASABearingDataLoader(data_dir="data/raw", test_set="invalid")

    def test_channels_property(self):
        loader = NASABearingDataLoader(data_dir="data/raw", test_set="2nd_test")
        assert loader.channels == ["ch1", "ch2", "ch3", "ch4"]

    def test_generate_synthetic_data_shape(self):
        loader = NASABearingDataLoader(data_dir="data/raw")
        data = loader.generate_synthetic_data(
            n_snapshots=10, n_samples=100, n_channels=4
        )

        assert isinstance(data, pd.DataFrame)
        assert data.index.names == ["timestamp", "sample_idx"]
        # 10 snapshots * 100 samples = 1000 rows
        assert len(data) == 1000
        assert list(data.columns) == ["ch1", "ch2", "ch3", "ch4"]

    def test_generate_synthetic_data_degradation(self):
        """Verify synthetic data simulates degradation (amplitude increases)."""
        loader = NASABearingDataLoader(data_dir="data/raw")
        data = loader.generate_synthetic_data(
            n_snapshots=100, n_samples=512, n_channels=4
        )

        timestamps = data.index.get_level_values("timestamp").unique()

        # Early data should have lower RMS than late data
        early_data = data.loc[timestamps[5]]["ch1"].values
        late_data = data.loc[timestamps[95]]["ch1"].values

        early_rms = np.sqrt(np.mean(early_data**2))
        late_rms = np.sqrt(np.mean(late_data**2))

        assert late_rms > early_rms


class TestTimeSeriesPreprocessor:
    """Tests for TimeSeriesPreprocessor."""

    def test_prepare_features(self):
        from src.data.preprocessing import TimeSeriesPreprocessor

        # Create sample feature matrix
        n_samples = 100
        n_features = 10
        features_df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            index=[f"ts_{i:04d}" for i in range(n_samples)],
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        preprocessor = TimeSeriesPreprocessor(train_ratio=0.7)
        processed = preprocessor.prepare_features(features_df)

        assert processed.X_train.shape == (70, 10)
        assert processed.X_test.shape == (30, 10)
        assert len(processed.timestamps_train) == 70
        assert len(processed.timestamps_test) == 30

    def test_create_sequences(self):
        from src.data.preprocessing import TimeSeriesPreprocessor

        preprocessor = TimeSeriesPreprocessor()
        data = np.random.randn(50, 5)
        sequences = preprocessor.create_sequences(data, sequence_length=10)

        assert sequences.shape == (41, 10, 5)

    def test_create_sequences_too_short(self):
        from src.data.preprocessing import TimeSeriesPreprocessor

        preprocessor = TimeSeriesPreprocessor()
        data = np.random.randn(5, 3)

        with pytest.raises(ValueError, match="Data length"):
            preprocessor.create_sequences(data, sequence_length=10)
