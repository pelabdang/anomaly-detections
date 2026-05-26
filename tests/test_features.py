"""Tests for feature extraction module."""

import numpy as np
import pandas as pd

from src.features.extraction import FeatureExtractor


class TestFeatureExtractor:
    """Tests for FeatureExtractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor(sample_rate=20480)

    def test_rms(self):
        signal = np.array([1.0, -1.0, 1.0, -1.0])
        assert self.extractor._rms(signal) == 1.0

    def test_rms_zero(self):
        signal = np.zeros(100)
        assert self.extractor._rms(signal) == 0.0

    def test_peak_to_peak(self):
        signal = np.array([-3.0, 0.0, 5.0, 1.0])
        assert self.extractor._peak_to_peak(signal) == 8.0

    def test_crest_factor(self):
        # Sine wave crest factor should be ~sqrt(2)
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * t)
        crest = self.extractor._crest_factor(signal)
        assert abs(crest - np.sqrt(2)) < 0.1

    def test_crest_factor_zero_signal(self):
        signal = np.zeros(100)
        assert self.extractor._crest_factor(signal) == 0.0

    def test_extract_from_snapshots(self):
        """Test full feature extraction pipeline."""
        # Create synthetic multi-index DataFrame
        n_snapshots = 5
        n_samples = 1024
        n_channels = 4

        rng = np.random.default_rng(42)
        frames = []
        for i in range(n_snapshots):
            df = pd.DataFrame(
                rng.normal(0, 1, (n_samples, n_channels)),
                columns=[f"ch{j+1}" for j in range(n_channels)],
            )
            df["timestamp"] = f"snap_{i:03d}"
            df["sample_idx"] = range(n_samples)
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)
        data.set_index(["timestamp", "sample_idx"], inplace=True)

        features_df = self.extractor.extract_from_snapshots(data)

        # Should have 5 rows (one per snapshot)
        assert features_df.shape[0] == n_snapshots
        # Should have 13 features * 4 channels = 52 features
        assert features_df.shape[1] == 13 * n_channels

    def test_spectral_features_deterministic(self):
        """Test that spectral features are deterministic."""
        rng = np.random.default_rng(123)
        signal = rng.normal(0, 1, 2048)

        centroid1 = self.extractor._spectral_centroid(signal)
        centroid2 = self.extractor._spectral_centroid(signal)
        assert centroid1 == centroid2

        bandwidth1 = self.extractor._spectral_bandwidth(signal)
        bandwidth2 = self.extractor._spectral_bandwidth(signal)
        assert bandwidth1 == bandwidth2
