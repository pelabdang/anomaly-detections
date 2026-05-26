"""Feature extraction module for vibration sensor data.

Extracts time-domain and frequency-domain features from raw vibration signals,
creating a feature matrix suitable for anomaly detection models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract statistical and frequency features from time series windows.

    Converts raw vibration signals into a feature representation that captures
    the health state of mechanical components.

    Parameters
    ----------
    sample_rate : int
        Sampling rate of the vibration signal in Hz.
    """

    def __init__(self, sample_rate: int = 20480) -> None:
        self.sample_rate = sample_rate

    def extract_from_snapshots(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from multi-index DataFrame (timestamp, sample_idx).

        Parameters
        ----------
        data : pd.DataFrame
            Raw vibration data with multi-level index (timestamp, sample_idx).

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per timestamp.
        """
        timestamps = data.index.get_level_values("timestamp").unique()
        channels = [col for col in data.columns if col.startswith("ch")]

        features_list = []
        for ts in timestamps:
            snapshot = data.loc[ts]
            row_features = {}
            for channel in channels:
                signal = snapshot[channel].values
                channel_features = self._extract_single_channel(signal, channel)
                row_features.update(channel_features)
            features_list.append(row_features)

        features_df = pd.DataFrame(features_list, index=timestamps)
        features_df.index.name = "timestamp"

        logger.info(
            f"Extracted {features_df.shape[1]} features from "
            f"{features_df.shape[0]} snapshots"
        )
        return features_df

    def _extract_single_channel(
        self, signal: np.ndarray, channel_name: str
    ) -> dict[str, float]:
        """Extract all features from a single channel signal.

        Parameters
        ----------
        signal : np.ndarray
            1D vibration signal array.
        channel_name : str
            Name prefix for the features.

        Returns
        -------
        dict[str, float]
            Dictionary of feature name -> value pairs.
        """
        prefix = channel_name

        features = {}

        # Time-domain features
        features[f"{prefix}_rms"] = self._rms(signal)
        features[f"{prefix}_kurtosis"] = self._kurtosis(signal)
        features[f"{prefix}_skewness"] = self._skewness(signal)
        features[f"{prefix}_peak_to_peak"] = self._peak_to_peak(signal)
        features[f"{prefix}_crest_factor"] = self._crest_factor(signal)
        features[f"{prefix}_std"] = np.std(signal)
        features[f"{prefix}_variance"] = np.var(signal)
        features[f"{prefix}_mean_abs"] = np.mean(np.abs(signal))
        features[f"{prefix}_max_abs"] = np.max(np.abs(signal))

        # Frequency-domain features
        features[f"{prefix}_spectral_centroid"] = self._spectral_centroid(signal)
        features[f"{prefix}_spectral_bandwidth"] = self._spectral_bandwidth(signal)
        features[f"{prefix}_dominant_freq"] = self._dominant_frequency(signal)
        features[f"{prefix}_spectral_energy"] = self._spectral_energy(signal)

        return features

    @staticmethod
    def _rms(signal: np.ndarray) -> float:
        """Root Mean Square."""
        return float(np.sqrt(np.mean(signal**2)))

    @staticmethod
    def _kurtosis(signal: np.ndarray) -> float:
        """Kurtosis - measures 'tailedness' of distribution."""
        return float(stats.kurtosis(signal))

    @staticmethod
    def _skewness(signal: np.ndarray) -> float:
        """Skewness - measures asymmetry of distribution."""
        return float(stats.skew(signal))

    @staticmethod
    def _peak_to_peak(signal: np.ndarray) -> float:
        """Peak-to-peak amplitude."""
        return float(np.max(signal) - np.min(signal))

    @staticmethod
    def _crest_factor(signal: np.ndarray) -> float:
        """Crest factor = peak / RMS."""
        rms = np.sqrt(np.mean(signal**2))
        if rms == 0:
            return 0.0
        return float(np.max(np.abs(signal)) / rms)

    def _spectral_centroid(self, signal: np.ndarray) -> float:
        """Spectral centroid - 'center of mass' of the spectrum."""
        magnitude = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), d=1.0 / self.sample_rate)

        if np.sum(magnitude) == 0:
            return 0.0
        return float(np.sum(freqs * magnitude) / np.sum(magnitude))

    def _spectral_bandwidth(self, signal: np.ndarray) -> float:
        """Spectral bandwidth - spread of the spectrum around centroid."""
        magnitude = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), d=1.0 / self.sample_rate)

        if np.sum(magnitude) == 0:
            return 0.0

        centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)
        )
        return float(bandwidth)

    def _dominant_frequency(self, signal: np.ndarray) -> float:
        """Dominant frequency - frequency with highest magnitude."""
        magnitude = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), d=1.0 / self.sample_rate)

        if len(magnitude) == 0:
            return 0.0
        return float(freqs[np.argmax(magnitude)])

    def _spectral_energy(self, signal: np.ndarray) -> float:
        """Total spectral energy."""
        magnitude = np.abs(rfft(signal))
        return float(np.sum(magnitude**2))
