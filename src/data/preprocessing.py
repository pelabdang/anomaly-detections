"""Data preprocessing module for time series sensor data.

Handles normalization, windowing, and train/test splitting of sensor data
for both Isolation Forest and LSTM Autoencoder models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ProcessedData:
    """Container for preprocessed data splits."""

    X_train: np.ndarray
    X_test: np.ndarray
    timestamps_train: np.ndarray
    timestamps_test: np.ndarray
    scaler: StandardScaler


class TimeSeriesPreprocessor:
    """Preprocess time series data for anomaly detection models.

    Parameters
    ----------
    train_ratio : float
        Fraction of data to use for training (from the beginning,
        as early data is assumed to be normal).
    normalize : bool
        Whether to apply standard scaling.
    """

    def __init__(self, train_ratio: float = 0.7, normalize: bool = True) -> None:
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.scaler = StandardScaler()

    def prepare_features(self, features_df: pd.DataFrame) -> ProcessedData:
        """Prepare feature matrix for model training.

        Assumes data is ordered chronologically and early data represents
        normal operating conditions.

        Parameters
        ----------
        features_df : pd.DataFrame
            Feature matrix with timestamps as index.

        Returns
        -------
        ProcessedData
            Preprocessed and split data.
        """
        timestamps = features_df.index.values
        X = features_df.values.astype(np.float64)

        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Split chronologically
        split_idx = int(len(X) * self.train_ratio)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        ts_train = timestamps[:split_idx]
        ts_test = timestamps[split_idx:]

        # Normalize
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        return ProcessedData(
            X_train=X_train,
            X_test=X_test,
            timestamps_train=ts_train,
            timestamps_test=ts_test,
            scaler=self.scaler,
        )

    def create_sequences(
        self, data: np.ndarray, sequence_length: int = 30
    ) -> np.ndarray:
        """Create overlapping sequences for LSTM input.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (n_samples, n_features).
        sequence_length : int
            Number of time steps per sequence.

        Returns
        -------
        np.ndarray
            3D array of shape (n_sequences, sequence_length, n_features).
        """
        if len(data) < sequence_length:
            raise ValueError(
                f"Data length ({len(data)}) must be >= sequence_length ({sequence_length})"
            )

        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i : i + sequence_length])

        return np.array(sequences)
