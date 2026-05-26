"""Data ingestion module for NASA Bearing dataset.

The NASA Bearing dataset contains vibration signal data collected from bearings
run to failure. This module handles downloading, loading, and organizing the raw data.

Reference:
    Lee, J., Qiu, H., Yu, G., Lin, J. and Rexnord Technical Services (2007).
    IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NASABearingDataLoader:
    """Load and organize NASA Bearing dataset.

    The dataset consists of vibration signals from 4 bearings recorded
    at 20 kHz sample rate. Each file contains a 1-second vibration snapshot.

    Parameters
    ----------
    data_dir : str or Path
        Root directory for raw data files.
    test_set : str
        Which test set to use: '1st_test', '2nd_test', or '3rd_test'.
    """

    SAMPLE_RATE = 20480  # Hz
    CHANNELS = {
        "1st_test": ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"],
        "2nd_test": ["ch1", "ch2", "ch3", "ch4"],
        "3rd_test": ["ch1", "ch2", "ch3", "ch4"],
    }

    def __init__(self, data_dir: str | Path, test_set: str = "2nd_test") -> None:
        self.data_dir = Path(data_dir)
        self.test_set = test_set
        self._validate_test_set()

    def _validate_test_set(self) -> None:
        """Validate test set parameter."""
        valid_sets = {"1st_test", "2nd_test", "3rd_test"}
        if self.test_set not in valid_sets:
            raise ValueError(f"test_set must be one of {valid_sets}, got '{self.test_set}'")

    @property
    def channels(self) -> list[str]:
        """Return channel names for the selected test set."""
        return self.CHANNELS[self.test_set]

    def load_single_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single snapshot file.

        Parameters
        ----------
        filepath : Path
            Path to a single data file.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each bearing channel.
        """
        data = pd.read_csv(filepath, sep="\t", header=None)
        data.columns = self.channels[: data.shape[1]]
        return data

    def load_all_snapshots(self) -> dict[str, pd.DataFrame]:
        """Load all snapshot files from the test set directory.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping timestamp strings to DataFrames.
        """
        test_dir = self.data_dir / self.test_set
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {test_dir}. "
                f"Please download the NASA Bearing dataset first."
            )

        files = sorted(test_dir.glob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith(".")]

        if not files:
            raise FileNotFoundError(f"No data files found in {test_dir}")

        logger.info(f"Loading {len(files)} snapshot files from {test_dir}")

        snapshots = {}
        for filepath in files:
            timestamp = filepath.stem
            snapshots[timestamp] = self.load_single_file(filepath)

        logger.info(f"Successfully loaded {len(snapshots)} snapshots")
        return snapshots

    def load_as_dataframe(self) -> pd.DataFrame:
        """Load all snapshots and return as a single DataFrame with timestamp index.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with multi-level index (timestamp, sample).
        """
        snapshots = self.load_all_snapshots()

        frames = []
        for timestamp, df in snapshots.items():
            df = df.copy()
            df["timestamp"] = timestamp
            df["sample_idx"] = range(len(df))
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined.set_index(["timestamp", "sample_idx"], inplace=True)
        return combined

    def generate_synthetic_data(
        self, n_snapshots: int = 1000, n_samples: int = 2048, n_channels: int = 4
    ) -> pd.DataFrame:
        """Generate synthetic bearing vibration data for testing.

        Creates data that mimics real bearing degradation patterns:
        - Normal operation: low amplitude noise
        - Degradation: gradually increasing amplitude and kurtosis
        - Failure: high amplitude with impulse patterns

        Parameters
        ----------
        n_snapshots : int
            Number of time snapshots to generate.
        n_samples : int
            Number of samples per snapshot.
        n_channels : int
            Number of bearing channels.

        Returns
        -------
        pd.DataFrame
            Synthetic vibration data with features extracted.
        """
        rng = np.random.default_rng(42)
        channels = [f"ch{i+1}" for i in range(n_channels)]

        frames = []
        for i in range(n_snapshots):
            # Simulate degradation: amplitude increases over time
            progress = i / n_snapshots
            if progress < 0.7:
                # Normal operation
                amplitude = 0.1 + 0.05 * progress
                noise = rng.normal(0, amplitude, (n_samples, n_channels))
            elif progress < 0.9:
                # Early degradation
                amplitude = 0.2 + 0.5 * (progress - 0.7)
                noise = rng.normal(0, amplitude, (n_samples, n_channels))
                # Add impulse signals
                n_impulses = int(10 * (progress - 0.7) / 0.2)
                for _ in range(n_impulses):
                    pos = rng.integers(0, n_samples)
                    ch = rng.integers(0, n_channels)
                    noise[pos, ch] += rng.normal(0, amplitude * 5)
            else:
                # Failure mode
                amplitude = 0.5 + 2.0 * (progress - 0.9)
                noise = rng.normal(0, amplitude, (n_samples, n_channels))
                # Heavy impulse patterns
                n_impulses = int(50 * (progress - 0.9) / 0.1)
                for _ in range(n_impulses):
                    pos = rng.integers(0, n_samples)
                    ch = rng.integers(0, n_channels)
                    noise[pos, ch] += rng.normal(0, amplitude * 10)

            df = pd.DataFrame(noise, columns=channels)
            df["timestamp"] = f"snapshot_{i:04d}"
            df["sample_idx"] = range(n_samples)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined.set_index(["timestamp", "sample_idx"], inplace=True)
        return combined
