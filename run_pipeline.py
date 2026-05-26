"""Main entry point for the training pipeline."""

import logging
import sys
from pathlib import Path

import yaml

from src.models.train import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {path}, using defaults")
        return {}

    with open(path) as f:
        return yaml.safe_load(f)


def main():
    """Run the training pipeline."""
    logger.info("🚀 Starting Sensor Anomaly Detection Pipeline")
    logger.info("=" * 60)

    # Load config
    config = load_config()

    # Run pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run(use_synthetic=True)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESULTS SUMMARY")
    logger.info("=" * 60)

    if "isolation_forest" in results:
        r = results["isolation_forest"]
        logger.info(
            f"  Isolation Forest: {r['n_anomalies']} anomalies detected "
            f"({r['anomaly_ratio']:.2%})"
        )

    if "lstm_autoencoder" in results:
        r = results["lstm_autoencoder"]
        logger.info(
            f"  LSTM Autoencoder: {r['n_anomalies']} anomalies detected "
            f"({r['anomaly_ratio']:.2%})"
        )

    logger.info("=" * 60)
    logger.info("✅ Pipeline completed! Check MLflow UI for detailed results.")
    logger.info("   Run: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
