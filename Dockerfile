FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY run_pipeline.py .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install pyyaml httpx

# Expose API port
EXPOSE 8000

# Default command: run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
