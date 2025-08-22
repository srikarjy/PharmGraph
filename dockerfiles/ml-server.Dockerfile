# Specialized Dockerfile for ML serving
# Optimized for machine learning workloads with model serving capabilities

FROM python:3.11-slim as base

# Install ML-optimized system dependencies
RUN apt-get update && apt-get install -y \
    # Scientific computing libraries
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    libblas3 \
    libatlas3-base \
    # OpenMP for parallel processing
    libomp-dev \
    libgomp1 \
    # Essential tools
    curl \
    procps \
    netcat-openbsd \
    # For GPU support (optional)
    # nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for ML service
RUN groupadd -r mluser && useradd -r -g mluser -u 1002 mluser

# Set work directory
WORKDIR /app

# Copy and install ML dependencies
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-ml.txt

# Copy ML-specific source code
COPY src/ml/ src/ml/
COPY src/config/ src/config/
COPY src/storage/ src/storage/
COPY src/__init__.py src/

# Copy configuration and scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# Create model and cache directories with proper permissions
RUN mkdir -p /app/models /app/cache /app/logs /app/temp && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# ML-optimized environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # OpenMP and threading configuration
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMBA_NUM_THREADS=4 \
    # ML serving configuration
    ML_MODEL_CACHE_SIZE=5 \
    ML_BATCH_SIZE=32 \
    ML_TIMEOUT=60 \
    ML_MAX_WORKERS=2 \
    # Model serving settings
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_REGISTRY_URI=http://mlflow:5000 \
    # Service identification
    SERVICE_TYPE=ml-server

# Expose ML serving port
EXPOSE 8001

# Health check optimized for ML service (longer timeout for model loading)
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Use entrypoint for proper signal handling
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start ML server
CMD ["python", "-m", "src.ml.model_server", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "2", \
     "--model-cache-size", "5"]