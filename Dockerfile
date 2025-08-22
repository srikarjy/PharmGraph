# Multi-stage Dockerfile for Pharmacogenomics ML Platform
# Optimized for production deployment with security best practices

# Base stage - System dependencies and build tools
FROM python:3.11-slim as base

# Set build arguments for metadata
ARG BUILD_DATE
ARG VERSION=latest
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Pharmacogenomics Team" \
      org.opencontainers.image.title="Pharmacogenomics ML Platform" \
      org.opencontainers.image.description="High-performance ML platform for pharmacogenomics research and clinical decision support" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/pharmacogenomics/ml-platform"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Scientific computing libraries
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libblas-dev \
    libatlas-base-dev \
    # System utilities
    pkg-config \
    curl \
    wget \
    git \
    # Health check utilities
    procps \
    net-tools \
    # Timezone and locale
    tzdata \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set timezone and locale
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# Dependencies stage - Install Python packages
FROM base as dependencies

# Set work directory
WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt requirements-ml.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-ml.txt

# Production stage - Application code and optimization
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    libblas3 \
    libatlas3-base \
    # Health check utilities
    curl \
    procps \
    # Timezone
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Set work directory
WORKDIR /app

# Copy virtual environment from dependencies stage
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser setup.py pyproject.toml ./

# Install the application
RUN pip install --no-cache-dir -e .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data /app/models /app/cache /app/temp \
    && chown -R appuser:appuser /app

# Copy health check script
COPY --chown=appuser:appuser docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Copy entrypoint script
COPY --chown=appuser:appuser docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Application settings
    APP_ENV=production \
    LOG_LEVEL=INFO \
    # Performance settings
    WORKERS=4 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100 \
    TIMEOUT=30

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Use entrypoint for proper signal handling
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ML Serving stage - Specialized for ML workloads
FROM production as ml-serving

# Switch back to root for package installation
USER root

# Install ML-specific system dependencies
RUN apt-get update && apt-get install -y \
    # OpenMP for parallel processing
    libomp-dev \
    # Additional ML libraries
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Switch back to appuser
USER appuser

# Set ML-specific environment variables
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMBA_NUM_THREADS=4 \
    # ML serving settings
    MODEL_CACHE_SIZE=5 \
    MODEL_BATCH_SIZE=32 \
    MODEL_TIMEOUT=60

# Expose ML serving port
EXPOSE 8001

# ML-specific health check
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ML serving command
CMD ["python", "-m", "src.ml.model_server", "--host", "0.0.0.0", "--port", "8001"]

# Development stage - Includes development tools
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    less \
    # Debugging tools
    strace \
    gdb \
    # Network tools
    netcat-openbsd \
    telnet \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for development
RUN groupadd -r devuser && useradd -r -g devuser -u 1000 devuser \
    && mkdir -p /home/devuser \
    && chown -R devuser:devuser /home/devuser

# Set work directory and permissions
WORKDIR /app
RUN chown -R devuser:devuser /app

# Switch to development user
USER devuser

# Set development environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    # Development settings
    APP_ENV=development \
    DEBUG=1 \
    LOG_LEVEL=DEBUG \
    # Hot reload settings
    RELOAD=1

# Expose development ports
EXPOSE 8000 8001 8002 5678

# Development health check (more frequent)
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command with auto-reload
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# Testing stage - For running tests
FROM development as testing

# Copy test files
COPY --chown=devuser:devuser tests/ tests/
COPY --chown=devuser:devuser pytest.ini ./

# Set testing environment
ENV APP_ENV=testing \
    TESTING=1

# Run tests during build (optional)
RUN python -m pytest tests/ -v --tb=short

# Default test command
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]