# Specialized Dockerfile for API service
# Optimized for high-performance API serving with FastAPI

FROM python:3.11-slim as base

# Install system dependencies for API service
RUN apt-get update && apt-get install -y \
    # Essential tools
    curl \
    procps \
    # For health checks
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r apiuser && useradd -r -g apiuser -u 1001 apiuser

# Set work directory
WORKDIR /app

# Copy and install only API dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy API-specific source code
COPY src/api/ src/api/
COPY src/config/ src/config/
COPY src/storage/ src/storage/
COPY src/__init__.py src/

# Copy configuration and scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/cache && \
    chown -R apiuser:apiuser /app

# Switch to non-root user
USER apiuser

# API-specific environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # API configuration
    API_WORKERS=4 \
    API_PORT=8000 \
    API_HOST=0.0.0.0 \
    API_TIMEOUT=30 \
    # Performance settings
    UVICORN_WORKER_CLASS=uvicorn.workers.UvicornWorker \
    UVICORN_MAX_REQUESTS=1000 \
    UVICORN_MAX_REQUESTS_JITTER=100 \
    # Service identification
    SERVICE_TYPE=api

# Expose API port
EXPOSE 8000

# Health check optimized for API service
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Use entrypoint for proper signal handling
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start API server with optimized settings
CMD ["python", "-m", "uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100"]