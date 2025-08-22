# Specialized Dockerfile for data processing workers
# Optimized for background data processing and NCBI API integration

FROM python:3.11-slim as base

# Install system dependencies for data processing
RUN apt-get update && apt-get install -y \
    # Scientific computing libraries
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    # Database connectivity
    libpq-dev \
    # Network tools
    curl \
    wget \
    netcat-openbsd \
    # Process monitoring
    procps \
    htop \
    # For XML/HTML parsing
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for worker service
RUN groupadd -r worker && useradd -r -g worker -u 1003 worker

# Set work directory
WORKDIR /app

# Copy and install worker dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Additional data processing libraries
    pip install --no-cache-dir \
        celery[redis]==5.3.4 \
        pandas==2.1.4 \
        numpy==1.25.2 \
        biopython==1.81 \
        lxml==4.9.3 \
        beautifulsoup4==4.12.2

# Copy data processing source code
COPY src/data_ingestion/ src/data_ingestion/
COPY src/processing/ src/processing/
COPY src/quality/ src/quality/
COPY src/config/ src/config/
COPY src/storage/ src/storage/
COPY src/__init__.py src/

# Copy configuration and scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# Create necessary directories for data processing
RUN mkdir -p /app/data /app/logs /app/temp /app/cache && \
    chown -R worker:worker /app

# Switch to non-root user
USER worker

# Worker-specific environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Celery configuration
    CELERY_BROKER_URL=redis://redis:6379/1 \
    CELERY_RESULT_BACKEND=redis://redis:6379/1 \
    CELERY_TASK_SERIALIZER=json \
    CELERY_RESULT_SERIALIZER=json \
    CELERY_ACCEPT_CONTENT=json \
    # Worker configuration
    WORKER_CONCURRENCY=4 \
    WORKER_PREFETCH_MULTIPLIER=1 \
    WORKER_MAX_TASKS_PER_CHILD=1000 \
    # NCBI API configuration
    NCBI_RATE_LIMIT=10 \
    NCBI_TIMEOUT=30 \
    NCBI_RETRIES=3 \
    # Data processing settings
    BATCH_SIZE=100 \
    MAX_QUEUE_SIZE=1000 \
    # Service identification
    SERVICE_TYPE=worker

# Health check for worker service
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Use entrypoint for proper signal handling
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start Celery worker
CMD ["python", "-m", "celery", "worker", \
     "-A", "src.processing.celery_app", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--prefetch-multiplier=1", \
     "--max-tasks-per-child=1000"]