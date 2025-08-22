# Specialized Dockerfile for analytics service
# Optimized for report generation and data visualization

FROM python:3.11-slim as base

# Install system dependencies for analytics and visualization
RUN apt-get update && apt-get install -y \
    # Scientific computing libraries
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    # Graphics and visualization
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # PDF generation
    wkhtmltopdf \
    # Font support
    fonts-dejavu-core \
    fontconfig \
    # Essential tools
    curl \
    procps \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for analytics service
RUN groupadd -r analytics && useradd -r -g analytics -u 1004 analytics

# Set work directory
WORKDIR /app

# Copy and install analytics dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Additional analytics and visualization libraries
    pip install --no-cache-dir \
        matplotlib==3.8.2 \
        seaborn==0.13.0 \
        plotly==5.17.0 \
        bokeh==3.3.2 \
        altair==5.2.0 \
        jinja2==3.1.2 \
        weasyprint==60.2 \
        reportlab==4.0.7 \
        openpyxl==3.1.2 \
        xlsxwriter==3.1.9

# Copy analytics source code
COPY src/analytics/ src/analytics/
COPY src/config/ src/config/
COPY src/storage/ src/storage/
COPY src/__init__.py src/

# Copy configuration and scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/healthcheck.sh

# Create directories for reports and visualizations
RUN mkdir -p /app/reports /app/visualizations /app/logs /app/cache /app/temp && \
    chown -R analytics:analytics /app

# Switch to non-root user
USER analytics

# Analytics-specific environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Matplotlib backend for headless operation
    MPLBACKEND=Agg \
    # Analytics configuration
    ANALYTICS_CACHE_SIZE=100 \
    REPORT_TIMEOUT=300 \
    MAX_REPORT_SIZE=50 \
    # Visualization settings
    PLOT_DPI=300 \
    PLOT_FORMAT=png \
    # Service identification
    SERVICE_TYPE=analytics

# Expose analytics service port
EXPOSE 8002

# Health check for analytics service
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Use entrypoint for proper signal handling
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Start analytics server
CMD ["python", "-m", "uvicorn", "src.analytics.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "2"]