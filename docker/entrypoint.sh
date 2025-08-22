#!/bin/bash
set -e

# Entrypoint script for Pharmacogenomics ML Platform
# Handles proper signal forwarding and graceful shutdown

# Function to handle shutdown signals
shutdown() {
    echo "Received shutdown signal, gracefully stopping..."
    
    # If there's a PID file, kill the process gracefully
    if [ -f /tmp/app.pid ]; then
        PID=$(cat /tmp/app.pid)
        echo "Stopping process with PID: $PID"
        kill -TERM "$PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        for i in {1..30}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "Process stopped gracefully"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing process"
            kill -KILL "$PID" 2>/dev/null || true
        fi
        
        rm -f /tmp/app.pid
    fi
    
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT SIGQUIT

# Wait for dependencies if specified
if [ -n "$WAIT_FOR_SERVICES" ]; then
    echo "Waiting for services: $WAIT_FOR_SERVICES"
    for service in $WAIT_FOR_SERVICES; do
        echo "Waiting for $service..."
        /usr/local/bin/wait-for-it.sh "$service" --timeout=60 --strict
    done
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m src.storage.migrations upgrade
fi

# Initialize application if needed
if [ "$INIT_APP" = "true" ]; then
    echo "Initializing application..."
    python -m src.config.init_app
fi

# Create necessary directories
mkdir -p /app/logs /app/data /app/models /app/cache /app/temp

# Set proper permissions
chown -R appuser:appuser /app/logs /app/data /app/models /app/cache /app/temp 2>/dev/null || true

# Log startup information
echo "Starting Pharmacogenomics ML Platform"
echo "Environment: ${APP_ENV:-production}"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Command: $*"

# Execute the main command
if [ $# -eq 0 ]; then
    # No arguments provided, use default command
    exec python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
else
    # Execute provided command
    exec "$@"
fi