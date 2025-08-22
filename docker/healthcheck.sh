#!/bin/bash
set -e

# Health check script for Pharmacogenomics ML Platform
# Performs comprehensive health checks for different services

# Configuration
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"
MAX_RETRIES="${HEALTH_CHECK_RETRIES:-3}"

# Function to check HTTP endpoint
check_http_endpoint() {
    local url="$1"
    local timeout="$2"
    
    if curl -f -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    if [ -n "$DATABASE_URL" ]; then
        python -c "
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

try:
    engine = create_engine('$DATABASE_URL')
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('Database: OK')
except Exception as e:
    print(f'Database: FAILED - {e}')
    sys.exit(1)
" 2>/dev/null
    fi
}

# Function to check Redis connectivity
check_redis() {
    if [ -n "$REDIS_URL" ]; then
        python -c "
import sys
import redis
from urllib.parse import urlparse

try:
    parsed = urlparse('$REDIS_URL')
    r = redis.Redis(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 6379,
        db=int(parsed.path.lstrip('/')) if parsed.path else 0,
        socket_timeout=5
    )
    r.ping()
    print('Redis: OK')
except Exception as e:
    print(f'Redis: FAILED - {e}')
    sys.exit(1)
" 2>/dev/null
    fi
}

# Function to check disk space
check_disk_space() {
    local threshold=90
    local usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$usage" -gt "$threshold" ]; then
        echo "Disk space: CRITICAL - ${usage}% used"
        return 1
    else
        echo "Disk space: OK - ${usage}% used"
        return 0
    fi
}

# Function to check memory usage
check_memory() {
    local threshold=90
    local usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    if [ "$usage" -gt "$threshold" ]; then
        echo "Memory: WARNING - ${usage}% used"
        # Don't fail on memory warning, just log it
        return 0
    else
        echo "Memory: OK - ${usage}% used"
        return 0
    fi
}

# Function to check if required processes are running
check_processes() {
    local required_processes="python"
    
    for process in $required_processes; do
        if ! pgrep -f "$process" > /dev/null; then
            echo "Process check: FAILED - $process not running"
            return 1
        fi
    done
    
    echo "Processes: OK"
    return 0
}

# Main health check function
main_health_check() {
    local retries=0
    local failed_checks=""
    
    echo "Starting health check..."
    
    # Check HTTP endpoint
    while [ $retries -lt $MAX_RETRIES ]; do
        if check_http_endpoint "$HEALTH_CHECK_URL" "$TIMEOUT"; then
            echo "HTTP endpoint: OK"
            break
        else
            retries=$((retries + 1))
            if [ $retries -eq $MAX_RETRIES ]; then
                echo "HTTP endpoint: FAILED after $MAX_RETRIES attempts"
                failed_checks="$failed_checks HTTP"
            else
                echo "HTTP endpoint: Retry $retries/$MAX_RETRIES"
                sleep 2
            fi
        fi
    done
    
    # Check system resources
    if ! check_disk_space; then
        failed_checks="$failed_checks DISK"
    fi
    
    check_memory  # Memory check doesn't fail the health check
    
    if ! check_processes; then
        failed_checks="$failed_checks PROCESSES"
    fi
    
    # Check external dependencies (optional)
    if [ "$CHECK_DEPENDENCIES" = "true" ]; then
        check_database || failed_checks="$failed_checks DATABASE"
        check_redis || failed_checks="$failed_checks REDIS"
    fi
    
    # Determine overall health status
    if [ -n "$failed_checks" ]; then
        echo "Health check FAILED. Failed components:$failed_checks"
        return 1
    else
        echo "Health check PASSED. All systems operational."
        return 0
    fi
}

# Service-specific health checks
case "${SERVICE_TYPE:-api}" in
    "api")
        HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
        ;;
    "ml-server")
        HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8001/health}"
        ;;
    "worker")
        # For workers, check if Celery is running
        if ! pgrep -f "celery worker" > /dev/null; then
            echo "Worker health check FAILED - Celery worker not running"
            exit 1
        fi
        echo "Worker health check PASSED"
        exit 0
        ;;
    "analytics")
        HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8002/health}"
        ;;
esac

# Run the main health check
main_health_check