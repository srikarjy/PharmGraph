#!/bin/bash
set -e

# Comprehensive health check script for all services
echo "🏥 Running comprehensive health checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check HTTP endpoint
check_http() {
    local service=$1
    local url=$2
    local timeout=${3:-10}
    
    if curl -f -s --max-time $timeout "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $service is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ $service is not responding${NC}"
        return 1
    fi
}

# Function to check TCP port
check_tcp() {
    local service=$1
    local host=$2
    local port=$3
    local timeout=${4:-5}
    
    if timeout $timeout bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "${GREEN}✅ $service is accessible${NC}"
        return 0
    else
        echo -e "${RED}❌ $service is not accessible${NC}"
        return 1
    fi
}

# Function to check Docker container health
check_container() {
    local container=$1
    local status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
    
    case $status in
        "healthy")
            echo -e "${GREEN}✅ $container is healthy${NC}"
            return 0
            ;;
        "unhealthy")
            echo -e "${RED}❌ $container is unhealthy${NC}"
            return 1
            ;;
        "starting")
            echo -e "${YELLOW}⏳ $container is starting${NC}"
            return 1
            ;;
        *)
            echo -e "${RED}❌ $container status unknown${NC}"
            return 1
            ;;
    esac
}

# Initialize counters
total_checks=0
passed_checks=0

# Check core infrastructure services
echo "🔍 Checking core infrastructure..."

echo "Checking PostgreSQL..."
total_checks=$((total_checks + 1))
if check_tcp "PostgreSQL" "localhost" "5432"; then
    passed_checks=$((passed_checks + 1))
fi

echo "Checking Redis..."
total_checks=$((total_checks + 1))
if check_tcp "Redis" "localhost" "6379"; then
    passed_checks=$((passed_checks + 1))
fi

# Check application services
echo ""
echo "🔍 Checking application services..."

echo "Checking API service..."
total_checks=$((total_checks + 1))
if check_http "API Service" "http://localhost:8000/health"; then
    passed_checks=$((passed_checks + 1))
fi

echo "Checking ML server..."
total_checks=$((total_checks + 1))
if check_http "ML Server" "http://localhost:8001/health"; then
    passed_checks=$((passed_checks + 1))
fi

echo "Checking Analytics service..."
total_checks=$((total_checks + 1))
if check_http "Analytics Service" "http://localhost:8002/health"; then
    passed_checks=$((passed_checks + 1))
fi

# Check monitoring services
echo ""
echo "🔍 Checking monitoring services..."

echo "Checking Prometheus..."
total_checks=$((total_checks + 1))
if check_http "Prometheus" "http://localhost:9090/-/healthy"; then
    passed_checks=$((passed_checks + 1))
fi

echo "Checking Grafana..."
total_checks=$((total_checks + 1))
if check_http "Grafana" "http://localhost:3000/api/health"; then
    passed_checks=$((passed_checks + 1))
fi

# Check MLflow
echo "Checking MLflow..."
total_checks=$((total_checks + 1))
if check_http "MLflow" "http://localhost:5000/health" 15; then
    passed_checks=$((passed_checks + 1))
fi

# Check container health status
echo ""
echo "🔍 Checking container health status..."

containers=("pgx-api" "pgx-ml-server" "pgx-worker" "pgx-analytics" "pgx-postgres" "pgx-redis")
for container in "${containers[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^$container$"; then
        echo "Checking $container..."
        total_checks=$((total_checks + 1))
        if check_container "$container"; then
            passed_checks=$((passed_checks + 1))
        fi
    fi
done

# Check inter-service connectivity
echo ""
echo "🔍 Checking inter-service connectivity..."

if docker ps --format '{{.Names}}' | grep -q "pgx-api"; then
    echo "Testing API to database connectivity..."
    total_checks=$((total_checks + 1))
    if docker exec pgx-api python -c "
import os
from sqlalchemy import create_engine
try:
    engine = create_engine(os.getenv('DATABASE_URL'))
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('✅ API can connect to database')
    exit(0)
except Exception as e:
    print(f'❌ API cannot connect to database: {e}')
    exit(1)
" 2>/dev/null; then
        passed_checks=$((passed_checks + 1))
    fi

    echo "Testing API to Redis connectivity..."
    total_checks=$((total_checks + 1))
    if docker exec pgx-api python -c "
import redis
import os
from urllib.parse import urlparse
try:
    parsed = urlparse(os.getenv('REDIS_URL', 'redis://redis:6379/0'))
    r = redis.Redis(host=parsed.hostname, port=parsed.port, db=0)
    r.ping()
    print('✅ API can connect to Redis')
    exit(0)
except Exception as e:
    print(f'❌ API cannot connect to Redis: {e}')
    exit(1)
" 2>/dev/null; then
        passed_checks=$((passed_checks + 1))
    fi
fi

# Summary
echo ""
echo "📊 Health Check Summary:"
echo "  Total checks: $total_checks"
echo "  Passed: $passed_checks"
echo "  Failed: $((total_checks - passed_checks))"

if [ $passed_checks -eq $total_checks ]; then
    echo -e "${GREEN}🎉 All health checks passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some health checks failed. Please check the logs.${NC}"
    exit 1
fi