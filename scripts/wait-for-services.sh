#!/bin/bash
set -e

# Service dependency management script
# Waits for services to be ready before starting dependent services

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to wait for TCP service
wait_for_tcp() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-60}
    
    echo -e "${YELLOW}⏳ Waiting for $service_name at $host:$port...${NC}"
    
    local count=0
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout waiting for $service_name${NC}"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    
    echo -e "${GREEN}✅ $service_name is ready${NC}"
    return 0
}

# Function to wait for HTTP service
wait_for_http() {
    local url=$1
    local service_name=$2
    local timeout=${3:-60}
    
    echo -e "${YELLOW}⏳ Waiting for $service_name at $url...${NC}"
    
    local count=0
    while ! curl -f -s "$url" > /dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout waiting for $service_name${NC}"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    
    echo -e "${GREEN}✅ $service_name is ready${NC}"
    return 0
}

# Function to wait for database to be ready
wait_for_database() {
    local host=${1:-postgres}
    local port=${2:-5432}
    local user=${3:-pgx_user}
    local database=${4:-pharmacogenomics}
    local timeout=${5:-60}
    
    echo -e "${YELLOW}⏳ Waiting for database $database at $host:$port...${NC}"
    
    local count=0
    while ! PGPASSWORD=pgx_password psql -h "$host" -p "$port" -U "$user" -d "$database" -c "SELECT 1;" > /dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout waiting for database${NC}"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    
    echo -e "${GREEN}✅ Database is ready${NC}"
    return 0
}

# Function to wait for Redis
wait_for_redis() {
    local host=${1:-redis}
    local port=${2:-6379}
    local timeout=${3:-60}
    
    echo -e "${YELLOW}⏳ Waiting for Redis at $host:$port...${NC}"
    
    local count=0
    while ! redis-cli -h "$host" -p "$port" ping > /dev/null 2>&1; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout waiting for Redis${NC}"
            return 1
        fi
        count=$((count + 1))
        sleep 1
    done
    
    echo -e "${GREEN}✅ Redis is ready${NC}"
    return 0
}

# Main function to orchestrate service startup
main() {
    echo "🚀 Starting service dependency management..."
    
    # Parse command line arguments
    local service_type=${1:-all}
    
    case $service_type in
        "database")
            wait_for_tcp postgres 5432 "PostgreSQL"
            wait_for_database
            ;;
        "cache")
            wait_for_tcp redis 6379 "Redis"
            wait_for_redis
            ;;
        "api")
            wait_for_tcp postgres 5432 "PostgreSQL"
            wait_for_tcp redis 6379 "Redis"
            wait_for_database
            wait_for_redis
            ;;
        "ml")
            wait_for_tcp redis 6379 "Redis"
            wait_for_redis
            wait_for_http "http://mlflow:5000/health" "MLflow" 120
            ;;
        "worker")
            wait_for_tcp postgres 5432 "PostgreSQL"
            wait_for_tcp redis 6379 "Redis"
            wait_for_database
            wait_for_redis
            ;;
        "analytics")
            wait_for_tcp postgres 5432 "PostgreSQL"
            wait_for_tcp redis 6379 "Redis"
            wait_for_database
            wait_for_redis
            ;;
        "monitoring")
            wait_for_http "http://prometheus:9090/-/healthy" "Prometheus"
            ;;
        "all"|*)
            echo "🔄 Waiting for all core services..."
            wait_for_tcp postgres 5432 "PostgreSQL"
            wait_for_tcp redis 6379 "Redis"
            wait_for_database
            wait_for_redis
            
            echo "🔄 Waiting for application services..."
            wait_for_http "http://pharmacogenomics-api:8000/health" "API Service" 120
            wait_for_http "http://pharmacogenomics-ml:8001/health" "ML Server" 120
            wait_for_http "http://pharmacogenomics-analytics:8002/health" "Analytics Service" 120
            
            echo "🔄 Waiting for monitoring services..."
            wait_for_http "http://prometheus:9090/-/healthy" "Prometheus" 60
            wait_for_http "http://grafana:3000/api/health" "Grafana" 60
            ;;
    esac
    
    echo -e "${GREEN}✅ All required services are ready!${NC}"
}

# Run main function with all arguments
main "$@"