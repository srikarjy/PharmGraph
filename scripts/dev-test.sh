#!/bin/bash
set -e

# Development testing script
echo "🧪 Running development tests for Pharmacogenomics ML Platform..."

# Check if services are running
if ! docker-compose ps | grep -q "Up"; then
    echo "❌ Services are not running. Please start them first with:"
    echo "   ./scripts/dev-setup.sh"
    exit 1
fi

# Test API health
echo "🔍 Testing API health..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
    exit 1
fi

# Test ML server health
echo "🔍 Testing ML server health..."
if curl -f -s http://localhost:8001/health > /dev/null; then
    echo "✅ ML server is healthy"
else
    echo "❌ ML server is not responding"
    exit 1
fi

# Test Analytics service health
echo "🔍 Testing Analytics service health..."
if curl -f -s http://localhost:8002/health > /dev/null; then
    echo "✅ Analytics service is healthy"
else
    echo "❌ Analytics service is not responding"
    exit 1
fi

# Test database connectivity
echo "🔍 Testing database connectivity..."
if docker-compose exec -T postgres pg_isready -U pgx_user -d pharmacogenomics > /dev/null; then
    echo "✅ Database is accessible"
else
    echo "❌ Database is not accessible"
    exit 1
fi

# Test Redis connectivity
echo "🔍 Testing Redis connectivity..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is accessible"
else
    echo "❌ Redis is not accessible"
    exit 1
fi

# Run unit tests in containers
echo "🧪 Running unit tests..."
docker-compose exec -T pharmacogenomics-api python -m pytest tests/ -v --tb=short

# Test API endpoints
echo "🔍 Testing API endpoints..."
echo "  Testing root endpoint..."
curl -s http://localhost:8000/ | grep -q "Pharmacogenomics" && echo "✅ Root endpoint working"

echo "  Testing health endpoint..."
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✅ Health endpoint working"

echo "  Testing docs endpoint..."
curl -s http://localhost:8000/docs | grep -q "swagger" && echo "✅ Documentation endpoint working"

# Test inter-service communication
echo "🔍 Testing inter-service communication..."
docker-compose exec -T pharmacogenomics-api ping -c 1 postgres > /dev/null && echo "✅ API can reach database"
docker-compose exec -T pharmacogenomics-api ping -c 1 redis > /dev/null && echo "✅ API can reach Redis"
docker-compose exec -T pharmacogenomics-worker ping -c 1 postgres > /dev/null && echo "✅ Worker can reach database"

# Check resource usage
echo "📊 Resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo ""
echo "✅ All development tests passed!"
echo "🎉 Your development environment is ready!"