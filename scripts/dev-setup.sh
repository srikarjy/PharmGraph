#!/bin/bash
set -e

# Development environment setup script
echo "🚀 Setting up Pharmacogenomics ML Platform development environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs data models cache reports visualizations notebooks temp

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before starting services"
fi

# Pull required images
echo "📦 Pulling required Docker images..."
docker-compose pull postgres redis prometheus grafana elasticsearch kibana jupyter

# Build application images
echo "🔨 Building application images..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build

# Start core services first
echo "🚀 Starting core services (database, cache)..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
timeout 60 bash -c 'until docker-compose exec postgres pg_isready -U pgx_user -d pharmacogenomics; do sleep 2; done'

# Start all services
echo "🚀 Starting all services..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Show service status
echo "📊 Service status:"
docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps

# Show service URLs
echo ""
echo "🌐 Service URLs:"
echo "  API Documentation: http://localhost:8000/docs"
echo "  ML Server: http://localhost:8001/health"
echo "  Analytics: http://localhost:8002/health"
echo "  Grafana: http://localhost:3000 (admin/admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Jupyter: http://localhost:8888 (token: pharmacogenomics)"
echo "  PgAdmin: http://localhost:5050 (admin@pharmacogenomics.org/admin)"
echo "  Redis Commander: http://localhost:8081"
echo "  Kibana: http://localhost:5601"
echo ""

# Show logs command
echo "📋 Useful commands:"
echo "  View logs: docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f [service]"
echo "  Stop services: docker-compose -f docker-compose.yml -f docker-compose.dev.yml down"
echo "  Restart service: docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart [service]"
echo "  Shell access: docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec [service] bash"
echo ""

echo "✅ Development environment setup complete!"
echo "🎉 Happy coding!"