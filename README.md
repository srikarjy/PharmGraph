# Pharmacogenomics ML Platform

A production-scale platform for processing pharmacogenomics data with automated quality scoring, machine learning pipelines, and clinical decision support APIs.

[![Build Status](https://github.com/yourusername/pharmacogenomics-platform/workflows/CI/badge.svg)](https://github.com/yourusername/pharmacogenomics-platform/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/)

## Overview

The Pharmacogenomics ML Platform is designed for researchers and clinicians working with pharmacogenomics data. The platform provides automated data collection from NCBI databases, intelligent quality assessment, and machine learning-powered insights for precision medicine research.

### Key Features

- **Real-time Data Processing**: Automated collection and processing from NCBI/PubMed APIs
- **Quality Assessment**: Domain-specific algorithms for research literature evaluation
- **ML Pipeline**: AutoML capabilities with feature engineering and model serving
- **Clinical APIs**: RESTful APIs for clinical decision support
- **Analytics Engine**: Statistical analysis and visualization tools
- **Production Ready**: Containerized deployment with Kubernetes support
- **Monitoring**: Comprehensive observability with Prometheus and Grafana

### Target Audience

- Pharmacogenomics researchers
- Clinical decision support teams
- Precision medicine practitioners
- Bioinformatics professionals

## Technical Architecture

### Core Components

- **Data Ingestion Layer**: NCBI API integration with intelligent rate limiting and error handling
- **Quality Scoring Engine**: Automated assessment of pharmacogenomics literature quality
- **ML Pipeline**: Feature extraction, model training, and inference serving
- **API Gateway**: RESTful APIs with authentication and rate limiting
- **Analytics Engine**: Statistical analysis and report generation
- **Storage Layer**: PostgreSQL for structured data, Redis for caching

### Technology Stack

- **Backend**: Python 3.11, FastAPI, SQLAlchemy, Celery
- **Database**: PostgreSQL 15, Redis 7
- **ML/AI**: scikit-learn, PyTorch, Transformers, MLflow, Ray Serve
- **Containerization**: Docker, Kubernetes, Helm
- **Monitoring**: Prometheus, Grafana, ELK Stack
- **CI/CD**: GitHub Actions, Docker Registry
- **Message Queue**: Apache Kafka, Redis

### Integration Points

- NCBI E-utilities API for literature retrieval
- PubMed API for metadata extraction
- MLflow for experiment tracking
- Prometheus for metrics collection
- External clinical databases (configurable)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9 or higher
- Git
- 4GB+ RAM recommended
- NCBI API key (free registration required)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pharmacogenomics-platform.git
   cd pharmacogenomics-platform
   ```

2. **Copy environment configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Verify installation**:
   ```bash
   curl http://localhost:8000/health
   ```

5. **Access the API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following key variables:

- `NCBI_EMAIL`: Required for NCBI API access (your email address)
- `NCBI_API_KEY`: Optional but recommended for higher rate limits
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Application secret key for security

See `.env.example` for complete configuration options and descriptions.

### Database Setup

The platform uses PostgreSQL for primary data storage:

```bash
# Using Docker Compose (recommended)
docker-compose up -d postgres

# Manual setup
createdb pharmacogenomics
python -m src.storage.migrations upgrade
```

## Usage Examples

### Basic API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Search pharmacogenomics literature
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "warfarin CYP2C9",
        "max_results": 100,
        "quality_threshold": 0.7
    }
)
```

### ML Pipeline Usage

```python
from src.ml.automl_engine import AutoMLEngine

# Initialize AutoML engine
engine = AutoMLEngine()

# Train model on pharmacogenomics data
model = engine.train(
    data_path="data/pgx_dataset.csv",
    target_column="drug_response",
    feature_columns=["gene_variant", "drug_dose", "patient_age"]
)

# Make predictions
predictions = model.predict(new_data)
```

## Development

### Local Development Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Run development server**:
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/
```

### Code Quality

The project uses several tools for code quality:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## Deployment

### Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t pharmacogenomics-platform .

# Run container
docker run -p 8000:8000 --env-file .env pharmacogenomics-platform
```

### Kubernetes Deployment

1. **Configure Kubernetes secrets**:
   ```bash
   kubectl create secret generic pgx-secrets \
     --from-literal=database-password=your-password \
     --from-literal=ncbi-api-key=your-api-key
   ```

2. **Apply manifests**:
   ```bash
   kubectl apply -f k8s/
   ```

3. **Verify deployment**:
   ```bash
   kubectl get pods
   kubectl get services
   ```

### Production Considerations

- Use external PostgreSQL database (AWS RDS, Google Cloud SQL)
- Configure Redis cluster for high availability
- Set up SSL/TLS certificates
- Configure monitoring and alerting
- Implement backup and disaster recovery
- Use secrets management (Kubernetes secrets, AWS Secrets Manager)

## API Documentation

Once the platform is running, comprehensive API documentation is available:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Key API Endpoints

- `GET /health` - Health check
- `POST /api/v1/search` - Search pharmacogenomics literature
- `GET /api/v1/papers/{paper_id}` - Get paper details
- `POST /api/v1/ml/predict` - ML model predictions
- `GET /api/v1/analytics/reports` - Generate analytics reports

## Monitoring and Observability

The platform includes comprehensive monitoring:

- **Metrics**: Prometheus metrics at `/metrics`
- **Health Checks**: Kubernetes-ready health endpoints
- **Logging**: Structured logging with configurable levels
- **Tracing**: Distributed tracing support (optional)

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Troubleshooting

### Common Issues

1. **Database Connection Issues**:
   - Verify PostgreSQL is running
   - Check DATABASE_URL in .env file
   - Ensure database exists and migrations are applied

2. **NCBI API Rate Limiting**:
   - Verify NCBI_EMAIL is set
   - Consider getting an API key for higher limits
   - Check rate limiting configuration

3. **Memory Issues**:
   - Increase Docker memory limits
   - Adjust ML model cache size
   - Monitor memory usage with `docker stats`

### Getting Help

- Check the [Issues](https://github.com/yourusername/pharmacogenomics-platform/issues) page
- Review the [Documentation](docs/)
- Contact the development team

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@institution.edu]
- **GitHub**: [@yourusername]
- **Institution**: [Your Institution]

## Acknowledgments

- NCBI for providing access to genomics databases
- Open source community for foundational tools and libraries
- Research collaborators and contributors
- Clinical partners for domain expertise

## Citation

If you use this platform in your research, please cite:

```
[Your Name] et al. (2024). Pharmacogenomics ML Platform: A Production-Scale 
System for Precision Medicine Research. [Journal/Conference].
```