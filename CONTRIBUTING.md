# Contributing to Pharmacogenomics ML Platform

Thank you for your interest in contributing to the Pharmacogenomics ML Platform! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@yourinstitution.edu].

## How to Contribute

### Reporting Issues

Before creating an issue, please:

1. Check existing issues to avoid duplicates
2. Use the issue templates when available
3. Provide clear, detailed information
4. Include steps to reproduce bugs
5. Specify your environment (OS, Python version, etc.)

### Suggesting Features

Feature requests are welcome! Please:

1. Check if the feature already exists or is planned
2. Clearly describe the use case and benefits
3. Consider the scope and complexity
4. Be open to discussion and feedback

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the development setup** instructions in README.md
3. **Make your changes** with clear, focused commits
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Ensure all tests pass** and code quality checks succeed
7. **Submit a pull request** with a clear description

## Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run these tools before submitting:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Testing

- Write unit tests for new functions and classes
- Add integration tests for new features
- Ensure all tests pass: `pytest`
- Maintain or improve test coverage
- Use meaningful test names and docstrings

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Update API documentation
- Include examples for new features

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add new pharmacogenomics quality scoring algorithm

- Implement domain-specific scoring metrics
- Add tests for quality assessment
- Update documentation with usage examples

Closes #123
```

Commit message format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions or modifications
- `refactor:` code refactoring
- `style:` formatting changes
- `chore:` maintenance tasks

### Branch Naming

Use descriptive branch names:
- `feature/add-quality-scoring`
- `fix/database-connection-issue`
- `docs/update-api-documentation`

## Development Environment

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- PostgreSQL (for local development)

### Setup

1. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/pharmacogenomics-platform.git
   cd pharmacogenomics-platform
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

6. **Run tests**:
   ```bash
   pytest
   ```

### Running Locally

```bash
# Start database
docker-compose up -d postgres redis

# Run migrations
python -m src.storage.migrations upgrade

# Start development server
uvicorn src.api.main:app --reload
```

## Project Structure

```
pharmacogenomics-platform/
├── src/                    # Source code
│   ├── api/               # API layer
│   ├── ml/                # Machine learning components
│   ├── data_ingestion/    # Data collection and processing
│   ├── quality/           # Quality scoring algorithms
│   ├── analytics/         # Analytics and reporting
│   └── storage/           # Database models and operations
├── tests/                 # Test suite
├── docs/                  # Documentation
├── docker/                # Docker configuration
├── k8s/                   # Kubernetes manifests
└── scripts/               # Utility scripts
```

## Review Process

### Pull Request Review

All pull requests require:

1. **Code review** by at least one maintainer
2. **Passing CI/CD** checks
3. **Test coverage** maintained or improved
4. **Documentation** updated as needed

### Review Criteria

Reviewers will check for:

- Code quality and style
- Test coverage and quality
- Documentation completeness
- Security considerations
- Performance implications
- Backward compatibility

## Release Process

1. **Version Bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Comprehensive testing in staging
4. **Documentation**: Update release documentation
5. **Tagging**: Create release tag
6. **Deployment**: Deploy to production

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: [dev@yourinstitution.edu] for development questions

### Resources

- [README.md](README.md) - Getting started guide
- [API Documentation](http://localhost:8000/docs) - API reference
- [Architecture Documentation](docs/architecture.md) - System design
- [Deployment Guide](docs/deployment.md) - Production deployment

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Academic publications (where appropriate)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to the Pharmacogenomics ML Platform!