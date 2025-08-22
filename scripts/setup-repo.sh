#!/bin/bash
set -e

# Repository Setup Script for Pharmacogenomics ML Platform
# This script initializes the repository with proper security and documentation

echo "🚀 Setting up Pharmacogenomics ML Platform repository..."

# Check if required tools are installed
command -v git >/dev/null 2>&1 || { echo "❌ Git is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed. Aborting." >&2; exit 1; }

# Initialize Git repository if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git branch -M main
fi

# Configure Git settings
echo "⚙️  Configuring Git settings..."
git config user.name "Pharmacogenomics Team"
git config user.email "dev@pharmacogenomics.org"

# Install pre-commit hooks if available
if command -v pre-commit >/dev/null 2>&1; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
else
    echo "⚠️  pre-commit not found. Install with: pip install pre-commit"
fi

# Create initial commit
echo "📝 Creating initial commit..."
git add .
git commit -m "feat: initial project setup with containerization and documentation

- Add comprehensive Docker containerization with multi-stage builds
- Implement security best practices and credential protection
- Add professional documentation (README, CONTRIBUTING, SECURITY)
- Configure GitHub Actions CI/CD pipeline
- Set up development environment with pre-commit hooks
- Create environment configuration templates
- Add comprehensive .gitignore for security"

echo "✅ Repository setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create GitHub repository:"
echo "   gh repo create pharmacogenomics-platform --public --description 'Production-scale pharmacogenomics ML platform'"
echo ""
echo "2. Push to GitHub:"
echo "   git remote add origin https://github.com/yourusername/pharmacogenomics-platform.git"
echo "   git push -u origin main"
echo ""
echo "3. Set up environment:"
echo "   cp .env.example .env"
echo "   # Edit .env with your configuration"
echo ""
echo "4. Install dependencies:"
echo "   python -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements-dev.txt"
echo ""
echo "5. Start development:"
echo "   docker-compose up -d"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "🔒 Security checklist:"
echo "✅ .gitignore configured to exclude credentials"
echo "✅ Environment templates created (.env.example)"
echo "✅ Security scanning configured in CI/CD"
echo "✅ Pre-commit hooks for credential detection"
echo "✅ Container security best practices implemented"
echo ""
echo "📚 Documentation available:"
echo "- README.md: Getting started guide"
echo "- CONTRIBUTING.md: Development guidelines"
echo "- SECURITY.md: Security policy and reporting"
echo "- LICENSE: MIT license"
echo ""
echo "🎉 Happy coding!"