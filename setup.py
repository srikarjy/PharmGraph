from setuptools import setup, find_packages

setup(
    name="pharmacogenomics-platform",
    version="1.0.0",
    description="High-performance ML platform for pharmacogenomics research and clinical decision support",
    author="Pharmacogenomics Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "sqlalchemy>=2.0.0",
        "pandas>=2.1.0",
        "numpy>=1.25.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "ml": [
            "torch>=2.1.0",
            "transformers>=4.36.0",
            "mlflow>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pgx-api=src.api.main:main",
            "pgx-ml-server=src.ml.model_server:main",
            "pgx-worker=src.processing.worker:main",
        ],
    },
)