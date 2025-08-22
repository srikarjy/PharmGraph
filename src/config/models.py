"""Configuration data models with validation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class NCBIConfig:
    """NCBI API configuration settings."""
    email: str
    api_key: Optional[str] = None
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    rate_limit_requests_per_second: float = 3.0
    rate_limit_burst: int = 10
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    
    def __post_init__(self):
        """Validate NCBI configuration."""
        if not self.email or "@" not in self.email:
            raise ValueError("Valid email address is required for NCBI API")
        if self.rate_limit_requests_per_second <= 0:
            raise ValueError("Rate limit must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    def __post_init__(self):
        """Validate database configuration."""
        if not self.url:
            raise ValueError("Database URL is required")
        if self.pool_size <= 0:
            raise ValueError("Pool size must be positive")


@dataclass
class QualityScoringConfig:
    """Quality scoring weights and thresholds."""
    variant_quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "allele_frequency": 0.3,
        "clinical_significance": 0.4,
        "population_data": 0.2,
        "study_replication": 0.1
    })
    association_quality_weights: Dict[str, float] = field(default_factory=lambda: {
        "evidence_level": 0.4,
        "study_quality": 0.3,
        "replication_count": 0.2,
        "effect_size": 0.1
    })
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "minimum_variant_quality": 0.5,
        "minimum_association_quality": 0.6,
        "high_quality_threshold": 0.8
    })
    confidence_level: float = 0.95
    
    def __post_init__(self):
        """Validate quality scoring configuration."""
        # Validate weights sum to 1.0
        variant_sum = sum(self.variant_quality_weights.values())
        if abs(variant_sum - 1.0) > 0.01:
            raise ValueError(f"Variant quality weights must sum to 1.0, got {variant_sum}")
        
        association_sum = sum(self.association_quality_weights.values())
        if abs(association_sum - 1.0) > 0.01:
            raise ValueError(f"Association quality weights must sum to 1.0, got {association_sum}")
        
        # Validate thresholds are between 0 and 1
        for name, threshold in self.quality_thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"Quality threshold {name} must be between 0 and 1, got {threshold}")


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = True
    structured_logging: bool = True
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        if self.max_file_size_mb <= 0:
            raise ValueError("Max file size must be positive")


@dataclass
class AppConfig:
    """Main application configuration."""
    ncbi: NCBIConfig
    database: DatabaseConfig
    quality_scoring: QualityScoringConfig
    logging: LoggingConfig
    environment: str = "development"
    debug: bool = False
    data_directory: str = "data"
    temp_directory: str = "tmp"
    
    def __post_init__(self):
        """Validate application configuration."""
        valid_environments = ["development", "testing", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        
        # Create directories if they don't exist
        os.makedirs(self.data_directory, exist_ok=True)
        os.makedirs(self.temp_directory, exist_ok=True)