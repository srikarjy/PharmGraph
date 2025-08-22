"""Tests for configuration management system."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from src.config import ConfigManager, ConfigError, AppConfig, NCBIConfig


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self, config_content: str):
        """Create a test configuration file."""
        with open(self.config_file, 'w') as f:
            f.write(config_content)
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_content = """
environment: development
debug: true
data_directory: data
temp_directory: tmp

ncbi:
  email: "test@example.com"
  api_key: "test_key"
  rate_limit_requests_per_second: 3.0

database:
  url: "sqlite:///test.db"
  pool_size: 5

quality_scoring:
  variant_quality_weights:
    allele_frequency: 0.5
    clinical_significance: 0.5
  association_quality_weights:
    evidence_level: 1.0
  quality_thresholds:
    minimum_variant_quality: 0.5

logging:
  level: "INFO"
  console_output: true
"""
        self.create_test_config(config_content)
        
        manager = ConfigManager(str(self.config_file))
        config = manager.load_config()
        
        assert isinstance(config, AppConfig)
        assert config.environment == "development"
        assert config.debug is True
        assert config.ncbi.email == "test@example.com"
        assert config.ncbi.api_key == "test_key"
        assert config.database.url == "sqlite:///test.db"
    
    def test_missing_config_file(self):
        """Test error handling for missing configuration file."""
        manager = ConfigManager("nonexistent.yaml")
        
        with pytest.raises(ConfigError, match="Configuration file not found"):
            manager.load_config()
    
    def test_invalid_yaml(self):
        """Test error handling for invalid YAML."""
        config_content = """
invalid: yaml: content:
  - missing
    proper: indentation
"""
        self.create_test_config(config_content)
        
        manager = ConfigManager(str(self.config_file))
        
        with pytest.raises(ConfigError, match="Invalid YAML"):
            manager.load_config()
    
    def test_environment_overrides(self):
        """Test environment variable overrides."""
        config_content = """
environment: development
ncbi:
  email: "original@example.com"
  rate_limit_requests_per_second: 3.0
database:
  url: "sqlite:///original.db"
logging:
  level: "INFO"
"""
        self.create_test_config(config_content)
        
        # Set environment variables
        env_vars = {
            'GENOMICS_NCBI__EMAIL': 'override@example.com',
            'GENOMICS_NCBI__RATE_LIMIT_REQUESTS_PER_SECOND': '5.0',
            'GENOMICS_DATABASE__URL': 'postgresql://test',
            'GENOMICS_LOGGING__LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager(str(self.config_file))
            config = manager.load_config()
            
            assert config.ncbi.email == "override@example.com"
            assert config.ncbi.rate_limit_requests_per_second == 5.0
            assert config.database.url == "postgresql://test"
            assert config.logging.level == "DEBUG"
    
    def test_boolean_environment_conversion(self):
        """Test boolean conversion from environment variables."""
        config_content = """
environment: development
debug: false
ncbi:
  email: "test@example.com"
"""
        self.create_test_config(config_content)
        
        with patch.dict(os.environ, {'GENOMICS_DEBUG': 'true'}):
            manager = ConfigManager(str(self.config_file))
            config = manager.load_config()
            
            assert config.debug is True
    
    def test_config_validation_error(self):
        """Test configuration validation errors."""
        config_content = """
environment: development
ncbi:
  email: "invalid-email"  # Missing @ symbol
  rate_limit_requests_per_second: -1  # Invalid negative value
"""
        self.create_test_config(config_content)
        
        manager = ConfigManager(str(self.config_file))
        
        with pytest.raises(ConfigError, match="Configuration validation failed"):
            manager.load_config()
    
    def test_get_config_lazy_loading(self):
        """Test that get_config() loads configuration lazily."""
        config_content = """
environment: development
ncbi:
  email: "test@example.com"
database:
  url: "sqlite:///test.db"
"""
        self.create_test_config(config_content)
        
        manager = ConfigManager(str(self.config_file))
        
        # First call should load the config
        config1 = manager.get_config()
        assert isinstance(config1, AppConfig)
        
        # Second call should return the same instance
        config2 = manager.get_config()
        assert config1 is config2
    
    def test_reload_config(self):
        """Test configuration reloading."""
        config_content = """
environment: development
debug: false
ncbi:
  email: "test@example.com"
"""
        self.create_test_config(config_content)
        
        manager = ConfigManager(str(self.config_file))
        config1 = manager.load_config()
        assert config1.debug is False
        
        # Update config file
        updated_content = config_content.replace("debug: false", "debug: true")
        self.create_test_config(updated_content)
        
        # Reload should pick up changes
        config2 = manager.reload_config()
        assert config2.debug is True
        assert config1 is not config2


class TestConfigModels:
    """Test cases for configuration model validation."""
    
    def test_ncbi_config_validation(self):
        """Test NCBI configuration validation."""
        # Valid configuration
        config = NCBIConfig(email="test@example.com")
        assert config.email == "test@example.com"
        
        # Invalid email
        with pytest.raises(ValueError, match="Valid email address is required"):
            NCBIConfig(email="invalid-email")
        
        # Invalid rate limit
        with pytest.raises(ValueError, match="Rate limit must be positive"):
            NCBIConfig(email="test@example.com", rate_limit_requests_per_second=-1)
    
    def test_quality_scoring_weights_validation(self):
        """Test quality scoring weights validation."""
        from src.config.models import QualityScoringConfig
        
        # Valid weights (sum to 1.0)
        config = QualityScoringConfig(
            variant_quality_weights={"a": 0.5, "b": 0.5},
            association_quality_weights={"c": 1.0}
        )
        assert config.variant_quality_weights == {"a": 0.5, "b": 0.5}
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            QualityScoringConfig(
                variant_quality_weights={"a": 0.3, "b": 0.3}  # Sum = 0.6
            )
    
    def test_quality_thresholds_validation(self):
        """Test quality threshold validation."""
        from src.config.models import QualityScoringConfig
        
        # Invalid threshold (outside 0-1 range)
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            QualityScoringConfig(
                quality_thresholds={"test_threshold": 1.5}
            )