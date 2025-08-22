"""Configuration manager with YAML loading and environment overrides."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .models import AppConfig, NCBIConfig, DatabaseConfig, QualityScoringConfig, LoggingConfig


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class ConfigManager:
    """Manages application configuration with YAML files and environment overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. Defaults to config/config.yaml
        """
        self.config_path = config_path or "config/config.yaml"
        self.logger = logging.getLogger(__name__)
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from YAML file with environment overrides.
        
        Returns:
            AppConfig: Loaded and validated configuration
            
        Raises:
            ConfigError: If configuration loading or validation fails
        """
        try:
            # Load base configuration from YAML
            config_data = self._load_yaml_config()
            
            # Apply environment overrides
            config_data = self._apply_environment_overrides(config_data)
            
            # Create and validate configuration objects
            self._config = self._create_config_objects(config_data)
            
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            return self._config
            
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {str(e)}") from e
    
    def get_config(self) -> AppConfig:
        """Get current configuration, loading if necessary.
        
        Returns:
            AppConfig: Current configuration
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file.
        
        Returns:
            AppConfig: Reloaded configuration
        """
        self._config = None
        return self.load_config()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dict[str, Any]: Raw configuration data
            
        Raises:
            ConfigError: If file cannot be loaded or parsed
        """
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise ConfigError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not isinstance(config_data, dict):
                raise ConfigError("Configuration file must contain a YAML dictionary")
            
            return config_data
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in configuration file: {str(e)}")
        except IOError as e:
            raise ConfigError(f"Cannot read configuration file: {str(e)}")
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.
        
        Environment variables should be prefixed with GENOMICS_ and use double
        underscores to separate nested keys (e.g., GENOMICS_NCBI__API_KEY).
        
        Args:
            config_data: Base configuration data
            
        Returns:
            Dict[str, Any]: Configuration with environment overrides applied
        """
        env_prefix = "GENOMICS_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue
            
            # Remove prefix and convert to nested key path
            config_key = env_key[len(env_prefix):].lower()
            key_parts = config_key.split('__')
            
            # Navigate to the nested dictionary location
            current_dict = config_data
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            # Set the value, attempting type conversion
            final_key = key_parts[-1]
            current_dict[final_key] = self._convert_env_value(env_value)
            
            self.logger.debug(f"Applied environment override: {config_key} = {env_value}")
        
        return config_data
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type.
        
        Args:
            value: Environment variable value
            
        Returns:
            Any: Converted value
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _create_config_objects(self, config_data: Dict[str, Any]) -> AppConfig:
        """Create validated configuration objects from raw data.
        
        Args:
            config_data: Raw configuration data
            
        Returns:
            AppConfig: Validated configuration object
            
        Raises:
            ConfigError: If configuration validation fails
        """
        try:
            # Create NCBI configuration
            ncbi_data = config_data.get('ncbi', {})
            ncbi_config = NCBIConfig(**ncbi_data)
            
            # Create database configuration with defaults
            db_data = config_data.get('database', {})
            if 'url' not in db_data:
                db_data['url'] = 'sqlite:///genomics_pipeline.db'
            db_config = DatabaseConfig(**db_data)
            
            # Create quality scoring configuration
            quality_data = config_data.get('quality_scoring', {})
            quality_config = QualityScoringConfig(**quality_data)
            
            # Create logging configuration
            logging_data = config_data.get('logging', {})
            logging_config = LoggingConfig(**logging_data)
            
            # Create main app configuration
            app_data = {
                'ncbi': ncbi_config,
                'database': db_config,
                'quality_scoring': quality_config,
                'logging': logging_config,
                'environment': config_data.get('environment', 'development'),
                'debug': config_data.get('debug', False),
                'data_directory': config_data.get('data_directory', 'data'),
                'temp_directory': config_data.get('temp_directory', 'tmp')
            }
            
            return AppConfig(**app_data)
            
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Configuration validation failed: {str(e)}")


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance.
    
    Returns:
        AppConfig: Global configuration
    """
    return _config_manager.get_config()


def reload_config() -> AppConfig:
    """Reload the global configuration.
    
    Returns:
        AppConfig: Reloaded configuration
    """
    return _config_manager.reload_config()