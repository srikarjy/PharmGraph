#!/usr/bin/env python3
"""Example demonstrating the configuration management system."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, get_config


def main():
    """Demonstrate configuration loading and usage."""
    print("=== Genomics Pipeline Configuration Example ===\n")
    
    try:
        # Load configuration using the global instance
        config = get_config()
        
        print("✅ Configuration loaded successfully!")
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug}")
        print(f"Data directory: {config.data_directory}")
        
        print("\n--- NCBI Configuration ---")
        print(f"Email: {config.ncbi.email}")
        print(f"API Key: {'***' if config.ncbi.api_key else 'Not set'}")
        print(f"Rate limit: {config.ncbi.rate_limit_requests_per_second} req/sec")
        print(f"Base URL: {config.ncbi.base_url}")
        
        print("\n--- Database Configuration ---")
        print(f"URL: {config.database.url}")
        print(f"Pool size: {config.database.pool_size}")
        print(f"Echo SQL: {config.database.echo}")
        
        print("\n--- Quality Scoring Configuration ---")
        print("Variant quality weights:")
        for factor, weight in config.quality_scoring.variant_quality_weights.items():
            print(f"  {factor}: {weight}")
        
        print("\nQuality thresholds:")
        for threshold, value in config.quality_scoring.quality_thresholds.items():
            print(f"  {threshold}: {value}")
        
        print("\n--- Logging Configuration ---")
        print(f"Level: {config.logging.level}")
        print(f"File path: {config.logging.file_path}")
        print(f"Console output: {config.logging.console_output}")
        
        # Demonstrate environment override
        print("\n--- Environment Override Example ---")
        print("Set GENOMICS_DEBUG=false to override debug setting")
        print("Set GENOMICS_NCBI__EMAIL=your@email.com to override NCBI email")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())