#!/usr/bin/env python3
"""Example demonstrating the structured logging system."""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_logger, configure_logging, set_logging_context, with_logging_context
from config.models import LoggingConfig


def main():
    """Demonstrate logging functionality."""
    print("=== Genomics Pipeline Logging Example ===\n")
    
    # Configure logging
    config = LoggingConfig(
        level="INFO",
        console_output=True,
        file_path="logs/example.log",
        structured_logging=True,
        max_file_size_mb=10,
        backup_count=3
    )
    
    configure_logging(config)
    logger = get_logger(__name__)
    
    print("✅ Logging configured successfully!")
    
    # Basic logging
    logger.info("Starting genomics pipeline example")
    logger.debug("This debug message won't show (level is INFO)")
    logger.warning("This is a warning message")
    
    # Context logging
    print("\n--- Context Logging Demo ---")
    set_logging_context(request_id="req_12345", user_id="scientist_001")
    logger.info("Processing genomics data with context")
    
    # Context manager
    with with_logging_context(operation="variant_analysis", batch_id="batch_789"):
        logger.info("Analyzing variants in batch")
        logger.error("Simulated error during analysis")
    
    # Different modules
    ncbi_logger = get_logger("ncbi.api")
    quality_logger = get_logger("quality.scoring")
    
    ncbi_logger.info("Fetching data from NCBI API")
    quality_logger.info("Calculating quality scores")
    
    # Exception logging
    try:
        raise ValueError("Simulated processing error")
    except ValueError as e:
        logger.error("Error occurred during processing", exc_info=True)
    
    print(f"\n✅ Logs written to: {config.file_path}")
    print("Check the log file to see structured JSON output!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())