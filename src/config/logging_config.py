"""Structured logging system with file rotation and context tracking."""

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime
import uuid

from .models import LoggingConfig
from .config_manager import get_config


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self):
        super().__init__()
        self._context = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record."""
        # Add default context if not set
        if not hasattr(self._context, 'request_id'):
            self._context.request_id = str(uuid.uuid4())[:8]
        if not hasattr(self._context, 'session_id'):
            self._context.session_id = str(uuid.uuid4())[:8]
        
        # Add standard context to record
        record.request_id = getattr(self._context, 'request_id', 'unknown')
        record.session_id = getattr(self._context, 'session_id', 'unknown')
        record.thread_name = threading.current_thread().name
        
        # Add all custom context fields to record
        if hasattr(self._context, '_custom_fields'):
            for key, value in self._context._custom_fields.items():
                setattr(record, key, value)
        
        return True
    
    def set_context(self, **kwargs):
        """Set context variables for the current thread."""
        # Initialize custom fields dict if not exists
        if not hasattr(self._context, '_custom_fields'):
            self._context._custom_fields = {}
        
        for key, value in kwargs.items():
            if key in ['request_id', 'session_id']:
                # Set standard context fields directly
                setattr(self._context, key, value)
            else:
                # Store custom fields in a separate dict
                self._context._custom_fields[key] = value
    
    def clear_context(self):
        """Clear context for the current thread."""
        if hasattr(self._context, 'request_id'):
            delattr(self._context, 'request_id')
        if hasattr(self._context, 'session_id'):
            delattr(self._context, 'session_id')
        if hasattr(self._context, '_custom_fields'):
            delattr(self._context, '_custom_fields')


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': getattr(record, 'thread_name', 'unknown'),
            'request_id': getattr(record, 'request_id', 'unknown'),
            'session_id': getattr(record, 'session_id', 'unknown')
        }
        
        # Add exception information if present
        if record.exc_info and record.exc_info != True:
            log_data['exception'] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_data['exception'] = record.exc_text
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'request_id', 'session_id',
                          'thread_name']:
                log_data['extra'] = log_data.get('extra', {})
                log_data['extra'][key] = value
        
        return json.dumps(log_data, default=str)


class LoggerFactory:
    """Factory for creating and configuring loggers."""
    
    def __init__(self):
        self._configured = False
        self._context_filter = ContextFilter()
        self._loggers: Dict[str, logging.Logger] = {}
    
    def configure_logging(self, config: Optional[LoggingConfig] = None) -> None:
        """Configure the logging system.
        
        Args:
            config: Logging configuration. If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.logging
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        root_logger.setLevel(getattr(logging, config.level.upper()))
        
        # Setup console handler if enabled
        if config.console_output:
            self._setup_console_handler(config)
        
        # Setup file handler if file path is specified
        if config.file_path:
            self._setup_file_handler(config)
        
        # Add context filter to root logger
        root_logger.addFilter(self._context_filter)
        
        self._configured = True
    
    def _setup_console_handler(self, config: LoggingConfig) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        
        if config.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(config.format)
        
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    def _setup_file_handler(self, config: LoggingConfig) -> None:
        """Setup file logging handler with rotation."""
        # Ensure log directory exists
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        
        file_handler.setLevel(getattr(logging, config.level.upper()))
        
        if config.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(config.format)
        
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name.
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if not self._configured:
            self.configure_logging()
        
        if name not in self._loggers:
            logger = logging.getLogger(name)
            # Add context filter to individual logger if not already added
            if not any(isinstance(f, ContextFilter) for f in logger.filters):
                logger.addFilter(self._context_filter)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_context(self, **kwargs) -> None:
        """Set logging context for the current thread.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self._context_filter.set_context(**kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context for the current thread."""
        self._context_filter.clear_context()
    
    def add_context_filter(self, logger: logging.Logger) -> None:
        """Add context filter to a specific logger.
        
        Args:
            logger: Logger to add context filter to
        """
        logger.addFilter(self._context_filter)


# Global logger factory instance
_logger_factory = LoggerFactory()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return _logger_factory.get_logger(name)


def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure the global logging system.
    
    Args:
        config: Logging configuration. If None, loads from global config.
    """
    _logger_factory.configure_logging(config)


def set_logging_context(**kwargs) -> None:
    """Set logging context for the current thread.
    
    Args:
        **kwargs: Context key-value pairs (e.g., request_id='abc123')
    """
    _logger_factory.set_context(**kwargs)


def clear_logging_context() -> None:
    """Clear logging context for the current thread."""
    _logger_factory.clear_context()


class LoggingContextManager:
    """Context manager for setting logging context."""
    
    def __init__(self, **kwargs):
        self.context = kwargs
    
    def __enter__(self):
        set_logging_context(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_logging_context()


def with_logging_context(**kwargs):
    """Context manager for setting logging context.
    
    Usage:
        with with_logging_context(request_id='123', user_id='456'):
            logger.info("Processing request")
    """
    return LoggingContextManager(**kwargs)