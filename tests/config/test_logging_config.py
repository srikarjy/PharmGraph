"""Tests for the structured logging system."""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src.config.logging_config import (
    LoggerFactory,
    ContextFilter,
    StructuredFormatter,
    get_logger,
    configure_logging,
    set_logging_context,
    clear_logging_context,
    with_logging_context
)
from src.config.models import LoggingConfig


class TestContextFilter:
    """Test cases for ContextFilter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = ContextFilter()
    
    def test_adds_default_context(self):
        """Test that default context is added to log records."""
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='test message', args=(), exc_info=None
        )
        
        result = self.filter.filter(record)
        
        assert result is True
        assert hasattr(record, 'request_id')
        assert hasattr(record, 'session_id')
        assert hasattr(record, 'thread_name')
        assert len(record.request_id) == 8  # UUID truncated to 8 chars
        assert len(record.session_id) == 8
    
    def test_set_custom_context(self):
        """Test setting custom context values."""
        self.filter.set_context(request_id='custom123', user_id='user456')
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='test message', args=(), exc_info=None
        )
        
        self.filter.filter(record)
        
        assert record.request_id == 'custom123'
        # session_id should still be auto-generated since not set
        assert len(record.session_id) == 8
    
    def test_clear_context(self):
        """Test clearing context."""
        self.filter.set_context(request_id='custom123')
        self.filter.clear_context()
        
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='test message', args=(), exc_info=None
        )
        
        self.filter.filter(record)
        
        # Should generate new request_id after clearing
        assert len(record.request_id) == 8
        assert record.request_id != 'custom123'
    
    def test_thread_isolation(self):
        """Test that context is isolated between threads."""
        results = {}
        
        def thread_func(thread_id):
            self.filter.set_context(request_id=f'thread_{thread_id}')
            time.sleep(0.1)  # Allow other thread to set context
            
            record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='', lineno=0,
                msg='test message', args=(), exc_info=None
            )
            self.filter.filter(record)
            results[thread_id] = record.request_id
        
        # Start two threads with different contexts
        thread1 = threading.Thread(target=thread_func, args=(1,))
        thread2 = threading.Thread(target=thread_func, args=(2,))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Each thread should maintain its own context
        assert results[1] == 'thread_1'
        assert results[2] == 'thread_2'


class TestStructuredFormatter:
    """Test cases for StructuredFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = StructuredFormatter()
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        record = logging.LogRecord(
            name='test.module', level=logging.INFO, pathname='/path/test.py',
            lineno=42, msg='Test message', args=(), exc_info=None
        )
        record.request_id = 'req123'
        record.session_id = 'sess456'
        record.thread_name = 'MainThread'
        
        result = self.formatter.format(record)
        log_data = json.loads(result)
        
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test.module'
        assert log_data['message'] == 'Test message'
        assert log_data['line'] == 42
        assert log_data['request_id'] == 'req123'
        assert log_data['session_id'] == 'sess456'
        assert log_data['thread'] == 'MainThread'
        assert 'timestamp' in log_data
    
    def test_exception_formatting(self):
        """Test formatting with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name='test', level=logging.ERROR, pathname='', lineno=0,
                msg='Error occurred', args=(), exc_info=sys.exc_info()
            )
            record.request_id = 'req123'
            record.session_id = 'sess456'
            record.thread_name = 'MainThread'
            
            result = self.formatter.format(record)
            log_data = json.loads(result)
            
            assert 'exception' in log_data
            assert 'ValueError: Test exception' in log_data['exception']
    
    def test_extra_fields(self):
        """Test handling of extra fields in log records."""
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='Test message', args=(), exc_info=None
        )
        record.request_id = 'req123'
        record.session_id = 'sess456'
        record.thread_name = 'MainThread'
        record.custom_field = 'custom_value'
        record.user_id = 'user789'
        
        result = self.formatter.format(record)
        log_data = json.loads(result)
        
        assert 'extra' in log_data
        assert log_data['extra']['custom_field'] == 'custom_value'
        assert log_data['extra']['user_id'] == 'user789'


class TestLoggerFactory:
    """Test cases for LoggerFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = LoggerFactory()
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset logging configuration
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for filter_obj in root_logger.filters[:]:
            root_logger.removeFilter(filter_obj)
        root_logger.setLevel(logging.WARNING)
        
        # Reset factory state
        self.factory._configured = False
        self.factory._loggers.clear()
    
    def test_configure_console_logging(self):
        """Test console logging configuration."""
        config = LoggingConfig(
            level="INFO",
            console_output=True,
            file_path=None,
            structured_logging=False
        )
        
        self.factory.configure_logging(config)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        
        # Should have console handler
        console_handlers = [h for h in root_logger.handlers 
                          if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) == 1
    
    def test_configure_file_logging(self):
        """Test file logging configuration."""
        config = LoggingConfig(
            level="DEBUG",
            console_output=False,
            file_path=str(self.log_file),
            max_file_size_mb=1,
            backup_count=3,
            structured_logging=False
        )
        
        self.factory.configure_logging(config)
        
        root_logger = logging.getLogger()
        
        # Should have rotating file handler
        file_handlers = [h for h in root_logger.handlers 
                        if isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(file_handlers) == 1
        
        file_handler = file_handlers[0]
        assert file_handler.maxBytes == 1024 * 1024  # 1MB
        assert file_handler.backupCount == 3
    
    def test_structured_logging_format(self):
        """Test structured logging format."""
        config = LoggingConfig(
            level="INFO",
            console_output=False,
            file_path=str(self.log_file),
            structured_logging=True
        )
        
        self.factory.configure_logging(config)
        logger = self.factory.get_logger('test.module')
        
        # Log a message
        logger.info("Test structured message")
        
        # Read the log file
        with open(self.log_file, 'r') as f:
            log_line = f.read().strip()
        
        # Should be valid JSON
        log_data = json.loads(log_line)
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test structured message'
        assert log_data['logger'] == 'test.module'
    
    def test_log_level_filtering(self):
        """Test that log level filtering works correctly."""
        config = LoggingConfig(
            level="WARNING",
            console_output=False,
            file_path=str(self.log_file),
            structured_logging=True
        )
        
        self.factory.configure_logging(config)
        logger = self.factory.get_logger('test.filtering')
        
        # Log messages at different levels
        logger.debug("Debug message")  # Should be filtered out
        logger.info("Info message")    # Should be filtered out
        logger.warning("Warning message")  # Should be logged
        logger.error("Error message")      # Should be logged
        
        # Read the log file
        with open(self.log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        # Should only have warning and error messages
        assert len(log_lines) == 2
        
        warning_data = json.loads(log_lines[0])
        error_data = json.loads(log_lines[1])
        
        assert warning_data['level'] == 'WARNING'
        assert warning_data['message'] == 'Warning message'
        assert error_data['level'] == 'ERROR'
        assert error_data['message'] == 'Error message'
    
    def test_file_rotation(self):
        """Test file rotation functionality."""
        # Create a small max file size to trigger rotation
        config = LoggingConfig(
            level="INFO",
            console_output=False,
            file_path=str(self.log_file),
            max_file_size_mb=0.001,  # Very small size (1KB)
            backup_count=2,
            structured_logging=False
        )
        
        self.factory.configure_logging(config)
        logger = self.factory.get_logger('test.rotation')
        
        # Log many messages to trigger rotation
        for i in range(100):
            logger.info(f"Log message {i} with some additional content to increase size")
        
        # Check that rotation occurred
        log_files = list(Path(self.temp_dir).glob("test.log*"))
        assert len(log_files) > 1  # Should have rotated files
    
    def test_context_integration(self):
        """Test context filter integration."""
        config = LoggingConfig(
            level="INFO",
            console_output=False,
            file_path=str(self.log_file),
            structured_logging=True
        )
        
        self.factory.configure_logging(config)
        self.factory.set_context(request_id='test123', user_id='user456')
        
        logger = self.factory.get_logger('test.context')
        logger.info("Message with context")
        
        # Read the log file
        with open(self.log_file, 'r') as f:
            log_line = f.read().strip()
        
        log_data = json.loads(log_line)
        assert log_data['request_id'] == 'test123'
        assert log_data['extra']['user_id'] == 'user456'
    
    def test_multiple_loggers_independence(self):
        """Test that multiple loggers work independently."""
        config = LoggingConfig(
            level="INFO",
            console_output=False,
            file_path=str(self.log_file),
            structured_logging=True
        )
        
        self.factory.configure_logging(config)
        
        logger1 = self.factory.get_logger('module1')
        logger2 = self.factory.get_logger('module2')
        
        logger1.info("Message from module1")
        logger2.warning("Message from module2")
        
        # Read the log file
        with open(self.log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        assert len(log_lines) == 2
        
        log1_data = json.loads(log_lines[0])
        log2_data = json.loads(log_lines[1])
        
        assert log1_data['logger'] == 'module1'
        assert log1_data['level'] == 'INFO'
        assert log2_data['logger'] == 'module2'
        assert log2_data['level'] == 'WARNING'


class TestGlobalFunctions:
    """Test cases for global logging functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
        
        # Clear any existing context before each test
        clear_logging_context()
        
        # Reset global logger factory state
        from src.config.logging_config import _logger_factory, ContextFilter
        _logger_factory._configured = False
        _logger_factory._loggers.clear()
        _logger_factory._context_filter = ContextFilter()
        # Clear any existing context before each test
        clear_logging_context()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset logging configuration
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for filter_obj in root_logger.filters[:]:
            root_logger.removeFilter(filter_obj)
        root_logger.setLevel(logging.WARNING)
        clear_logging_context()
        
        # Reset global logger factory
        from src.config.logging_config import _logger_factory
        _logger_factory._configured = False
        _logger_factory._loggers.clear()
        # Create a fresh context filter
        from src.config.logging_config import ContextFilter
        _logger_factory._context_filter = ContextFilter()
    
    def test_get_logger_function(self):
        """Test global get_logger function."""
        config = LoggingConfig(
            level="INFO",
            console_output=False,
            file_path=str(self.log_file),
            structured_logging=True
        )
        
        configure_logging(config)
        logger = get_logger('test.global')
        
        logger.info("Test global logger")
        
        # Verify log was written
        with open(self.log_file, 'r') as f:
            log_line = f.read().strip()
        
        log_data = json.loads(log_line)
        assert log_data['logger'] == 'test.global'
        assert log_data['message'] == 'Test global logger'
    
    def test_context_functions(self):
        """Test global context management functions."""
        # Create a completely isolated logger for this test
        import logging
        from src.config.logging_config import ContextFilter, StructuredFormatter
        
        # Create isolated logger
        test_logger = logging.getLogger('isolated.test.context')
        test_logger.handlers.clear()
        test_logger.filters.clear()
        test_logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(str(self.log_file))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(StructuredFormatter())
        
        # Create and add context filter
        context_filter = ContextFilter()
        test_logger.addFilter(context_filter)
        test_logger.addHandler(file_handler)
        
        # Set context
        context_filter.set_context(request_id='global123', operation='test')
        test_logger.info("Message with global context")
        
        # Clear context
        context_filter.clear_context()
        test_logger.info("Message without context")
        
        # Flush handler
        file_handler.flush()
        
        # Read log file
        with open(self.log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        assert len(log_lines) == 2
        
        # First message should have context
        log1_data = json.loads(log_lines[0])
        assert log1_data['request_id'] == 'global123'
        assert log1_data['extra']['operation'] == 'test'
        
        # Second message should have new auto-generated context
        log2_data = json.loads(log_lines[1])
        assert log2_data['request_id'] != 'global123'
        assert len(log2_data['request_id']) == 8
    
    def test_context_manager(self):
        """Test logging context manager."""
        clear_logging_context()  # Clear any existing context before test
        config = LoggingConfig(
            level="INFO",
            console_output=False,
            file_path=str(self.log_file),
            structured_logging=True
        )
        
        configure_logging(config)
        logger = get_logger('test.context_manager')
        
        # Use context manager
        with with_logging_context(request_id='ctx123', user_id='user789'):
            logger.info("Message inside context")
        
        # Outside context manager
        logger.info("Message outside context")
        
        # Read log file
        with open(self.log_file, 'r') as f:
            log_lines = f.read().strip().split('\n')
        
        assert len(log_lines) == 2
        
        # First message should have context
        log1_data = json.loads(log_lines[0])
        assert log1_data['request_id'] == 'ctx123'
        assert log1_data['extra']['user_id'] == 'user789'
        
        # Second message should have different context
        log2_data = json.loads(log_lines[1])
        assert log2_data['request_id'] != 'ctx123'