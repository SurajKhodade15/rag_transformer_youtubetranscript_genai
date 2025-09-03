"""
Logging configuration for the RAG application.
Provides structured logging with different levels and outputs.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from config.settings import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        if hasattr(record, 'execution_time'):
            log_obj['execution_time'] = record.execution_time
        
        return json.dumps(log_obj)


class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self, context: Optional[dict] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        """Add context information to record"""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


def setup_logging() -> logging.Logger:
    """Setup application logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    logger = logging.getLogger("rag_app")
    logger.setLevel(getattr(logging, settings.app.log_level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    if settings.app.log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.app.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, settings.app.log_level))
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        "logs/error.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    
    # Set formatters
    if settings.is_production():
        # Use JSON formatter in production
        json_formatter = JSONFormatter()
        console_handler.setFormatter(json_formatter)
        if settings.app.log_file:
            file_handler.setFormatter(json_formatter)
        error_handler.setFormatter(json_formatter)
    else:
        # Use standard formatter in development
        formatter = logging.Formatter(settings.app.log_format)
        console_handler.setFormatter(formatter)
        if settings.app.log_file:
            file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    if settings.app.log_file:
        logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name"""
    return logging.getLogger(f"rag_app.{name}")


class LoggerMixin:
    """Mixin class to add logging capability to other classes"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class"""
        return get_logger(self.__class__.__name__)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_execution_time(self, operation: str, execution_time: float, **kwargs):
        """Log execution time for an operation"""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'execution_time': execution_time,
                'operation': operation,
                **kwargs
            }
        )
    
    def log_api_call(self, api_name: str, endpoint: str, response_time: float, status_code: int = None):
        """Log API call metrics"""
        self.logger.info(
            f"API call: {api_name}",
            extra={
                'api_name': api_name,
                'endpoint': endpoint,
                'response_time': response_time,
                'status_code': status_code
            }
        )
    
    def log_cache_hit(self, cache_key: str, hit: bool):
        """Log cache hit/miss"""
        self.logger.info(
            f"Cache {'hit' if hit else 'miss'}: {cache_key}",
            extra={
                'cache_key': cache_key,
                'cache_hit': hit
            }
        )


# Global logger instances
app_logger = setup_logging()
performance_logger = PerformanceLogger()
