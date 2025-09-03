"""
Simple Utility Functions
Basic helper functions for the RAG application.
"""

import logging
import time
from pathlib import Path
from functools import wraps
from typing import Any, Callable


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup basic logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger('rag_app')


def timer(func: Callable) -> Callable:
    """Simple timing decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def ensure_directory(path: str) -> Path:
    """Ensure directory exists"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def safe_execute(func: Callable, default_value: Any = None, error_message: str = "Error occurred") -> Any:
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        print(f"{error_message}: {str(e)}")
        return default_value


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common unwanted characters
    text = text.replace('\ufeff', '')  # BOM
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    
    return text.strip()
