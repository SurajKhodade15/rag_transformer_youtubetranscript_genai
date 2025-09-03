"""
Simple Configuration Management
Uses environment variables with sensible defaults.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("📋 Loading configuration from environment variables")


def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    
    # Validate required API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    
    if not openai_key:
        error_msg = "OPENAI_API_KEY environment variable is required"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    if not groq_key:
        error_msg = "GROQ_API_KEY environment variable is required"
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    config = {
        # API Keys
        'openai_api_key': openai_key,
        'groq_api_key': groq_key,
        
        # Model Configuration
        'groq_model': os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
        'groq_temperature': float(os.getenv('GROQ_TEMPERATURE', '0.2')),
        'groq_max_tokens': int(os.getenv('GROQ_MAX_TOKENS', '1000')),
        'openai_embedding_model': os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
        
        # Document Processing
        'chunk_size': int(os.getenv('CHUNK_SIZE', '1000')),
        'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200')),
        
        # Vector Store
        'similarity_k': int(os.getenv('SIMILARITY_K', '4')),
        'vector_store_path': os.getenv('VECTOR_STORE_PATH', 'data/vector_store'),
        
        # Data Sources
        'youtube_video_id': os.getenv('YOUTUBE_VIDEO_ID', 'bCz4OMemCcA'),
        'pdf_path': os.getenv('PDF_PATH', 'Attention.pdf'),
        
        # LangChain (Optional)
        'langchain_api_key': os.getenv('LANGCHAIN_API_KEY'),
        'langchain_project': os.getenv('LANGCHAIN_PROJECT'),
        'langchain_tracing_v2': os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true',
    }
    
    logger.info("✅ Configuration loaded successfully")
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    required_keys = ['openai_api_key', 'groq_api_key']
    
    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Required configuration key '{key}' is missing")
    
    return True


def get_environment() -> str:
    """Get current environment"""
    return os.getenv('ENVIRONMENT', 'development')


def is_development() -> bool:
    """Check if running in development mode"""
    return get_environment().lower() == 'development'
