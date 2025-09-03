"""
Configuration settings for the RAG application.
Industry-standard configuration management with environment variables.
"""

import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class APISettings(BaseSettings):
    """API configuration settings"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    
    # Groq Configuration
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")
    groq_temperature: float = Field(default=0.2, env="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=1000, env="GROQ_MAX_TOKENS")
    
    # LangChain Configuration (Optional)
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    langchain_project: Optional[str] = Field(default=None, env="LANGCHAIN_PROJECT")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    
    # HuggingFace Configuration (Optional)
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment


class RAGSettings(BaseSettings):
    """RAG pipeline configuration settings"""
    
    # Document Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="FAISS", env="VECTOR_STORE_TYPE")
    similarity_search_k: int = Field(default=4, env="SIMILARITY_SEARCH_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # YouTube Configuration
    youtube_video_id: str = Field(default="bCz4OMemCcA", env="YOUTUBE_VIDEO_ID")
    youtube_languages: List[str] = Field(default=["en"], env="YOUTUBE_LANGUAGES")
    
    # PDF Configuration
    pdf_path: str = Field(default="Attention.pdf", env="PDF_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment


class AppSettings(BaseSettings):
    """Application configuration settings"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Streamlit Configuration
    page_title: str = Field(default="Attention Mechanism RAG Assistant", env="PAGE_TITLE")
    page_icon: str = Field(default="🔍", env="PAGE_ICON")
    layout: str = Field(default="wide", env="LAYOUT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default="logs/app.log", env="LOG_FILE")
    
    # Caching Configuration
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    cache_type: str = Field(default="memory", env="CACHE_TYPE")  # memory, redis, etc.
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    @validator("environment")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be one of: development, staging, production")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment


class DatabaseSettings(BaseSettings):
    """Database and storage configuration"""
    
    # Vector Database
    vector_db_path: str = Field(default="data/vector_store", env="VECTOR_DB_PATH")
    
    # Cache Database (if using Redis)
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment


class Settings:
    """Main settings class that combines all configuration"""
    
    def __init__(self):
        self.api = APISettings()
        self.rag = RAGSettings()
        self.app = AppSettings()
        self.database = DatabaseSettings()
        
        # Set up LangChain environment if configured
        self._setup_langchain_environment()
        
        # Create necessary directories
        self._create_directories()
    
    def _setup_langchain_environment(self):
        """Set up LangChain environment variables"""
        if self.api.langchain_api_key and self.api.langchain_project:
            os.environ["LANGCHAIN_API_KEY"] = self.api.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.api.langchain_project
            os.environ["LANGCHAIN_TRACING_V2"] = str(self.api.langchain_tracing_v2).lower()
            os.environ["LANGCHAIN_ENDPOINT"] = self.api.langchain_endpoint
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            "logs",
            "data",
            "data/vector_store",
            "data/cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.app.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.app.environment == "development"


# Global settings instance
settings = Settings()
