"""
Custom exceptions for the RAG application.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class RAGApplicationError(Exception):
    """Base exception for all RAG application errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ConfigurationError(RAGApplicationError):
    """Raised when there's an issue with application configuration"""
    pass


class APIError(RAGApplicationError):
    """Base class for API-related errors"""
    pass


class OpenAIAPIError(APIError):
    """Raised when OpenAI API requests fail"""
    pass


class GroqAPIError(APIError):
    """Raised when Groq API requests fail"""
    pass


class YouTubeAPIError(APIError):
    """Raised when YouTube transcript API fails"""
    pass


class DocumentProcessingError(RAGApplicationError):
    """Raised when document processing fails"""
    pass


class PDFProcessingError(DocumentProcessingError):
    """Raised when PDF processing fails"""
    pass


class TranscriptProcessingError(DocumentProcessingError):
    """Raised when transcript processing fails"""
    pass


class VectorStoreError(RAGApplicationError):
    """Raised when vector store operations fail"""
    pass


class EmbeddingError(RAGApplicationError):
    """Raised when embedding generation fails"""
    pass


class RetrievalError(RAGApplicationError):
    """Raised when document retrieval fails"""
    pass


class ModelError(RAGApplicationError):
    """Raised when model operations fail"""
    pass


class CacheError(RAGApplicationError):
    """Raised when cache operations fail"""
    pass


class ValidationError(RAGApplicationError):
    """Raised when input validation fails"""
    pass


# Mapping of common errors to user-friendly messages
ERROR_MESSAGES = {
    "OPENAI_API_KEY_MISSING": "OpenAI API key is not configured. Please check your environment variables.",
    "GROQ_API_KEY_MISSING": "Groq API key is not configured. Please check your environment variables.",
    "YOUTUBE_TRANSCRIPT_UNAVAILABLE": "YouTube transcript is not available for the specified video.",
    "PDF_FILE_NOT_FOUND": "The specified PDF file could not be found.",
    "VECTOR_STORE_INITIALIZATION_FAILED": "Failed to initialize the vector store. Please try again.",
    "MODEL_INITIALIZATION_FAILED": "Failed to initialize the language model. Please check your API keys.",
    "EMBEDDING_GENERATION_FAILED": "Failed to generate embeddings for the documents.",
    "RETRIEVAL_FAILED": "Failed to retrieve relevant documents for your query.",
    "GENERATION_FAILED": "Failed to generate a response. Please try again.",
    "CACHE_UNAVAILABLE": "Cache service is temporarily unavailable.",
    "RATE_LIMIT_EXCEEDED": "API rate limit exceeded. Please wait before making another request.",
    "INVALID_INPUT": "The provided input is invalid. Please check your query and try again.",
    "SYSTEM_UNAVAILABLE": "The system is temporarily unavailable. Please try again later."
}


def get_user_friendly_message(error_code: str) -> str:
    """Get a user-friendly error message for the given error code"""
    return ERROR_MESSAGES.get(error_code, "An unexpected error occurred. Please try again.")


class ErrorHandler:
    """Centralized error handling utilities"""
    
    @staticmethod
    def handle_api_error(error: Exception, api_name: str) -> RAGApplicationError:
        """Handle API errors and convert to appropriate exception type"""
        error_message = f"{api_name} API error: {str(error)}"
        
        if "rate limit" in str(error).lower():
            return APIError(error_message, "RATE_LIMIT_EXCEEDED")
        elif "authentication" in str(error).lower() or "api key" in str(error).lower():
            return APIError(error_message, f"{api_name.upper()}_API_KEY_MISSING")
        else:
            return APIError(error_message, f"{api_name.upper()}_API_ERROR")
    
    @staticmethod
    def handle_document_error(error: Exception, document_type: str) -> DocumentProcessingError:
        """Handle document processing errors"""
        error_message = f"{document_type} processing error: {str(error)}"
        
        if "not found" in str(error).lower():
            return DocumentProcessingError(error_message, f"{document_type.upper()}_FILE_NOT_FOUND")
        else:
            return DocumentProcessingError(error_message, f"{document_type.upper()}_PROCESSING_FAILED")
    
    @staticmethod
    def handle_vector_store_error(error: Exception) -> VectorStoreError:
        """Handle vector store errors"""
        error_message = f"Vector store error: {str(error)}"
        return VectorStoreError(error_message, "VECTOR_STORE_ERROR")
    
    @staticmethod
    def handle_model_error(error: Exception) -> ModelError:
        """Handle model-related errors"""
        error_message = f"Model error: {str(error)}"
        
        if "initialization" in str(error).lower():
            return ModelError(error_message, "MODEL_INITIALIZATION_FAILED")
        elif "generation" in str(error).lower():
            return ModelError(error_message, "GENERATION_FAILED")
        else:
            return ModelError(error_message, "MODEL_ERROR")
