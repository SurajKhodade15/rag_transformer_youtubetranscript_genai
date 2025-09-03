"""
PDF processing service for extracting and processing document content.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader

from config.settings import settings
from src.core.exceptions import PDFProcessingError, DocumentProcessingError
from src.utils.logging_config import LoggerMixin, performance_logger
from src.utils.cache import cache_manager


class PDFService(LoggerMixin):
    """Service for handling PDF document operations"""
    
    def __init__(self):
        self.cache_ttl = 86400  # 24 hours for PDF content
    
    @cache_manager.cache_result("pdf_content", ttl=86400)
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            PDFProcessingError: When PDF processing fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Extracting text from PDF: {pdf_path}")
            
            # Validate file exists
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise PDFProcessingError(
                    f"PDF file not found: {pdf_path}",
                    "PDF_FILE_NOT_FOUND"
                )
            
            if not pdf_file.suffix.lower() == '.pdf':
                raise PDFProcessingError(
                    f"File is not a PDF: {pdf_path}",
                    "INVALID_FILE_TYPE"
                )
            
            # Load PDF using LangChain
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            if not documents:
                raise PDFProcessingError(
                    f"No content extracted from PDF: {pdf_path}",
                    "EMPTY_PDF_CONTENT"
                )
            
            # Extract text from all pages
            text_content = ""
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    text_content += doc.page_content + "\n\n"
                else:
                    text_content += str(doc) + "\n\n"
            
            # Clean and process text
            text_content = self._clean_pdf_text(text_content)
            
            if not text_content.strip():
                raise PDFProcessingError(
                    f"No readable text content found in PDF: {pdf_path}",
                    "NO_READABLE_CONTENT"
                )
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "pdf_text_extraction",
                execution_time,
                pdf_path=pdf_path,
                content_length=len(text_content),
                page_count=len(documents)
            )
            
            self.logger.info(
                f"Successfully extracted text from {pdf_path} "
                f"({len(documents)} pages, {len(text_content)} characters)"
            )
            
            return text_content
            
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "pdf_text_extraction_failed",
                execution_time,
                pdf_path=pdf_path,
                error=str(e)
            )
            
            raise PDFProcessingError(
                f"Failed to extract text from PDF {pdf_path}: {str(e)}",
                "PDF_PROCESSING_FAILED"
            )
    
    def extract_with_metadata(self, pdf_path: str) -> Dict:
        """
        Extract text content along with metadata from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with content and metadata
        """
        try:
            self.logger.info(f"Extracting text and metadata from PDF: {pdf_path}")
            
            # Validate file exists
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise PDFProcessingError(
                    f"PDF file not found: {pdf_path}",
                    "PDF_FILE_NOT_FOUND"
                )
            
            # Load PDF using LangChain
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            if not documents:
                raise PDFProcessingError(
                    f"No content extracted from PDF: {pdf_path}",
                    "EMPTY_PDF_CONTENT"
                )
            
            # Process documents and extract metadata
            pages_content = []
            total_text = ""
            
            for i, doc in enumerate(documents):
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                page_metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                pages_content.append({
                    "page_number": i + 1,
                    "content": page_content,
                    "metadata": page_metadata
                })
                
                total_text += page_content + "\n\n"
            
            # Clean total text
            total_text = self._clean_pdf_text(total_text)
            
            return {
                "file_path": pdf_path,
                "total_pages": len(documents),
                "total_content": total_text,
                "total_characters": len(total_text),
                "pages": pages_content,
                "file_size": pdf_file.stat().st_size,
                "file_modified": pdf_file.stat().st_mtime
            }
            
        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            
            raise PDFProcessingError(
                f"Failed to extract content and metadata from PDF {pdf_path}: {str(e)}",
                "PDF_PROCESSING_FAILED"
            )
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean PDF text by removing artifacts and normalizing
        
        Args:
            text: Raw PDF text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace("\x00", "")  # Null bytes
        text = text.replace("\ufffd", "")  # Replacement characters
        
        # Normalize line breaks
        text = text.replace("\n\n\n", "\n\n")
        
        return text.strip()
    
    def validate_pdf(self, pdf_path: str) -> Dict:
        """
        Validate PDF file and return basic information
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Validation result with file information
        """
        try:
            pdf_file = Path(pdf_path)
            
            result = {
                "valid": False,
                "path": pdf_path,
                "exists": pdf_file.exists(),
                "is_pdf": False,
                "readable": False,
                "file_size": 0,
                "error": None
            }
            
            if not pdf_file.exists():
                result["error"] = "File does not exist"
                return result
            
            result["file_size"] = pdf_file.stat().st_size
            result["is_pdf"] = pdf_file.suffix.lower() == '.pdf'
            
            if not result["is_pdf"]:
                result["error"] = "File is not a PDF"
                return result
            
            # Try to load the PDF
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                result["readable"] = len(documents) > 0
                result["page_count"] = len(documents)
                
                if result["readable"]:
                    result["valid"] = True
                else:
                    result["error"] = "PDF contains no readable content"
                    
            except Exception as e:
                result["error"] = f"Failed to read PDF: {str(e)}"
            
            return result
            
        except Exception as e:
            return {
                "valid": False,
                "path": pdf_path,
                "exists": False,
                "is_pdf": False,
                "readable": False,
                "file_size": 0,
                "error": f"Validation failed: {str(e)}"
            }
    
    def get_pdf_info(self, pdf_path: str) -> Dict:
        """
        Get detailed information about PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            validation = self.validate_pdf(pdf_path)
            
            if not validation["valid"]:
                return validation
            
            # Get content sample
            try:
                text_content = self.extract_text(pdf_path)
                sample_length = min(500, len(text_content))
                content_sample = text_content[:sample_length]
                
                validation.update({
                    "total_characters": len(text_content),
                    "content_sample": content_sample,
                    "has_content": len(text_content.strip()) > 0
                })
                
            except Exception as e:
                validation["content_error"] = str(e)
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Failed to get PDF info for {pdf_path}: {e}")
            return {
                "valid": False,
                "path": pdf_path,
                "error": f"Info extraction failed: {str(e)}"
            }
    
    def clear_cache(self, pdf_path: Optional[str] = None):
        """
        Clear PDF content cache
        
        Args:
            pdf_path: Specific PDF path to clear, or None to clear all
        """
        if pdf_path:
            cache_key = cache_manager._generate_key("pdf_content", pdf_path)
            cache_manager.delete(cache_key)
            self.logger.info(f"Cleared cache for PDF {pdf_path}")
        else:
            self.logger.info("Cache clearing for all PDFs not implemented")
