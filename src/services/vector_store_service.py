"""
Vector store service for managing document embeddings and similarity search.
"""

import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import settings
from src.core.exceptions import VectorStoreError, EmbeddingError, RetrievalError
from src.utils.logging_config import LoggerMixin, performance_logger
from src.utils.cache import cache_manager


class VectorStoreService(LoggerMixin):
    """Service for managing vector store operations"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.chunk_size = settings.rag.chunk_size
        self.chunk_overlap = settings.rag.chunk_overlap
        self.similarity_k = settings.rag.similarity_search_k
        self.similarity_threshold = settings.rag.similarity_threshold
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        try:
            self.logger.info("Initializing OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.api.openai_api_key,
                model=settings.api.openai_embedding_model,
                max_retries=settings.api.openai_max_retries
            )
            self.logger.info(f"Embeddings initialized with model: {settings.api.openai_embedding_model}")
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize embeddings: {str(e)}",
                "EMBEDDING_INITIALIZATION_FAILED"
            )
    
    def create_text_splitter(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> RecursiveCharacterTextSplitter:
        """
        Create text splitter with specified parameters
        
        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            
        Returns:
            Configured text splitter
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_documents(self, content: str, source: str = "unknown") -> List[Document]:
        """
        Split text content into chunks
        
        Args:
            content: Text content to split
            source: Source identifier for metadata
            
        Returns:
            List of Document objects
        """
        try:
            self.logger.info(f"Splitting content from {source} ({len(content)} characters)")
            
            splitter = self.create_text_splitter()
            
            # Create a document from the content
            document = Document(
                page_content=content,
                metadata={
                    "source": source,
                    "total_length": len(content)
                }
            )
            
            # Split the document
            chunks = splitter.split_documents([document])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_count": len(chunks),
                    "chunk_length": len(chunk.page_content)
                })
            
            self.logger.info(f"Created {len(chunks)} chunks from {source}")
            return chunks
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to split documents from {source}: {str(e)}",
                "DOCUMENT_SPLITTING_FAILED"
            )
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents
        
        Args:
            documents: List of documents to vectorize
            
        Returns:
            FAISS vector store
            
        Raises:
            VectorStoreError: When vector store creation fails
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Creating vector store from {len(documents)} documents")
            
            if not documents:
                raise VectorStoreError(
                    "Cannot create vector store from empty document list",
                    "EMPTY_DOCUMENT_LIST"
                )
            
            if not self.embeddings:
                raise VectorStoreError(
                    "Embeddings not initialized",
                    "EMBEDDINGS_NOT_INITIALIZED"
                )
            
            # Create vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "vector_store_creation",
                execution_time,
                document_count=len(documents),
                index_size=vector_store.index.ntotal
            )
            
            self.logger.info(
                f"Vector store created successfully with {vector_store.index.ntotal} vectors "
                f"in {execution_time:.2f}s"
            )
            
            return vector_store
            
        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "vector_store_creation_failed",
                execution_time,
                document_count=len(documents) if documents else 0,
                error=str(e)
            )
            
            raise VectorStoreError(
                f"Failed to create vector store: {str(e)}",
                "VECTOR_STORE_CREATION_FAILED"
            )
    
    def initialize_from_content(self, content_dict: Dict[str, str]) -> FAISS:
        """
        Initialize vector store from content dictionary
        
        Args:
            content_dict: Dictionary mapping source names to content
            
        Returns:
            Initialized FAISS vector store
        """
        try:
            self.logger.info(f"Initializing vector store from {len(content_dict)} sources")
            
            all_documents = []
            
            # Process each content source
            for source, content in content_dict.items():
                if content and content.strip():
                    documents = self.split_documents(content, source)
                    all_documents.extend(documents)
                else:
                    self.logger.warning(f"Empty content for source: {source}")
            
            if not all_documents:
                raise VectorStoreError(
                    "No valid documents to create vector store",
                    "NO_VALID_DOCUMENTS"
                )
            
            # Create vector store
            self.vector_store = self.create_vector_store(all_documents)
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.similarity_k}
            )
            
            self.logger.info("Vector store and retriever initialized successfully")
            return self.vector_store
            
        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            
            raise VectorStoreError(
                f"Failed to initialize vector store from content: {str(e)}",
                "VECTOR_STORE_INITIALIZATION_FAILED"
            )
    
    def search_similar(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            RetrievalError: When search fails
        """
        start_time = time.time()
        k = k or self.similarity_k
        
        try:
            if not self.vector_store:
                raise RetrievalError(
                    "Vector store not initialized",
                    "VECTOR_STORE_NOT_INITIALIZED"
                )
            
            self.logger.info(f"Searching for similar documents: '{query[:50]}...'")
            
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "similarity_search",
                execution_time,
                query_length=len(query),
                results_count=len(results),
                k=k
            )
            
            self.logger.info(f"Found {len(results)} similar documents in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            if isinstance(e, RetrievalError):
                raise
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "similarity_search_failed",
                execution_time,
                query_length=len(query) if query else 0,
                k=k,
                error=str(e)
            )
            
            raise RetrievalError(
                f"Failed to search similar documents: {str(e)}",
                "SIMILARITY_SEARCH_FAILED"
            )
    
    def search_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """
        Search for similar documents with similarity scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        k = k or self.similarity_k
        
        try:
            if not self.vector_store:
                raise RetrievalError(
                    "Vector store not initialized",
                    "VECTOR_STORE_NOT_INITIALIZED"
                )
            
            self.logger.info(f"Searching with scores: '{query[:50]}...'")
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Filter by threshold if set
            if self.similarity_threshold > 0:
                filtered_results = [
                    (doc, score) for doc, score in results
                    if score >= self.similarity_threshold
                ]
                self.logger.info(
                    f"Filtered {len(results)} to {len(filtered_results)} results "
                    f"using threshold {self.similarity_threshold}"
                )
                return filtered_results
            
            return results
            
        except Exception as e:
            if isinstance(e, RetrievalError):
                raise
            
            raise RetrievalError(
                f"Failed to search with scores: {str(e)}",
                "SIMILARITY_SEARCH_WITH_SCORES_FAILED"
            )
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict] = None):
        """
        Get configured retriever
        
        Args:
            search_type: Type of search (similarity, mmr, etc.)
            search_kwargs: Additional search parameters
            
        Returns:
            Configured retriever
        """
        if not self.vector_store:
            raise RetrievalError(
                "Vector store not initialized",
                "VECTOR_STORE_NOT_INITIALIZED"
            )
        
        search_kwargs = search_kwargs or {"k": self.similarity_k}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def save_vector_store(self, save_path: str):
        """
        Save vector store to disk
        
        Args:
            save_path: Path to save the vector store
        """
        try:
            if not self.vector_store:
                raise VectorStoreError(
                    "No vector store to save",
                    "NO_VECTOR_STORE"
                )
            
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(str(save_dir))
            
            self.logger.info(f"Vector store saved to {save_path}")
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to save vector store: {str(e)}",
                "VECTOR_STORE_SAVE_FAILED"
            )
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """
        Load vector store from disk
        
        Args:
            load_path: Path to load the vector store from
            
        Returns:
            Loaded FAISS vector store
        """
        try:
            load_dir = Path(load_path)
            if not load_dir.exists():
                raise VectorStoreError(
                    f"Vector store path does not exist: {load_path}",
                    "VECTOR_STORE_PATH_NOT_FOUND"
                )
            
            # Initialize embeddings if not already done
            if not self.embeddings:
                self._initialize_embeddings()
            
            # Load FAISS index
            self.vector_store = FAISS.load_local(str(load_dir), self.embeddings)
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.similarity_k}
            )
            
            self.logger.info(f"Vector store loaded from {load_path}")
            return self.vector_store
            
        except Exception as e:
            if isinstance(e, VectorStoreError):
                raise
            
            raise VectorStoreError(
                f"Failed to load vector store: {str(e)}",
                "VECTOR_STORE_LOAD_FAILED"
            )
    
    def get_stats(self) -> Dict:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with vector store stats
        """
        if not self.vector_store:
            return {"initialized": False}
        
        try:
            stats = {
                "initialized": True,
                "total_vectors": self.vector_store.index.ntotal,
                "vector_dimension": self.vector_store.index.d,
                "embedding_model": settings.api.openai_embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "similarity_k": self.similarity_k,
                "similarity_threshold": self.similarity_threshold
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get vector store stats: {e}")
            return {"initialized": True, "error": str(e)}
