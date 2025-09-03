"""
RAG (Retrieval-Augmented Generation) service that orchestrates the entire pipeline.
"""

import time
from typing import List, Dict, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from config.settings import settings
from src.core.exceptions import ModelError, RetrievalError, RAGApplicationError
from src.utils.logging_config import LoggerMixin, performance_logger
from src.utils.cache import cache_manager
from src.services.youtube_service import YouTubeService
from src.services.pdf_service import PDFService
from src.services.vector_store_service import VectorStoreService


class RAGService(LoggerMixin):
    """Main RAG service that orchestrates document retrieval and generation"""
    
    def __init__(self):
        self.llm = None
        self.youtube_service = YouTubeService()
        self.pdf_service = PDFService()
        self.vector_store_service = VectorStoreService()
        self.main_chain = None
        self.initialized = False
        
        # Initialize LLM
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            self.logger.info("Initializing Groq LLM")
            self.llm = ChatGroq(
                groq_api_key=settings.api.groq_api_key,
                model=settings.api.groq_model,
                temperature=settings.api.groq_temperature,
                max_tokens=settings.api.groq_max_tokens,
                max_retries=3
            )
            self.logger.info(f"LLM initialized with model: {settings.api.groq_model}")
            
        except Exception as e:
            raise ModelError(
                f"Failed to initialize LLM: {str(e)}",
                "LLM_INITIALIZATION_FAILED"
            )
    
    def initialize_system(self, youtube_video_id: Optional[str] = None, pdf_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize the complete RAG system
        
        Args:
            youtube_video_id: YouTube video ID (default from settings)
            pdf_path: Path to PDF file (default from settings)
            
        Returns:
            Initialization status and metrics
        """
        start_time = time.time()
        
        try:
            self.logger.info("Initializing RAG system")
            
            # Use defaults from settings if not provided
            youtube_video_id = youtube_video_id or settings.rag.youtube_video_id
            pdf_path = pdf_path or settings.rag.pdf_path
            
            # Load content from sources
            self.logger.info("Loading content from sources")
            content_dict = {}
            
            # Load YouTube transcript
            try:
                youtube_content = self.youtube_service.get_transcript(youtube_video_id)
                content_dict["youtube"] = youtube_content
                self.logger.info(f"YouTube content loaded: {len(youtube_content)} characters")
            except Exception as e:
                self.logger.error(f"Failed to load YouTube content: {e}")
                content_dict["youtube"] = ""
            
            # Load PDF content
            try:
                pdf_content = self.pdf_service.extract_text(pdf_path)
                content_dict["pdf"] = pdf_content
                self.logger.info(f"PDF content loaded: {len(pdf_content)} characters")
            except Exception as e:
                self.logger.error(f"Failed to load PDF content: {e}")
                content_dict["pdf"] = ""
            
            # Validate we have some content
            total_content = sum(len(content) for content in content_dict.values())
            if total_content == 0:
                raise RAGApplicationError(
                    "No content could be loaded from any source",
                    "NO_CONTENT_LOADED"
                )
            
            # Initialize vector store
            self.logger.info("Creating vector store")
            self.vector_store_service.initialize_from_content(content_dict)
            
            # Setup RAG chain
            self._setup_rag_chain()
            
            self.initialized = True
            execution_time = time.time() - start_time
            
            # Get system stats
            vector_stats = self.vector_store_service.get_stats()
            
            performance_logger.log_execution_time(
                "rag_system_initialization",
                execution_time,
                total_content_length=total_content,
                sources_loaded=len([k for k, v in content_dict.items() if v]),
                **vector_stats
            )
            
            result = {
                "success": True,
                "execution_time": execution_time,
                "sources": {
                    "youtube": {
                        "loaded": bool(content_dict.get("youtube")),
                        "content_length": len(content_dict.get("youtube", ""))
                    },
                    "pdf": {
                        "loaded": bool(content_dict.get("pdf")),
                        "content_length": len(content_dict.get("pdf", ""))
                    }
                },
                "vector_store": vector_stats,
                "total_content_length": total_content
            }
            
            self.logger.info(f"RAG system initialized successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.initialized = False
            execution_time = time.time() - start_time
            
            performance_logger.log_execution_time(
                "rag_system_initialization_failed",
                execution_time,
                error=str(e)
            )
            
            if isinstance(e, RAGApplicationError):
                raise
            
            raise RAGApplicationError(
                f"Failed to initialize RAG system: {str(e)}",
                "RAG_INITIALIZATION_FAILED"
            )
    
    def _setup_rag_chain(self):
        """Setup the RAG chain for question answering"""
        try:
            self.logger.info("Setting up RAG chain")
            
            # Create prompt template
            prompt = PromptTemplate(
                template="""You are a helpful assistant specializing in the "Attention Is All You Need" paper and transformer architecture.
                
Use the provided context to answer the question accurately and comprehensively. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available.

Context:
{context}

Question: {question}

Answer: Provide a detailed, accurate response based on the context provided. Include specific details and explanations when possible.""",
                input_variables=['context', 'question']
            )
            
            # Create context formatter
            def format_docs(retrieved_docs):
                """Format retrieved documents into context string"""
                if not retrieved_docs:
                    return "No relevant context found."
                
                context_parts = []
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', 'unknown')
                    chunk_id = doc.metadata.get('chunk_id', i)
                    content = doc.page_content
                    
                    context_parts.append(f"[Source: {source}, Chunk: {chunk_id}]\n{content}")
                
                return "\n\n".join(context_parts)
            
            # Create retriever
            retriever = self.vector_store_service.get_retriever()
            
            # Create parallel chain for context and question
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            
            # Create main chain
            self.main_chain = parallel_chain | prompt | self.llm | StrOutputParser()
            
            self.logger.info("RAG chain setup completed")
            
        except Exception as e:
            raise RAGApplicationError(
                f"Failed to setup RAG chain: {str(e)}",
                "RAG_CHAIN_SETUP_FAILED"
            )
    
    @cache_manager.cache_result("rag_query", ttl=1800)  # 30 minutes
    def query(self, question: str) -> str:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User question
            
        Returns:
            Generated answer
            
        Raises:
            RAGApplicationError: When query processing fails
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                raise RAGApplicationError(
                    "RAG system not initialized",
                    "SYSTEM_NOT_INITIALIZED"
                )
            
            if not question or not question.strip():
                raise RAGApplicationError(
                    "Empty question provided",
                    "EMPTY_QUESTION"
                )
            
            self.logger.info(f"Processing query: '{question[:100]}...'")
            
            # Process query through RAG chain
            answer = self.main_chain.invoke(question.strip())
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "rag_query",
                execution_time,
                question_length=len(question),
                answer_length=len(answer) if answer else 0
            )
            
            self.logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return answer
            
        except Exception as e:
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "rag_query_failed",
                execution_time,
                question_length=len(question) if question else 0,
                error=str(e)
            )
            
            if isinstance(e, RAGApplicationError):
                raise
            
            raise RAGApplicationError(
                f"Failed to process query: {str(e)}",
                "QUERY_PROCESSING_FAILED"
            )
    
    def search_documents(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search for relevant documents without generating an answer
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            if not self.initialized:
                raise RAGApplicationError(
                    "RAG system not initialized",
                    "SYSTEM_NOT_INITIALIZED"
                )
            
            documents = self.vector_store_service.search_similar(query, k)
            
            results = []
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            return results
            
        except Exception as e:
            if isinstance(e, (RAGApplicationError, RetrievalError)):
                raise
            
            raise RetrievalError(
                f"Failed to search documents: {str(e)}",
                "DOCUMENT_SEARCH_FAILED"
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and metrics
        
        Returns:
            System status information
        """
        try:
            status = {
                "initialized": self.initialized,
                "llm_initialized": self.llm is not None,
                "services": {
                    "youtube": "available",
                    "pdf": "available", 
                    "vector_store": "available" if self.vector_store_service.vector_store else "not_initialized"
                }
            }
            
            if self.initialized:
                status["vector_store_stats"] = self.vector_store_service.get_stats()
                status["chain_ready"] = self.main_chain is not None
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "initialized": False,
                "error": str(e)
            }
    
    def reinitialize(self, **kwargs) -> Dict[str, Any]:
        """
        Reinitialize the system with new parameters
        
        Args:
            **kwargs: Initialization parameters
            
        Returns:
            Reinitialization result
        """
        try:
            self.logger.info("Reinitializing RAG system")
            
            # Reset state
            self.initialized = False
            self.main_chain = None
            self.vector_store_service = VectorStoreService()
            
            # Reinitialize
            return self.initialize_system(**kwargs)
            
        except Exception as e:
            self.logger.error(f"Failed to reinitialize system: {e}")
            if isinstance(e, RAGApplicationError):
                raise
            
            raise RAGApplicationError(
                f"System reinitialization failed: {str(e)}",
                "REINITIALIZATION_FAILED"
            )
    
    def clear_cache(self):
        """Clear all caches"""
        try:
            cache_manager.clear()
            self.youtube_service.clear_cache()
            self.pdf_service.clear_cache()
            self.logger.info("All caches cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
    
    def get_paper_summary(self) -> str:
        """Get a summary of the Attention paper"""
        return """
        **"Attention Is All You Need" - Transformer Architecture**
        
        This groundbreaking 2017 paper by Vaswani et al. introduced the Transformer architecture, revolutionizing natural language processing and deep learning. 

        **Key Innovations:**
        
        🔹 **Self-Attention Mechanism**: Allows the model to weigh the importance of different words in a sequence when processing each word
        
        🔹 **Parallel Processing**: Unlike RNNs, Transformers can process all positions simultaneously, dramatically improving training speed
        
        🔹 **Multi-Head Attention**: Uses multiple attention heads to capture different types of relationships between words
        
        🔹 **Positional Encoding**: Injects sequence order information since the architecture doesn't inherently understand position
        
        🔹 **Encoder-Decoder Structure**: Six encoder and decoder layers with residual connections and layer normalization
        
        **Impact**: The Transformer architecture became the foundation for modern language models like BERT, GPT, T5, and many others, fundamentally changing how we approach NLP tasks.
        """
