"""
Simple RAG Service
Consolidates all RAG functionality into one clean service class.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

# Core dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import PyPDF2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGService:
    """Simple RAG service that handles all functionality"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG service"""
        self.config = config
        self.vector_store = None
        self.qa_chain = None
        
        try:
            logger.info("🚀 Initializing RAG Service...")
            # Initialize components
            self._setup_llm()
            self._setup_embeddings()
            self._initialize_or_load_vector_store()
            self._setup_qa_chain()
            logger.info("✅ RAG Service initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Service: {e}")
            raise
    
    def _setup_llm(self):
        """Setup the language model"""
        try:
            self.llm = ChatGroq(
                groq_api_key=self.config['groq_api_key'],
                model_name=self.config['groq_model'],
                temperature=self.config['groq_temperature'],
                max_tokens=self.config['groq_max_tokens']
            )
            logger.info("✅ LLM initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def _setup_embeddings(self):
        """Setup OpenAI embeddings"""
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.config['openai_api_key'],
                model=self.config['openai_embedding_model']
            )
            logger.info("✅ Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embeddings: {e}")
            raise
    
    def _get_youtube_transcript(self, video_id: str) -> str:
        """Get transcript from YouTube video"""
        try:
            logger.info(f"📹 Fetching YouTube transcript for video: {video_id}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = ' '.join([item['text'] for item in transcript_list])
            logger.info(f"✅ YouTube transcript fetched successfully ({len(transcript_text)} characters)")
            return transcript_text
        except AttributeError as e:
            logger.error(f"❌ YouTube API method error: {e}")
            # Try alternative method if available
            try:
                transcript_list = YouTubeTranscriptApi().get_transcript(video_id)
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                logger.info(f"✅ YouTube transcript fetched successfully with alternative method ({len(transcript_text)} characters)")
                return transcript_text
            except Exception as e2:
                logger.error(f"❌ Failed to get YouTube transcript with alternative method: {e2}")
                return ""
        except Exception as e:
            logger.error(f"❌ Error getting YouTube transcript: {e}")
            return ""
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            logger.info(f"📄 Extracting text from PDF: {pdf_path}")
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + "\n"
            logger.info(f"✅ PDF text extracted successfully ({len(text)} characters)")
            return text
        except Exception as e:
            logger.error(f"❌ Error extracting PDF text: {e}")
            return ""
    
    def _create_documents(self) -> List[Document]:
        """Create documents from YouTube and PDF sources"""
        documents = []
        
        # Get YouTube transcript
        youtube_text = self._get_youtube_transcript(self.config['youtube_video_id'])
        if youtube_text:
            documents.append(Document(
                page_content=youtube_text,
                metadata={"source": "youtube", "video_id": self.config['youtube_video_id']}
            ))
        
        # Get PDF text
        pdf_text = self._extract_pdf_text(self.config['pdf_path'])
        if pdf_text:
            documents.append(Document(
                page_content=pdf_text,
                metadata={"source": "pdf", "file": self.config['pdf_path']}
            ))
        
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                ))
        
        return split_docs
    
    def _initialize_or_load_vector_store(self):
        """Initialize vector store or load from cache"""
        cache_path = Path(self.config['vector_store_path'])
        
        # Try to load existing vector store
        if cache_path.exists():
            try:
                logger.info(f"📦 Loading cached vector store from: {cache_path}")
                self.vector_store = FAISS.load_local(
                    str(cache_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ Loaded existing vector store from cache")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cached vector store: {e}")
                logger.info("📦 Will create new vector store instead...")
        
        # Create new vector store
        logger.info("🔧 Creating new vector store...")
        documents = self._create_documents()
        
        if not documents:
            raise Exception("No documents found to create vector store")
        
        # Split documents
        split_docs = self._split_documents(documents)
        logger.info(f"📄 Created {len(split_docs)} document chunks")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(cache_path))
        logger.info(f"💾 Saved vector store to {cache_path}")
    
    def _setup_qa_chain(self):
        """Setup the QA chain"""
        try:
            logger.info("🔗 Setting up QA chain...")
            # Create custom prompt
            prompt_template = """
            You are an expert AI assistant specializing in the "Attention Is All You Need" paper and transformer architecture.
            
            Use the following context to answer the question. If you cannot find the answer in the context, 
            say "I don't have enough information in the provided context to answer that question."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer: Provide a detailed, accurate answer based on the context. Include specific details from the paper when relevant.
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config['similarity_k']}
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            logger.info("✅ QA chain initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to setup QA chain: {e}")
            raise
    
    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer"""
        if not self.qa_chain:
            raise Exception("QA chain not initialized")
        
        try:
            # Get response
            response = self.qa_chain({"query": question})
            answer = response['result']
            
            # Add source information
            sources = []
            for doc in response.get('source_documents', []):
                source = doc.metadata.get('source', 'unknown')
                if source == 'youtube':
                    sources.append("📹 YouTube video")
                elif source == 'pdf':
                    sources.append("📄 Research paper")
                else:
                    sources.append(f"📋 {source}")
            
            if sources:
                unique_sources = list(set(sources))
                answer += f"\n\n**Sources:** {', '.join(unique_sources)}"
            
            return answer
            
        except Exception as e:
            return f"❌ Error processing question: {str(e)}"
    
    def get_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """Get similar documents for a query"""
        if not self.vector_store:
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            "vector_store_initialized": self.vector_store is not None,
            "qa_chain_initialized": self.qa_chain is not None,
            "llm_model": self.config['groq_model'],
            "embedding_model": self.config['openai_embedding_model'],
            "pdf_available": os.path.exists(self.config['pdf_path']),
            "youtube_video_id": self.config['youtube_video_id']
        }
