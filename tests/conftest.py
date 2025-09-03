"""
Test configuration and fixtures for the RAG application.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Test environment setup
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"
os.environ["CACHE_ENABLED"] = "false"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key for testing"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
        yield "test-openai-key"


@pytest.fixture
def mock_groq_api_key():
    """Mock Groq API key for testing"""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
        yield "test-groq-key"


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Write some mock PDF content
        f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_youtube_transcript():
    """Sample YouTube transcript data for testing"""
    return [
        {"text": "Welcome to this video about transformers", "start": 0.0, "duration": 3.0},
        {"text": "In this paper, we introduce the attention mechanism", "start": 3.0, "duration": 4.0},
        {"text": "The transformer architecture is revolutionary", "start": 7.0, "duration": 3.5}
    ]


@pytest.fixture
def sample_documents():
    """Sample document chunks for testing"""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="The Transformer is a model architecture eschewing recurrence.",
            metadata={"source": "paper", "chunk_id": 0}
        ),
        Document(
            page_content="Attention mechanisms allow modeling of dependencies.",
            metadata={"source": "paper", "chunk_id": 1}
        ),
        Document(
            page_content="Multi-head attention allows the model to jointly attend.",
            metadata={"source": "youtube", "chunk_id": 0}
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Mock FAISS vector store for testing"""
    mock_store = Mock()
    mock_store.similarity_search.return_value = []
    mock_store.similarity_search_with_score.return_value = []
    mock_store.as_retriever.return_value = Mock()
    mock_store.index.ntotal = 100
    mock_store.index.d = 1536
    return mock_store


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings for testing"""
    mock_emb = Mock()
    mock_emb.embed_documents.return_value = [[0.1] * 1536] * 3
    mock_emb.embed_query.return_value = [0.1] * 1536
    return mock_emb


@pytest.fixture
def mock_llm():
    """Mock language model for testing"""
    mock_model = Mock()
    mock_model.invoke.return_value = Mock(content="This is a test response about transformers.")
    return mock_model


class TestSettings:
    """Test-specific settings and configurations"""
    
    @staticmethod
    def get_test_config():
        """Get test configuration"""
        return {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "similarity_k": 3,
            "youtube_video_id": "test_video_id",
            "pdf_path": "test.pdf"
        }
