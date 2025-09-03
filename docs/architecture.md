# Architecture Overview

## 🏗️ System Architecture

The RAG (Retrieval-Augmented Generation) application follows a modern, industry-standard architecture with clear separation of concerns, dependency injection, and comprehensive error handling.

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   RAG Service    │───▶│  Vector Store   │
│    (main.py)    │    │  (Orchestrator)  │    │    Service      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  UI Components  │    │    Services      │    │   Data Sources  │
│   & Theming     │    │  YouTube | PDF   │    │  YouTube | PDF  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Directory Structure

```
├── src/                          # Source code
│   ├── core/                     # Core utilities and exceptions
│   │   ├── __init__.py
│   │   └── exceptions.py         # Custom exception classes
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── youtube_service.py    # YouTube transcript handling
│   │   ├── pdf_service.py        # PDF document processing
│   │   ├── vector_store_service.py # Vector database operations
│   │   └── rag_service.py        # Main RAG orchestration
│   ├── ui/                       # User interface components
│   │   ├── __init__.py
│   │   └── components.py         # Streamlit UI components
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── logging_config.py     # Structured logging
│       ├── cache.py              # Caching mechanisms
│       └── monitoring.py         # Performance monitoring
├── config/                       # Configuration management
│   ├── __init__.py
│   └── settings.py              # Pydantic settings
├── tests/                        # Test suite
│   ├── conftest.py              # Test configuration
│   └── test_*.py                # Test modules
├── docs/                         # Documentation
├── logs/                         # Log files
├── data/                         # Data storage
│   └── vector_store/            # Vector database files
├── main.py                       # Application entry point
├── requirements.txt              # Dependencies
└── .env                         # Environment variables
```

## 🔧 Core Components

### 1. Configuration Management (`config/`)

**Purpose**: Centralized configuration using Pydantic for validation and type safety.

**Key Features**:
- Environment-based configuration
- Type validation and conversion
- Hierarchical settings (API, RAG, App, Database)
- Development/Production environment handling

```python
from config.settings import settings

# Access typed, validated configuration
api_key = settings.api.openai_api_key
chunk_size = settings.rag.chunk_size
```

### 2. Service Layer (`src/services/`)

**Purpose**: Business logic encapsulation with clear responsibilities.

#### RAGService (Orchestrator)
- Coordinates all other services
- Manages the complete RAG pipeline
- Handles query processing and response generation

#### YouTubeService
- Fetches and processes YouTube transcripts
- Handles multiple languages and fallbacks
- Implements caching for performance

#### PDFService
- Extracts text from PDF documents
- Validates file integrity
- Provides metadata extraction

#### VectorStoreService
- Manages FAISS vector database
- Handles document chunking and embedding
- Provides similarity search capabilities

### 3. Core Utilities (`src/core/`)

**Purpose**: Foundation classes and error handling.

#### Exception Hierarchy
```python
RAGApplicationError (Base)
├── ConfigurationError
├── APIError
│   ├── OpenAIAPIError
│   └── GroqAPIError
├── DocumentProcessingError
│   ├── PDFProcessingError
│   └── TranscriptProcessingError
├── VectorStoreError
├── EmbeddingError
├── RetrievalError
├── ModelError
└── CacheError
```

### 4. User Interface (`src/ui/`)

**Purpose**: Modern, responsive Streamlit interface.

**Features**:
- Component-based architecture
- Custom theming and styling
- Real-time status updates
- Chat interface with history
- Performance metrics display

### 5. Utilities (`src/utils/`)

**Purpose**: Cross-cutting concerns and infrastructure.

#### Logging
- Structured JSON logging for production
- Performance metrics logging
- Contextual log enrichment
- Rotating file handlers

#### Caching
- In-memory and Redis support
- TTL-based expiration
- Cache statistics and monitoring
- Decorator-based caching

#### Monitoring
- Performance metric collection
- Health check system
- System resource monitoring
- Dashboard data aggregation

## 🔄 Data Flow

### 1. System Initialization
```
1. Load configuration from environment
2. Initialize services (YouTube, PDF, VectorStore)
3. Load content from sources
4. Create document embeddings
5. Build FAISS vector index
6. Setup RAG chain (Retriever + LLM)
```

### 2. Query Processing
```
1. User submits question via UI
2. RAGService validates input
3. VectorStoreService performs similarity search
4. Relevant documents retrieved and formatted
5. LLM generates response using context
6. Response displayed in UI
7. Interaction logged and cached
```

## 🛡️ Error Handling Strategy

### Three-Layer Approach

1. **Service Layer**: Catches and converts exceptions to domain-specific errors
2. **Application Layer**: Handles domain errors and provides user-friendly messages
3. **UI Layer**: Displays appropriate feedback and maintains application state

### Error Types

- **Recoverable**: Temporary API failures, cache misses
- **Configuration**: Missing API keys, invalid settings
- **User Input**: Invalid queries, empty content
- **System**: Resource exhaustion, service unavailability

## 📊 Performance Considerations

### Caching Strategy
- **L1 (Memory)**: Frequently accessed data, embeddings
- **L2 (Redis)**: Shared cache across instances
- **L3 (Disk)**: Vector store persistence

### Optimization Techniques
- Lazy loading of heavy components
- Async operations where possible
- Connection pooling for external APIs
- Efficient document chunking strategies

## 🧪 Testing Architecture

### Test Pyramid
- **Unit Tests**: Individual service methods
- **Integration Tests**: Service interactions
- **System Tests**: End-to-end workflows
- **Performance Tests**: Load and stress testing

### Test Utilities
- Comprehensive fixtures and mocks
- Test data generators
- Performance benchmarking
- Coverage reporting

## 🚀 Deployment Considerations

### Environment Separation
- **Development**: Local with debug logging
- **Staging**: Production-like with test data
- **Production**: Optimized with monitoring

### Scalability
- Stateless service design
- External cache for multi-instance support
- Database connection pooling
- Resource monitoring and alerting

This architecture provides a solid foundation for enterprise-grade applications while maintaining developer productivity and code maintainability.
