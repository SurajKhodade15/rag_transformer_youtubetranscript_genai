# 🔍 Attention Mechanism RAG Assistant

A sophisticated, **production-ready** Retrieval-Augmented Generation (RAG) application that provides intelligent answers about the groundbreaking "Attention Is All You Need" paper using both YouTube video explanations and the original research paper.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)

## 🌟 Features

### 🎯 Core Functionality
- **Dual Source RAG**: Combines information from YouTube video transcript and original research paper
- **Intelligent Q&A**: Ask complex questions about transformer architecture and attention mechanisms
- **Fast Responses**: Powered by Groq's lightning-fast Llama 3.3 70B model
- **Semantic Search**: Uses OpenAI embeddings with FAISS vector database for accurate retrieval
- **Rich UI**: Beautiful, responsive Streamlit interface with custom styling

### 🏭 Production Features
- **Industry-Standard Architecture**: Clean separation of concerns with service-oriented design
- **Comprehensive Error Handling**: Custom exception hierarchy with user-friendly messages
- **Performance Monitoring**: Built-in metrics collection and health checks
- **Caching System**: Multi-layer caching (Memory/Redis) for optimal performance
- **Structured Logging**: JSON logging with performance metrics and debugging support
- **Configuration Management**: Type-safe settings with Pydantic validation
- **Test Coverage**: Comprehensive test suite with fixtures and mocks
- **Documentation**: Complete API docs, deployment guides, and architecture overview

### 📚 Knowledge Sources
1. **YouTube Video**: Technical explanation of the Attention paper (ID: bCz4OMemCcA)
2. **Research Paper**: Original "Attention Is All You Need" PDF document
3. **Comprehensive Coverage**: Architecture details, self-attention, multi-head attention, positional encoding

### 🔧 Technical Stack
- **Frontend**: Streamlit with custom CSS styling and component architecture
- **LLM**: Groq Llama 3.3 70B for fast, accurate responses
- **Embeddings**: OpenAI text-embedding-3-small for semantic understanding
- **Vector Database**: FAISS for efficient similarity search
- **Framework**: LangChain for RAG pipeline orchestration
- **Document Processing**: PyPDF for PDF parsing, YouTube Transcript API
- **Caching**: Redis/Memory caching with TTL support
- **Monitoring**: Performance metrics, health checks, and system monitoring
- **Testing**: Pytest with comprehensive fixtures and mocks

## 🏗️ Architecture

### Directory Structure
```
├── src/                          # Source code
│   ├── core/                     # Core utilities and exceptions
│   ├── services/                 # Business logic services
│   ├── ui/                       # User interface components
│   └── utils/                    # Utility modules (logging, caching, monitoring)
├── config/                       # Configuration management
├── tests/                        # Comprehensive test suite
├── docs/                         # Documentation
├── logs/                         # Application logs
├── data/                         # Data storage and vector store
├── main.py                       # Application entry point
└── requirements.txt              # Dependencies
```

### Service Architecture
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

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- API keys for OpenAI and Groq
- Git for cloning the repository

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag_transformer_youtubetranscript_genai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file:
   ```bash
   # Required API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   
   # Optional (for debugging)
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   LANGCHAIN_PROJECT=your_project_name
   LANGCHAIN_TRACING_V2=true
   
   # Application Settings
   ENVIRONMENT=development
   DEBUG=true
   LOG_LEVEL=INFO
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Access the app**
   Open your browser and navigate to `http://localhost:8501`

## 📖 How to Use

### Getting Started
1. **Initialize the System**: Click "🚀 Initialize RAG System" in the sidebar
2. **Wait for Loading**: The app will load YouTube transcript and PDF content
3. **Start Asking**: Type your questions in the input field
4. **Explore**: Use example questions or ask your own

### Example Questions
- "What is the self-attention mechanism and how does it work?"
- "How do transformers differ from RNNs and CNNs?"
- "What are the key components of the transformer architecture?"
- "How does multi-head attention improve the model?"
- "What is positional encoding and why is it needed?"
- "What were the main results and achievements of this paper?"

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_youtube_service.py

# Run with verbose output
pytest -v
```

## 📊 Monitoring

### Health Checks
The application includes built-in health checks:
- System resource monitoring
- API connectivity checks
- Service availability status

### Performance Metrics
- Query execution times
- Cache hit/miss ratios
- System resource usage
- API response times

### Logging
- Structured JSON logging for production
- Performance metrics logging
- Error tracking and debugging

## 🚀 Deployment

### Development
```bash
streamlit run main.py
```

### Production Options
- **Streamlit Cloud**: Easy deployment with GitHub integration
- **Docker**: Containerized deployment with Redis caching
- **AWS EC2**: Scalable cloud deployment
- **Heroku**: Simple platform-as-a-service deployment

See [Deployment Guide](docs/deployment.md) for detailed instructions.

## 🔧 Configuration

### Environment Variables
```bash
# Core Settings
ENVIRONMENT=development|staging|production
DEBUG=true|false
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_SEARCH_K=4

# Caching
CACHE_ENABLED=true
CACHE_TYPE=memory|redis
CACHE_TTL=3600

# Performance
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
```

## 📁 Project Structure

```
rag_transformer_youtubetranscript_genai/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── exceptions.py              # Custom exception classes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── youtube_service.py         # YouTube transcript handling
│   │   ├── pdf_service.py             # PDF document processing
│   │   ├── vector_store_service.py    # Vector database operations
│   │   └── rag_service.py             # Main RAG orchestration
│   ├── ui/
│   │   ├── __init__.py
│   │   └── components.py              # Streamlit UI components
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py          # Structured logging
│       ├── cache.py                   # Caching mechanisms
│       └── monitoring.py              # Performance monitoring
├── config/
│   ├── __init__.py
│   └── settings.py                    # Pydantic configuration
├── tests/
│   ├── conftest.py                    # Test configuration
│   └── test_*.py                      # Test modules
├── docs/
│   ├── architecture.md               # Architecture overview
│   └── deployment.md                 # Deployment guide
├── logs/                             # Application logs
├── data/                             # Data storage
│   └── vector_store/                 # Vector database files
├── main.py                           # Application entry point
├── requirements.txt                  # Dependencies
├── .env                             # Environment variables
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## 🎨 Customization

### Model Configuration
```python
# Change LLM model in config/settings.py
GROQ_MODEL=llama-3.1-8b-instant  # Faster alternative

# Change embedding model
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

### UI Customization
Modify `src/ui/components.py` to change:
- Color schemes and themes
- Layout and spacing
- Component behavior
- Custom styling

### Performance Tuning
Adjust settings in `config/settings.py`:
- Chunk size and overlap
- Similarity search parameters
- Cache TTL and strategies
- Concurrent request limits

## 🔑 API Keys Setup

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create account and generate API key
3. Add to `.env` file

### Groq API Key
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up and generate API key
3. Add to `.env` file

### LangChain (Optional)
1. Visit [LangSmith](https://smith.langchain.com/)
2. Create project and get API key
3. Add to `.env` file for debugging/tracing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## 📄 Documentation

- [Architecture Overview](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- API Documentation (auto-generated)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original "Attention Is All You Need" paper by Vaswani et al.
- Streamlit for the amazing web framework
- LangChain for RAG pipeline tools
- OpenAI and Groq for powerful AI models

---

**Built with ❤️ for the AI community**

*This application demonstrates production-ready RAG implementation with modern software engineering practices suitable for enterprise deployment and technical interviews.*
