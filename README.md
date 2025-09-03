# 🔍 Simple RAG Assistant for "Attention Is All You Need"

A clean, straightforward RAG (Retrieval-Augmented Generation) application that answers questions about the groundbreaking "Attention Is All You Need" paper using both YouTube video explanations and the original research document.

## 🌟 Features

- **📹 YouTube Integration**: Uses video transcript for detailed explanations
- **📄 PDF Processing**: Extracts content from the original research paper  
- **🤖 Smart Q&A**: Powered by Groq's Llama 3.3 70B for fast responses
- **🔍 Semantic Search**: OpenAI embeddings + FAISS for accurate retrieval
- **🎨 Clean UI**: Simple, responsive Streamlit interface

## 📁 Project Structure

```
📦 rag-assistant/
├── 📄 app.py              # Main Streamlit application
├── 📄 rag_service.py      # Core RAG functionality
├── 📄 config.py           # Configuration management
├── 📄 utils.py            # Helper functions
├── 📄 requirements.txt    # Dependencies
├── 📄 .env               # Environment variables
├── 📄 Attention.pdf      # Research paper
├── 📄 README.md          # This file
└── 📁 data/              # Vector store cache
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd rag-assistant
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
YOUTUBE_VIDEO_ID=bCz4OMemCcA
```

### 3. Run the Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 🔧 Configuration

All configuration is handled through environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | **Required** |
| `GROQ_API_KEY` | Groq API key for LLM | **Required** |
| `YOUTUBE_VIDEO_ID` | YouTube video ID | `bCz4OMemCcA` |
| `PDF_PATH` | Path to PDF file | `Attention.pdf` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `SIMILARITY_K` | Number of similar docs | `4` |

## 💡 How It Works

1. **Document Processing**: Extracts text from YouTube transcript and PDF
2. **Chunking**: Splits documents into manageable pieces
3. **Embeddings**: Creates vector representations using OpenAI
4. **Vector Store**: Stores embeddings in FAISS for fast search
5. **Retrieval**: Finds relevant chunks for user questions
6. **Generation**: Uses Groq LLM to generate answers

## 🎯 Example Questions

- "What is the attention mechanism?"
- "How does multi-head attention work?"
- "What are the key innovations in the Transformer?"
- "Explain positional encoding"
- "How is self-attention computed?"

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq Llama 3.3 70B
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: FAISS
- **Framework**: LangChain
- **PDF**: PyPDF2
- **Video**: YouTube Transcript API

## 📚 Dependencies

Core dependencies:
- `streamlit` - Web interface
- `langchain` - RAG framework
- `langchain-groq` - Groq integration
- `langchain-openai` - OpenAI integration
- `faiss-cpu` - Vector database
- `youtube-transcript-api` - YouTube transcripts
- `PyPDF2` - PDF processing

## 🔍 File Overview

### `app.py`
Main Streamlit application with UI components, chat interface, and user interaction logic.

### `rag_service.py`
Core RAG functionality including:
- Document processing (YouTube + PDF)
- Vector store creation and management
- Question answering pipeline
- LLM and embedding initialization

### `config.py`
Simple configuration management using environment variables with validation and defaults.

### `utils.py`
Helper functions for logging, file operations, and text processing.

## 🚨 Troubleshooting

**API Key Issues**:
- Ensure `.env` file has valid API keys
- Check API key permissions and quotas

**Vector Store Issues**:
- Delete `data/vector_store` folder to rebuild
- Check PDF file exists and is readable

**Dependencies**:
- Use Python 3.8+ 
- Install all requirements: `pip install -r requirements.txt`

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built with ❤️ using Streamlit, LangChain, and modern AI tools**
