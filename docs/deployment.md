# Deployment Guide

## 🚀 Deployment Options

This guide covers various deployment strategies for the RAG application, from local development to production environments.

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for production)
- 2GB+ available disk space
- Internet connection for API access

### Required API Keys
- **OpenAI API Key**: For embeddings generation
- **Groq API Key**: For LLM inference
- **LangChain API Key**: (Optional) For tracing and debugging

## 🏠 Local Development Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd rag_transformer_youtubetranscript_genai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# LangChain (Optional - for debugging)
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_TRACING_V2=true

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_SEARCH_K=4

# Caching
CACHE_ENABLED=true
CACHE_TYPE=memory
CACHE_TTL=3600
```

### 3. Run the Application

```bash
# Using Streamlit directly
streamlit run main.py

# Or using Python
python main.py
```

The application will be available at `http://localhost:8501`

## ☁️ Cloud Deployment

### Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Select the main branch
   - Set `main.py` as the main file
   - Add environment variables in the dashboard

3. **Environment Variables Setup**:
   - Go to App Settings → Secrets
   - Add your environment variables:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key"
   GROQ_API_KEY = "your_groq_api_key"
   ENVIRONMENT = "production"
   DEBUG = false
   LOG_LEVEL = "INFO"
   ```

### Heroku Deployment

1. **Create Heroku App**:
   ```bash
   # Install Heroku CLI first
   heroku create your-app-name
   ```

2. **Add Buildpack**:
   ```bash
   heroku buildpacks:set heroku/python
   ```

3. **Create Procfile**:
   ```bash
   echo "web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   ```

4. **Set Environment Variables**:
   ```bash
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set GROQ_API_KEY=your_key
   heroku config:set ENVIRONMENT=production
   ```

5. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       software-properties-common \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Copy application code
   COPY . .

   # Expose port
   EXPOSE 8501

   # Health check
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

   # Run the application
   ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Create docker-compose.yml** (with Redis cache):
   ```yaml
   version: '3.8'
   
   services:
     app:
       build: .
       ports:
         - "8501:8501"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - GROQ_API_KEY=${GROQ_API_KEY}
         - ENVIRONMENT=production
         - CACHE_TYPE=redis
         - REDIS_HOST=redis
       depends_on:
         - redis
       volumes:
         - ./logs:/app/logs
         - ./data:/app/data

     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis_data:/data

   volumes:
     redis_data:
   ```

3. **Build and Run**:
   ```bash
   # Build image
   docker build -t rag-app .

   # Run with docker-compose
   docker-compose up -d
   ```

### AWS EC2 Deployment

1. **Launch EC2 Instance**:
   - Choose Ubuntu 22.04 LTS
   - t3.medium or larger recommended
   - Configure security group to allow port 8501

2. **Setup Instance**:
   ```bash
   # SSH into instance
   ssh -i your-key.pem ubuntu@your-instance-ip

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python and tools
   sudo apt install python3 python3-pip python3-venv git -y

   # Clone repository
   git clone <your-repo-url>
   cd rag_transformer_youtubetranscript_genai

   # Setup virtual environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Create environment file
   nano .env
   # Add your configuration
   ```

3. **Setup as Service**:
   ```bash
   # Create systemd service file
   sudo nano /etc/systemd/system/rag-app.service
   ```

   ```ini
   [Unit]
   Description=RAG Application
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/rag_transformer_youtubetranscript_genai
   Environment=PATH=/home/ubuntu/rag_transformer_youtubetranscript_genai/venv/bin
   ExecStart=/home/ubuntu/rag_transformer_youtubetranscript_genai/venv/bin/streamlit run main.py --server.port=8501 --server.address=0.0.0.0
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   # Enable and start service
   sudo systemctl enable rag-app
   sudo systemctl start rag-app
   sudo systemctl status rag-app
   ```

## 🔧 Production Configuration

### Environment Variables for Production

```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Performance
MAX_CONCURRENT_REQUESTS=20
REQUEST_TIMEOUT=60

# Caching (Redis recommended for production)
CACHE_ENABLED=true
CACHE_TYPE=redis
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langchain-key
```

### Nginx Reverse Proxy (Optional)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📊 Monitoring and Logging

### Log Configuration

```bash
# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Structured logging for production
LOG_FORMAT=json  # or 'standard' for development
```

### Health Checks

The application provides health check endpoints:
- System health: Built-in monitoring
- API connectivity: Automatic checks
- Resource usage: CPU and memory monitoring

### Performance Monitoring

```python
# Access monitoring dashboard data
from src.utils.monitoring import app_monitor

dashboard_data = app_monitor.get_dashboard_data()
```

## 🔒 Security Considerations

### API Key Management
- Use environment variables, never commit keys
- Rotate keys regularly
- Use least-privilege principles

### Network Security
- Use HTTPS in production
- Implement rate limiting
- Configure proper CORS settings

### Resource Limits
```bash
# Memory limits
MAX_MEMORY_USAGE=4GB

# Request limits
MAX_REQUEST_SIZE=10MB
REQUEST_TIMEOUT=30s
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
   ```

2. **API Key Issues**:
   ```bash
   # Verify keys are loaded
   python -c "from config.settings import settings; print(settings.api.openai_api_key[:10])"
   ```

3. **Memory Issues**:
   ```bash
   # Monitor memory usage
   pip install psutil
   python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
   ```

4. **Port Already in Use**:
   ```bash
   # Change port
   streamlit run main.py --server.port=8502
   ```

### Log Analysis

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log

# Performance analysis
grep "execution_time" logs/app.log | tail -20
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use Redis for shared caching
- Load balancer configuration
- Session affinity considerations

### Vertical Scaling
- Memory optimization for vector stores
- CPU considerations for embeddings
- Storage requirements for logs and data

### Cost Optimization
- API usage monitoring
- Cache hit ratio optimization
- Resource utilization analysis

This deployment guide provides comprehensive instructions for various environments. Choose the deployment method that best fits your infrastructure and requirements.
