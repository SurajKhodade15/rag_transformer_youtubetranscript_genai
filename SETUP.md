# Environment Setup

This project requires several API keys to function properly. Follow these steps to set up your environment:

## 1. Copy the Environment Template
```bash
cp .env.example .env
```

## 2. Get Your API Keys

### OpenAI API Key
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy the key and replace `your_openai_api_key_here` in your `.env` file

### Groq API Key
1. Go to [Groq Console](https://console.groq.com/keys)
2. Create a new API key
3. Copy the key and replace `your_groq_api_key_here` in your `.env` file

### LangChain API Key (Optional - for tracing)
1. Go to [LangSmith](https://smith.langchain.com/)
2. Create an account and get your API key
3. Copy the key and replace `your_langchain_api_key_here` in your `.env` file

### Hugging Face Token (Optional)
1. Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens)
2. Create a new token
3. Copy the token and replace `your_huggingface_token_here` in your `.env` file

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Run the Application
```bash
streamlit run app.py
```

## Security Note
- Never commit your `.env` file to Git
- Keep your API keys secure and don't share them
- The `.env.backup` file (if present) contains your actual keys for local reference only
