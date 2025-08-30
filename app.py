import streamlit as st
import os
import time
from dotenv import load_dotenv
from typing import List, Dict, Optional
import traceback

# Langchain imports
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# YouTube transcript API
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Attention Mechanism RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .paper-summary {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .chat-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    .sidebar-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AttentionRAGApp:
    def __init__(self):
        self.initialize_environment()
        self.video_id = "bCz4OMemCcA"
        self.pdf_path = "Attention.pdf"
        self.vector_store = None
        self.retriever = None
        self.main_chain = None
        self.llm = None
        self.embeddings = None
        
        # Initialize models - this will set llm and embeddings
        self.models_initialized = self.setup_models()
        
    def initialize_environment(self):
        """Initialize environment variables for LangChain and APIs"""
        # Set up LangChain tracing if configured
        if os.getenv("LANGCHAIN_API_KEY") and os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    def setup_models(self):
        """Initialize the language model and embeddings"""
        try:
            # Use Groq for fast inference - Updated to use supported model
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",  # Updated to current supported model
                temperature=0.2,
                max_tokens=1000
            )
            
            # Use OpenAI embeddings for better semantic understanding
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up models: {str(e)}")
            # Set default values to prevent attribute errors
            self.llm = None
            self.embeddings = None
            return False
    
    def load_youtube_transcript(self) -> str:
        """Load and return YouTube transcript"""
        try:
            with st.spinner("Loading YouTube transcript..."):
                # Create API instance
                api = YouTubeTranscriptApi()
                
                # Get transcript list
                transcript_list = api.list(self.video_id)
                
                # Find English transcript
                transcript_data = None
                for transcript in transcript_list:
                    if transcript.language_code == 'en':
                        transcript_data = transcript.fetch()
                        break
                
                if not transcript_data:
                    # Try any available transcript
                    transcript_data = transcript_list[0].fetch()
                
                # Convert to plain text
                transcript = " ".join(chunk.text for chunk in transcript_data)
                return transcript
                
        except TranscriptsDisabled:
            st.error("No captions available for this video.")
            return ""
        except Exception as e:
            st.error(f"Error loading transcript: {str(e)}")
            return ""
    
    def load_pdf_content(self) -> str:
        """Load and return PDF content"""
        try:
            if not os.path.exists(self.pdf_path):
                st.error(f"PDF file '{self.pdf_path}' not found.")
                return ""
            
            with st.spinner("Loading PDF content..."):
                loader = PyPDFLoader(self.pdf_path)
                pages = loader.load()
                pdf_content = "\n\n".join([page.page_content for page in pages])
                return pdf_content
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return ""
    
    def create_vector_store(self, youtube_content: str, pdf_content: str):
        """Create FAISS vector store from combined content"""
        try:
            # Check if models are properly initialized
            if not self.models_initialized or self.embeddings is None:
                st.error("Models not properly initialized. Please check your API keys and try again.")
                return False
                
            with st.spinner("Creating vector embeddings..."):
                # Combine content with source labels
                combined_content = f"""
                SOURCE: YouTube Video Transcript
                {youtube_content}
                
                SOURCE: Attention Paper PDF
                {pdf_content}
                """
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", "! ", "? ", " "]
                )
                
                documents = text_splitter.create_documents([combined_content])
                
                # Create vector store
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                
                # Create retriever
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                
                return True
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def setup_rag_chain(self):
        """Setup the RAG chain for question answering"""
        try:
            # Check if models are properly initialized
            if not self.models_initialized or self.llm is None or self.retriever is None:
                st.error("Models or retriever not properly initialized. Please initialize the RAG system first.")
                return False
                
            # Define prompt template
            prompt = PromptTemplate(
                template="""
                You are an expert AI assistant specializing in the "Attention Is All You Need" paper and transformer architectures.
                
                Use the following context from both the research paper and video explanation to answer the question.
                Provide detailed, accurate, and insightful responses. If you're citing specific information from the video or paper, mention the source.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer: Provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available.
                """,
                input_variables=['context', 'question']
            )
            
            # Format documents function
            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            # Create the RAG chain
            parallel_chain = RunnableParallel({
                'context': self.retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })
            
            parser = StrOutputParser()
            self.main_chain = parallel_chain | prompt | self.llm | parser
            
            return True
        except Exception as e:
            st.error(f"Error setting up RAG chain: {str(e)}")
            return False
    
    def get_paper_summary(self) -> str:
        """Return a summary of the Attention paper"""
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

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🔍 Attention Mechanism RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about the "Attention Is All You Need" paper and transformer architecture</p>', unsafe_allow_html=True)
    
    # Initialize the app
    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = AttentionRAGApp()
        st.session_state.initialized = False
        st.session_state.chat_history = []
    
    app = st.session_state.rag_app
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 About This App")
        st.markdown("""
        <div class="sidebar-info">
        This RAG application combines information from:
        <br>• YouTube video explanation
        <br>• Original "Attention Is All You Need" paper
        <br><br>
        <strong>Powered by:</strong>
        <br>• Groq (Llama 3.1 70B) for fast responses
        <br>• OpenAI embeddings for semantic search
        <br>• FAISS vector database
        <br>• LangChain for RAG pipeline
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔧 System Status")
        
        # Initialize system
        if st.button("🚀 Initialize RAG System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                # Check if models are properly initialized
                if not app.models_initialized:
                    st.error("❌ Models failed to initialize. Please check your API keys and try again.")
                    st.error("Make sure you have set OPENAI_API_KEY and GROQ_API_KEY in your .env file.")
                    return
                
                # Load content
                youtube_content = app.load_youtube_transcript()
                pdf_content = app.load_pdf_content()
                
                if youtube_content and pdf_content:
                    # Create vector store
                    if app.create_vector_store(youtube_content, pdf_content):
                        # Setup RAG chain
                        if app.setup_rag_chain():
                            st.session_state.initialized = True
                            st.success("✅ RAG system initialized successfully!")
                        else:
                            st.error("❌ Failed to setup RAG chain")
                    else:
                        st.error("❌ Failed to create vector store")
                else:
                    st.error("❌ Failed to load content")
        
        # Show status
        if app.models_initialized:
            st.success("✅ Models Loaded")
        else:
            st.error("❌ Models Failed")
            st.error("Check API keys in .env file")
            
        if st.session_state.initialized:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ System Not Initialized")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Paper summary
        with st.expander("📄 About the Attention Paper", expanded=True):
            st.markdown(f'<div class="paper-summary">{app.get_paper_summary()}</div>', unsafe_allow_html=True)
        
        # Chat interface
        st.markdown("### 💬 Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Question input
        question = st.text_input("Enter your question about the Attention paper or transformer architecture:", key="question_input")
        
        col_ask, col_examples = st.columns([1, 1])
        
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary")
        
        with col_examples:
            if st.button("💡 Show Example Questions"):
                st.session_state.show_examples = not getattr(st.session_state, 'show_examples', False)
        
        # Example questions
        if getattr(st.session_state, 'show_examples', False):
            st.markdown("**Example Questions:**")
            examples = [
                "What is the self-attention mechanism and how does it work?",
                "How do transformers differ from RNNs and CNNs?",
                "What are the key components of the transformer architecture?",
                "How does multi-head attention improve the model?",
                "What is positional encoding and why is it needed?",
                "What were the main results and achievements of this paper?"
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    st.session_state.question_input = example
                    st.rerun()
        
        # Process question
        if ask_button and question and st.session_state.initialized:
            with st.spinner("Thinking... 🤔"):
                try:
                    # Get answer from RAG chain
                    answer = app.main_chain.invoke(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        
        elif ask_button and not st.session_state.initialized:
            st.warning("Please initialize the RAG system first using the sidebar.")
        
        elif ask_button and not question:
            st.warning("Please enter a question.")
    
    with col2:
        # Quick stats and info
        st.markdown("### 📊 Quick Info")
        
        info_data = {
            "📺 Video ID": "bCz4OMemCcA",
            "📄 PDF Source": "Attention.pdf",
            "🧠 LLM": "Llama 3.1 70B (Groq)",
            "🔍 Embeddings": "OpenAI text-embedding-3-small",
            "💾 Vector DB": "FAISS",
            "🔗 Framework": "LangChain"
        }
        
        for key, value in info_data.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("---")
        
        # Recent questions counter
        if st.session_state.chat_history:
            user_questions = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
            st.markdown(f"**📈 Questions Asked:** {len(user_questions)}")
        
        # Performance metrics placeholder
        if st.session_state.initialized:
            st.markdown("**🚀 System Status:** Ready")
            st.markdown("**📊 Vector Store:** Loaded")
        
if __name__ == "__main__":
    main()
