"""
Simple RAG Application for "Attention Is All You Need" Paper
A straightforward Retrieval-Augmented Generation app using YouTube transcripts and PDF documents.
"""

import streamlit as st
import os
import logging
from typing import Optional

# Import our simple modules
from rag_service import RAGService
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="🔍 Attention Mechanism RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    /* Make form submit button more prominent */
    .stForm > div > div > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stForm > div > div > button:hover {
        background-color: #45a049 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'rag_service' not in st.session_state:
        st.session_state.rag_service = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    if 'initialization_attempted' not in st.session_state:
        st.session_state.initialization_attempted = False


def initialize_rag_system():
    """Initialize RAG system on first load"""
    if not st.session_state.initialization_attempted:
        st.session_state.initialization_attempted = True
        
        try:
            with st.spinner("🔧 Initializing RAG system..."):
                config = get_config()
                logger.info("🚀 Creating RAG service with configuration")
                st.session_state.rag_service = RAGService(config)
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
                logger.info("✅ RAG system initialization completed")
        except Exception as e:
            error_msg = f"❌ Failed to initialize system: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)
            st.session_state.system_initialized = False


def get_rag_service() -> Optional[RAGService]:
    """Get RAG service instance"""
    return st.session_state.rag_service


def display_chat_history():
    """Display chat history"""
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        # User question
        st.markdown(f"""
        <div class="chat-message" style="background-color: #e3f2fd;">
            <strong>🤔 You:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
        
        # AI answer
        st.markdown(f"""
        <div class="chat-message" style="background-color: #f3e5f5;">
            <strong>🤖 Assistant:</strong> {answer}
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Attention Mechanism RAG Assistant</h1>
        <p>Ask questions about the "Attention Is All You Need" paper</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This RAG system combines:
        - 📹 YouTube video explanation 
        - 📄 Original research paper
        - 🧠 Groq Llama 3.3 70B model
        - 🔍 OpenAI embeddings + FAISS
        """)
        
        st.header("🎯 Example Questions")
        example_questions = [
            "What is the attention mechanism?",
            "How does multi-head attention work?",
            "What are the key innovations in the Transformer?",
            "Explain positional encoding",
            "How is self-attention computed?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state.current_question = question
        
        # System status
        st.header("🔧 System Status")
        if st.session_state.system_initialized:
            st.success("✅ Ready")
        else:
            st.warning("⚠️ Not initialized")
    
    # Main content area - Question Input Form
    st.header("💬 Ask Your Question")
    
    # Add clear instructions
    st.markdown("""
    ### 📝 How to use:
    1. **Type your question** in the text area below
    2. **Click the "SUBMIT QUESTION" button** to get your answer
    3. **View the response** in the conversation history
    """)
    
    # Create a more prominent form for better UX
    with st.form("question_form", clear_on_submit=False):
        # Question input with better layout
        question = st.text_area(
            "💭 Type your question about the 'Attention Is All You Need' paper:",
            height=120,
            placeholder="e.g., What is the attention mechanism? How does multi-head attention work?",
            value=st.session_state.get('current_question', ''),
            help="Ask anything about the Transformer architecture, attention mechanisms, or the research paper!"
        )
        
        # Make submit button more prominent
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("🚀 SUBMIT QUESTION", type="primary", use_container_width=True)
        
        # Add visual instruction
        if not submitted:
            st.info("👆 Click the SUBMIT QUESTION button above to get your answer!")
    
    # Clear the current question after it's displayed
    if 'current_question' in st.session_state:
        del st.session_state.current_question
    
    # Additional buttons outside the form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)
        
    with col2:
        # Add example button
        if st.button("💡 Show Example", use_container_width=True):
            st.session_state.current_question = "What is the attention mechanism?"
            st.rerun()
    
    # Handle form submission and button clicks
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submitted and question.strip():
        # Clear the instruction message
        st.empty()
        
        # Get RAG service
        rag_service = get_rag_service()
        
        if rag_service:
            try:
                # Show what question is being processed
                with st.container():
                    st.info(f"🤔 Processing: {question}")
                    
                with st.spinner("🤔 Thinking..."):
                    logger.info(f"🤔 Processing question: {question[:50]}...")
                    # Get answer from RAG service
                    answer = rag_service.ask_question(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    logger.info("✅ Question processed successfully")
                    
                    # Show success message
                    st.success("✅ Question answered successfully! Check the conversation history below.")
                    
                    # Rerun to update the display
                    st.rerun()
                    
            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
        else:
            st.error("❌ System not initialized. Please check your configuration.")
    
    elif submitted and not question.strip():
        st.warning("⚠️ Please enter a question before clicking Submit.")
    
    # Add helpful tips
    with st.expander("💡 Tips for Better Results"):
        st.write("""
        **How to get the best answers:**
        - Be specific about what you want to know
        - Ask about transformer architecture, attention mechanisms, or model details
        - Reference specific parts like "multi-head attention" or "positional encoding"
        - Ask for explanations, comparisons, or examples
        
        **Example questions:**
        - "How does self-attention work in transformers?"
        - "What are the advantages of multi-head attention?"
        - "Explain the encoder-decoder architecture"
        - "What is positional encoding and why is it needed?"
        """)
    
    # Quick question buttons
    st.subheader("🎯 Quick Questions")
    quick_questions = [
        "What is the attention mechanism?",
        "How does multi-head attention work?", 
        "What are the key innovations in the Transformer?",
        "Explain positional encoding"
    ]
    
    cols = st.columns(2)
    for i, quick_q in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(f"❓ {quick_q}", key=f"quick_{i}", use_container_width=True):
                st.session_state.current_question = quick_q
                st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💬 Conversation History")
        display_chat_history()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with Streamlit 🚀 | Powered by Groq ⚡ | Enhanced with OpenAI 🧠</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
