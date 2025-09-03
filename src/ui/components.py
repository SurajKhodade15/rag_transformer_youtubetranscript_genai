"""
Modern Streamlit UI components for the RAG application.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime


class UIComponents:
    """Reusable UI components for the application"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="color: #1f77b4; font-size: 3rem; font-weight: bold; margin-bottom: 0.5rem;">
                🔍 Attention Mechanism RAG Assistant
            </h1>
            <p style="color: #666; font-size: 1.2rem; margin-bottom: 2rem;">
                Ask questions about the "Attention Is All You Need" paper and transformer architecture
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_paper_summary():
        """Render paper summary in an expandable section"""
        with st.expander("📄 About the \"Attention Is All You Need\" Paper", expanded=False):
            st.markdown("""
            **"Attention Is All You Need" - Transformer Architecture**
            
            This groundbreaking 2017 paper by Vaswani et al. introduced the Transformer architecture, revolutionizing natural language processing and deep learning.

            **Key Innovations:**
            
            🔹 **Self-Attention Mechanism**: Allows the model to weigh the importance of different words in a sequence when processing each word
            
            🔹 **Parallel Processing**: Unlike RNNs, Transformers can process all positions simultaneously, dramatically improving training speed
            
            🔹 **Multi-Head Attention**: Uses multiple attention heads to capture different types of relationships between words
            
            🔹 **Positional Encoding**: Injects sequence order information since the architecture doesn't inherently understand position
            
            🔹 **Encoder-Decoder Structure**: Six encoder and decoder layers with residual connections and layer normalization
            
            **Impact**: The Transformer architecture became the foundation for modern language models like BERT, GPT, T5, and many others, fundamentally changing how we approach NLP tasks.
            """)
    
    @staticmethod
    def render_initialization_section(rag_service) -> bool:
        """
        Render system initialization section
        
        Returns:
            True if system is initialized, False otherwise
        """
        st.markdown("### 🚀 System Initialization")
        
        # Check current status
        status = rag_service.get_system_status()
        
        if status.get("initialized", False):
            st.success("✅ RAG System is ready!")
            
            # Show system stats
            with st.expander("📊 System Information", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("LLM Status", "Ready" if status.get("llm_initialized") else "Not Ready")
                    st.metric("Chain Status", "Ready" if status.get("chain_ready") else "Not Ready")
                
                with col2:
                    vector_stats = status.get("vector_store_stats", {})
                    if vector_stats.get("initialized"):
                        st.metric("Total Vectors", vector_stats.get("total_vectors", 0))
                        st.metric("Vector Dimension", vector_stats.get("vector_dimension", 0))
            
            # Reinitialize button
            if st.button("🔄 Reinitialize System", type="secondary"):
                return UIComponents._handle_initialization(rag_service, reinitialize=True)
            
            return True
        else:
            st.warning("⚠️ RAG System needs to be initialized")
            st.info("Click the button below to load and process the documents")
            
            if st.button("🚀 Initialize RAG System", type="primary", use_container_width=True):
                return UIComponents._handle_initialization(rag_service)
            
            return False
    
    @staticmethod
    def _handle_initialization(rag_service, reinitialize: bool = False) -> bool:
        """Handle system initialization with progress indicators"""
        try:
            with st.spinner("Initializing RAG system..." if not reinitialize else "Reinitializing system..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Progress updates
                status_text.text("🔄 Loading YouTube transcript...")
                progress_bar.progress(25)
                
                status_text.text("📄 Processing PDF document...")
                progress_bar.progress(50)
                
                status_text.text("🧮 Creating embeddings and vector store...")
                progress_bar.progress(75)
                
                # Initialize system
                if reinitialize:
                    result = rag_service.reinitialize()
                else:
                    result = rag_service.initialize_system()
                
                progress_bar.progress(100)
                status_text.text("✅ Initialization complete!")
                
                # Show success message with metrics
                if result.get("success"):
                    st.success(f"System initialized successfully in {result['execution_time']:.2f} seconds!")
                    
                    # Show source loading status
                    sources = result.get("sources", {})
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        youtube_status = "✅ Loaded" if sources.get("youtube", {}).get("loaded") else "❌ Failed"
                        st.write(f"**YouTube:** {youtube_status}")
                        if sources.get("youtube", {}).get("content_length"):
                            st.write(f"📝 {sources['youtube']['content_length']:,} characters")
                    
                    with col2:
                        pdf_status = "✅ Loaded" if sources.get("pdf", {}).get("loaded") else "❌ Failed"
                        st.write(f"**PDF:** {pdf_status}")
                        if sources.get("pdf", {}).get("content_length"):
                            st.write(f"📝 {sources['pdf']['content_length']:,} characters")
                    
                    return True
                else:
                    st.error("Initialization failed!")
                    return False
                    
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return False
    
    @staticmethod
    def render_chat_interface(rag_service) -> Optional[str]:
        """
        Render chat interface for questions and answers
        
        Returns:
            User question if submitted, None otherwise
        """
        st.markdown("### 💬 Ask Questions")
        
        # Question input
        question = st.text_area(
            "Enter your question about the Attention paper:",
            placeholder="e.g., What is the self-attention mechanism and how does it work?",
            height=100,
            key="question_input"
        )
        
        # Submit button
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.button("🤔 Ask Question", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.question_input = ""
            st.session_state.chat_history = []
            st.rerun()
        
        # Example questions
        UIComponents.render_example_questions()
        
        return question if submit_button and question else None
    
    @staticmethod
    def render_example_questions():
        """Render example questions"""
        with st.expander("💡 Example Questions", expanded=False):
            examples = [
                "What is the self-attention mechanism and how does it work?",
                "How do transformers differ from RNNs and CNNs?", 
                "What are the key components of the transformer architecture?",
                "How does multi-head attention improve the model?",
                "What is positional encoding and why is it needed?",
                "What were the main results and achievements of this paper?"
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.question_input = example
                    st.rerun()
    
    @staticmethod
    def render_chat_history():
        """Render chat history with modern styling"""
        if not st.session_state.get("chat_history"):
            return
        
        st.markdown("### 📚 Chat History")
        
        # Reverse to show newest first
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #2196f3;">
                    <strong>🧑 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #9c27b0;">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_info(rag_service):
        """Render sidebar with system information"""
        st.sidebar.markdown("### 📚 About This App")
        st.sidebar.info("""
        This RAG application combines information from:
        - 📺 YouTube video explanations
        - 📄 Original research paper
        - 🧠 Powered by Groq's Llama 3.3 70B
        - 🔍 OpenAI embeddings for search
        """)
        
        # System status
        st.sidebar.markdown("### 🔧 System Status")
        status = rag_service.get_system_status()
        
        if status.get("initialized"):
            st.sidebar.success("✅ System Ready")
            
            # Performance metrics
            vector_stats = status.get("vector_store_stats", {})
            if vector_stats.get("initialized"):
                st.sidebar.metric("Documents", vector_stats.get("total_vectors", 0))
                st.sidebar.metric("Dimensions", vector_stats.get("vector_dimension", 0))
        else:
            st.sidebar.warning("⚠️ System Not Ready")
        
        # Quick info
        st.sidebar.markdown("### 📊 Technical Details")
        info_data = {
            "🧠 LLM": "Llama 3.3 70B (Groq)",
            "🔍 Embeddings": "OpenAI text-embedding-3-small", 
            "💾 Vector DB": "FAISS",
            "🔗 Framework": "LangChain"
        }
        
        for key, value in info_data.items():
            st.sidebar.text(f"{key}: {value}")
        
        # Chat statistics
        if st.session_state.get("chat_history"):
            user_questions = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
            st.sidebar.metric("Questions Asked", len(user_questions))
        
        # Action buttons
        st.sidebar.markdown("### ⚙️ Actions")
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.sidebar.button("🔄 Clear Cache"):
            try:
                rag_service.clear_cache()
                st.sidebar.success("Cache cleared!")
            except Exception as e:
                st.sidebar.error(f"Failed to clear cache: {e}")
    
    @staticmethod
    def render_footer():
        """Render application footer"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            Built with ❤️ using Streamlit, LangChain, and Groq | 
            <a href="https://github.com" target="_blank">View Source</a>
        </div>
        """, unsafe_allow_html=True)


class ThemeManager:
    """Manage application theming and styling"""
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styling to the application"""
        st.markdown("""
        <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            color: #1f77b4;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid;
        }
        
        .chat-message.user {
            background-color: #e3f2fd;
            border-left-color: #2196f3;
        }
        
        .chat-message.assistant {
            background-color: #f3e5f5;
            border-left-color: #9c27b0;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            padding: 1rem;
        }
        
        /* Metric cards */
        [data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        /* Progress bar styling */
        .stProgress .st-bo {
            background-color: #e3f2fd;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #1f77b4;
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e9ecef;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #1f77b4;
            box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25);
        }
        
        /* Success/Error message styling */
        .stAlert {
            border-radius: 8px;
            border: none;
        }
        
        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
