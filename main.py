"""
Modern RAG Application - Industry-standard implementation
A sophisticated Retrieval-Augmented Generation application for the "Attention Is All You Need" paper.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.services import RAGService
from src.ui import UIComponents, ThemeManager
from src.utils import get_logger, setup_logging
from src.core.exceptions import RAGApplicationError, get_user_friendly_message
from config.settings import settings

# Setup logging
logger = setup_logging()
app_logger = get_logger("app")


class RAGApplication:
    """Main RAG Application class with modern architecture"""
    
    def __init__(self):
        self.rag_service = None
        self._initialize_session_state()
        self._configure_page()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'rag_service' not in st.session_state:
            st.session_state.rag_service = None
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
        
        if 'question_input' not in st.session_state:
            st.session_state.question_input = ""
    
    def _configure_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=settings.app.page_title,
            page_icon=settings.app.page_icon,
            layout=settings.app.layout,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/issues',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': "RAG Assistant for 'Attention Is All You Need' paper"
            }
        )
        
        # Apply custom styling
        ThemeManager.apply_custom_css()
    
    def _get_rag_service(self) -> RAGService:
        """Get or create RAG service instance"""
        if st.session_state.rag_service is None:
            try:
                app_logger.info("Creating new RAG service instance")
                st.session_state.rag_service = RAGService()
            except Exception as e:
                app_logger.error(f"Failed to create RAG service: {e}")
                st.error(f"Failed to initialize application: {get_user_friendly_message('SYSTEM_UNAVAILABLE')}")
                st.stop()
        
        return st.session_state.rag_service
    
    def _handle_question_submission(self, question: str):
        """Handle question submission and generate response"""
        if not question or not question.strip():
            st.warning("Please enter a question.")
            return
        
        try:
            rag_service = self._get_rag_service()
            
            # Check if system is initialized
            if not rag_service.get_system_status().get("initialized", False):
                st.warning("Please initialize the RAG system first using the sidebar.")
                return
            
            app_logger.info(f"Processing user question: {question[:100]}...")
            
            with st.spinner("🤔 Thinking..."):
                # Get answer from RAG service
                answer = rag_service.query(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": question,
                    "timestamp": st.experimental_get_query_params().get("timestamp", [""])[0]
                })
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "timestamp": st.experimental_get_query_params().get("timestamp", [""])[0]
                })
                
                app_logger.info("Question processed successfully")
                
                # Clear input and rerun to show response
                st.session_state.question_input = ""
                st.rerun()
                
        except RAGApplicationError as e:
            app_logger.error(f"RAG application error: {e}")
            error_message = get_user_friendly_message(e.error_code) if e.error_code else str(e)
            st.error(f"⚠️ {error_message}")
            
        except Exception as e:
            app_logger.error(f"Unexpected error processing question: {e}")
            st.error("❌ An unexpected error occurred. Please try again.")
    
    def run(self):
        """Main application entry point"""
        try:
            app_logger.info("Starting RAG application")
            
            # Get RAG service
            rag_service = self._get_rag_service()
            
            # Render header
            UIComponents.render_header()
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Paper summary
                UIComponents.render_paper_summary()
                
                # System initialization
                is_initialized = UIComponents.render_initialization_section(rag_service)
                st.session_state.initialized = is_initialized
                
                # Chat interface (only if initialized)
                if is_initialized:
                    question = UIComponents.render_chat_interface(rag_service)
                    
                    if question:
                        self._handle_question_submission(question)
                    
                    # Show chat history
                    UIComponents.render_chat_history()
                else:
                    st.info("👆 Please initialize the system above to start asking questions.")
            
            with col2:
                # Sidebar content
                UIComponents.render_sidebar_info(rag_service)
            
            # Footer
            UIComponents.render_footer()
            
        except Exception as e:
            app_logger.error(f"Critical application error: {e}")
            st.error("🚨 Critical application error. Please refresh the page.")
            
            if settings.is_development():
                st.exception(e)


def main():
    """Application entry point"""
    try:
        # Initialize and run application
        app = RAGApplication()
        app.run()
        
    except Exception as e:
        st.error("🚨 Failed to start application. Please check the logs.")
        if settings.is_development():
            st.exception(e)


if __name__ == "__main__":
    main()
