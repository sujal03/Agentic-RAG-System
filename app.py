"""Streamlit chat interface for the AI pipeline."""
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from src.pipeline.graph import AIPipeline
from src.agents.rag_agent import RAGAgent


# Page configuration
st.set_page_config(
    page_title="AI Pipeline Chat",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
    }
    
    .source-tag {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #e8f5e9;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    
    .agent-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .weather-badge {
        background-color: #fff3e0;
        color: #e65100;
    }
    
    .rag-badge {
        background-color: #e8eaf6;
        color: #3949ab;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = RAGAgent()
    
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = AIPipeline(rag_agent=st.session_state.rag_agent)
    
    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = 0
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []


def display_sidebar():
    """Display the sidebar with PDF upload and info."""
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF to ask questions about its content"
        )
        
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Load and index the PDF
                        chunks = st.session_state.rag_agent.load_pdf_from_bytes(
                            uploaded_file,
                            uploaded_file.name
                        )
                        num_indexed = st.session_state.rag_agent.index_documents(chunks)
                        
                        st.session_state.documents_indexed += num_indexed
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        
                        st.success(f"✅ Indexed {num_indexed} chunks from {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
        
        # Show indexed documents info
        if st.session_state.uploaded_files:
            st.divider()
            st.subheader("📚 Indexed Documents")
            for filename in st.session_state.uploaded_files:
                st.write(f"• {filename}")
            st.info(f"Total chunks: {st.session_state.documents_indexed}")
        
        st.divider()
        
        # Help section
        st.header("💡 How to Use")
        st.markdown("""
        **Weather Queries:**
        - "What's the weather in Indore?"
        - "Is it raining in Delhi?"
        - "Temperature in Mumbai tomorrow"
        
        **Document Queries:**
        - Upload a PDF first
        - "Summarize the document"
        - "What does it say about X?"
        """)
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def display_message(message: dict):
    """Display a chat message.
    
    Args:
        message: Message dictionary with role, content, and metadata
    """
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Show metadata for assistant messages
        if role == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                agent = metadata.get("agent_used", "")
                if agent == "weather":
                    st.markdown('<span class="agent-badge weather-badge">🌤️ Weather</span>', unsafe_allow_html=True)
                elif agent == "rag":
                    st.markdown('<span class="agent-badge rag-badge">📄 RAG</span>', unsafe_allow_html=True)
            
            with col2:
                sources = metadata.get("sources", [])
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.title("🤖 AI Pipeline Chat")
    st.caption("Ask about weather or your PDF documents")
    
    # Sidebar
    display_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the pipeline
                    result = st.session_state.pipeline.invoke(prompt)
                    
                    response = result.get("response", "I couldn't generate a response.")
                    
                    st.markdown(response)
                    
                    # Show agent badge
                    agent = result.get("agent_used", "")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if agent == "weather":
                            st.markdown('<span class="agent-badge weather-badge">🌤️ Weather</span>', unsafe_allow_html=True)
                        elif agent == "rag":
                            st.markdown('<span class="agent-badge rag-badge">📄 RAG</span>', unsafe_allow_html=True)
                    
                    with col2:
                        sources = result.get("sources", [])
                        if sources:
                            st.caption(f"Sources: {', '.join(sources)}")
                    
                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metadata": {
                            "agent_used": agent,
                            "sources": sources,
                            "query_type": result.get("query_type", ""),
                            "success": result.get("success", False)
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {"agent_used": "error"}
                    })


if __name__ == "__main__":
    main()
