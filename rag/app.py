"""Streamlit UI for Penile SCC Learning Tool."""

import streamlit as st
from pathlib import Path
from src.rag import PenileSCCRAG
from src.safety import check_safety, get_safe_redirect
from src.db import get_collection

# Initialize ingest on first load
@st.cache_resource
def initialize_rag():
    """Initialize RAG system on app startup."""
    # Check if data exists, if not ingest
    collection = get_collection()
    if collection is None:
        from src.ingest import ingest_sources
        ingest_sources()
    
    # Initialize RAG
    return PenileSCCRAG()

# Page config
st.set_page_config(
    page_title="Penile SCC Learning Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📚 Penile SCC Learning Tool")

# Disclaimer
st.warning(
    "⚠️ **Educational Only.** This tool provides educational information about penile cancer. "
    "It is **NOT** medical advice. Always consult with your healthcare team for personal decisions."
)

# Sidebar
st.sidebar.header("⚙️ Settings")
audience = st.sidebar.radio(
    "Who are you?",
    options=["family", "clinician"],
    format_func=lambda x: "Patient/Family" if x == "family" else "Clinician",
    help="Affects language complexity and focus"
)

depth = st.sidebar.radio(
    "How much detail?",
    options=["quick", "deep"],
    format_func=lambda x: "Quick Answer" if x == "quick" else "In-Depth",
    help="Quick = 1-2 sentences. Deep = detailed explanation."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
### About
This is an AI-powered learning tool that retrieves information from trusted sources
about penile cancer staging, symptoms, treatment, and more.

Every answer includes citations to sources.
    """
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG with auto-ingest fallback
try:
    rag = initialize_rag()
    rag_ready = True
    rag_error = None
except Exception as e:
    rag_ready = False
    rag_error = str(e)

# Check if RAG is ready
if not rag_ready:
    st.error(
        f"**RAG System Error**\n\n{rag_error}\n\n"
        "Please check your setup. Run: `python -m src.ingest` manually if needed."
    )
else:
    # Display chat history
    st.subheader("Chat History")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    # Input
    st.subheader("Ask a Question")
    user_input = st.text_input(
        "What would you like to know about penile cancer?",
        placeholder="e.g., What are the symptoms? How is penile cancer staged?",
        key="user_input"
    )
    
    if user_input:
        # Safety check
        is_safe, reason = check_safety(user_input)
        
        if not is_safe:
            st.error(f"**Cannot answer:** {reason}")
            st.info(f"**Suggestion:** {get_safe_redirect(user_input)}")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate answer
            with st.spinner("Retrieving information and generating answer..."):
                try:
                    result = rag.answer_question(
                        query=user_input,
                        audience=audience,
                        depth=depth
                    )
                    answer = result["answer"]
                except Exception as e:
                    answer = f"**Error generating answer:** {str(e)}"
            
            # Display answer
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            
            with st.chat_message("assistant"):
                st.write(answer)
            
            # Show sources
            with st.expander("📖 Retrieved Sources (Evidence)"):
                for i, chunk in enumerate(result["chunks"], 1):
                    st.markdown(f"**[{i}] {chunk['metadata']['source']}**")
                    st.markdown(f"*Chunk {chunk['metadata']['chunk_index']}*")
                    st.text(chunk["text"][:300] + "...")
            
            # Clear input
            st.rerun()
