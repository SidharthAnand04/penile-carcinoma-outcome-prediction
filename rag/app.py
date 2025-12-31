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


# Initialize RAG with auto-ingest fallback and auto-recovery
def try_initialize_rag_with_recovery():
    try:
        rag = initialize_rag()
        return rag, True, None
    except Exception as e:
        rag_error = str(e)
        # If error is due to missing collection or secrets parsing, try to auto-ingest and re-init
        if (
            "ChromaDB collection not found" in rag_error
            or "Error parsing secrets file" in rag_error
            or "OPENROUTER_API_KEY not set" in rag_error
            or "Invalid date or number" in rag_error
        ):
            try:
                from src.ingest import ingest_sources
                ingest_sources()
                rag = initialize_rag()
                return rag, True, None
            except Exception as e2:
                return None, False, f"{rag_error}\nAuto-ingest failed: {e2}"
        return None, False, rag_error

rag, rag_ready, rag_error = try_initialize_rag_with_recovery()

# Check if RAG is ready
if not rag_ready:
    st.error(
        f"**RAG System Error**\n\n{rag_error}\n\n"
        "Please check your setup. Run: `python -m src.ingest` manually if needed."
    )
else:
    # Display chat history
    st.subheader("Chat History")
    chat_anchor = st.empty()
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                # Show sources for this assistant message if present
                if "sources" in msg and msg["sources"]:
                    with st.expander("📖 Retrieved Sources (Evidence)", expanded=False):
                        for i, chunk in enumerate(msg["sources"], 1):
                            st.markdown(f"**[{i}] {chunk['metadata']['source']}**")
                            st.markdown(f"*Chunk {chunk['metadata']['chunk_index']}*")
                            st.text(chunk["text"][:300] + "...")
        # Place an anchor at the end of the last message for autoscroll
        if idx == len(st.session_state.messages) - 1:
            chat_anchor.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)


    # Input with submit button to avoid infinite loop
    st.subheader("Ask a Question")
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""
    with st.form(key="user_input_form", clear_on_submit=True):
        user_input = st.text_input(
            "What would you like to know about penile cancer?",
            value=st.session_state.input_value,
            placeholder="e.g., What are the symptoms? How is penile cancer staged?",
            key="user_input_box"
        )
        submitted = st.form_submit_button("Send")


    if submitted and user_input:
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
                    sources = result["chunks"]
                except Exception as e:
                    answer = f"**Error generating answer:** {str(e)}"
                    sources = []

            # Display answer and store sources with message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            # Clear input value in session state
            st.session_state.input_value = ""
            st.rerun()

    # (No global sources box; now shown per message)

    # Autoscroll to bottom using anchor and JS
    st.markdown("""
        <script>
        var chatBottom = document.getElementById('chat-bottom');
        if (chatBottom) { chatBottom.scrollIntoView({behavior: 'smooth'}); }
        </script>
    """, unsafe_allow_html=True)
