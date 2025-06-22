import os
import tempfile

from langchain_core.messages import HumanMessage
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# local
from utils import (
    load_pdf,
    create_embeddings,
    split_documents,
    build_vector_store,
    process_query,
    process_chunks_with_rate_limit_cohere,
    add_chunks_to_vector_store_hf_embeddings,
    reset_memory,
)

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set custom CSS styling
if os.path.exists("src/style.css"):
    with open("src/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main title
st.title("📄 RAG PDF Assistant")
st.markdown("Upload a PDF and ask intelligent questions about it.")

# Initialize session state
session_defaults = {
    "messages": [],
    "vector_store": None,
    "uploaded_file": None,
    "rendered_pages": None,
    "show_reasoning": False,
    "rag_agent_executer": None,
    "docs": [],
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Sidebar for settings
with st.sidebar:
    st.subheader("📄 1. Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and (
        st.session_state.uploaded_file is None
        or uploaded_file.name != st.session_state.uploaded_file.name
    ):
        with st.spinner("Processing PDF..."):
            st.session_state.uploaded_file = uploaded_file

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            docs = load_pdf(pdf_path)
            st.session_state.docs = docs

            with st.spinner("Splitting PDF and generating embeddings..."):
                embeddings = create_embeddings("huggingface")
                chunks = split_documents(docs)
                vector_store = build_vector_store(embeddings)
                vector_store = add_chunks_to_vector_store_hf_embeddings(
                    chunks, vector_store
                )
                st.session_state.vector_store = vector_store

            try:
                os.unlink(pdf_path)
            except PermissionError:
                pass

    st.subheader("📥 2. Download Conversation")
    if st.session_state.messages:
        chat_history = "\n\n".join(
            f"You: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
            for m in st.session_state.messages
        )
        st.download_button("Download Chat", chat_history, file_name="chat.txt")

    st.subheader("🗑️ 3. Clear Chat")
    if st.button("Clear Conversation"):
        if len(st.session_state.messages):
            st.session_state.messages = []
            st.success("✅ Conversation cleared!")
            reset_memory()
        else:
            st.info("No messages to clear.")

# Layout
query = st.chat_input("Ask a question about your uploaded PDF...")

# Chat Area
chat_container = st.container(height=600, border=False)

if st.session_state.vector_store:
    with chat_container:
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        if query:
            with st.chat_message("user"):
                st.markdown(query)
                st.session_state.messages.append(HumanMessage(content=query))
                process_query(query, "", None)

# PDF Viewer
if uploaded_file:
    with st.expander("📑 Preview PDF", expanded=True):
        binary_data = uploaded_file.getvalue()
        pdf_viewer(
            input=binary_data,
            height=600,
            pages_to_render=[*range(min(len(st.session_state.docs), 30))],
            resolution_boost=2,
            pages_vertical_spacing=1,
            render_text=True,
        )
