import os
import streamlit as st
from dotenv import load_dotenv
from document_loader import extract_pages_from_uploaded_files
from rag_pipeline import RAGPipeline

load_dotenv()

MAX_FILE_SIZE_MB = 20
ALLOWED_FILE_TYPES = ["pdf"]

st.set_page_config(page_title="Industry Data Chatbot", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False
if "processed_signature" not in st.session_state:
    st.session_state.processed_signature = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

with st.sidebar:
    st.markdown("### INDUSTRIAL DATA CHATBOT")
    st.caption("Grounded answers from your own documents.")
    st.divider()

    st.markdown("#### Session Stats")
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", st.session_state.doc_count)
    col_b.metric("Chunks Indexed", st.session_state.chunk_count)

    if st.session_state.documents_ready:
        st.success("Index ready")
    else:
        st.warning("No documents indexed yet")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("## Industry Data Chatbot")
st.caption("Ask questions about your uploaded PDF documents")
st.divider()

upload_container = st.container(border=True)
with upload_container:
    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop your PDF files here",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        file_cols = st.columns(min(len(uploaded_files), 4) or 1)
        for i, uploaded_file in enumerate(uploaded_files):
            size_mb = uploaded_file.size / (1024 * 1024)
            with file_cols[i % len(file_cols)]:
                if size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"{uploaded_file.name}\n\nExceeds {MAX_FILE_SIZE_MB} MB limit")
                else:
                    st.markdown(f"**{uploaded_file.name}**")
                    st.caption(f"{size_mb:.2f} MB")

groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

if uploaded_files:
    valid_files = [f for f in uploaded_files if f.size / (1024 * 1024) <= MAX_FILE_SIZE_MB]
    signature = tuple(sorted((f.name, f.size) for f in valid_files))

    if valid_files and signature != st.session_state.processed_signature:
        if not groq_api_key:
            st.error("GROQ_API_KEY is not set. Add it to your environment variables.")
        else:
            with st.spinner("Extracting text and building the search index..."):
                pages = extract_pages_from_uploaded_files(valid_files)
                if not pages:
                    st.error("No readable text found in the uploaded documents.")
                else:
                    pipeline = RAGPipeline(groq_api_key=groq_api_key, groq_model=groq_model)
                    chunk_count = pipeline.index_documents(pages)
                    st.session_state.pipeline = pipeline
                    st.session_state.documents_ready = True
                    st.session_state.processed_signature = signature
                    st.session_state.chunk_count = chunk_count
                    st.session_state.doc_count = len(valid_files)
            st.success("Documents processed and indexed successfully.")
elif st.session_state.processed_signature is not None:
    st.session_state.pipeline = None
    st.session_state.documents_ready = False
    st.session_state.processed_signature = None
    st.session_state.chunk_count = 0
    st.session_state.doc_count = 0

st.divider()
st.markdown("#### Chat")

chat_container = st.container(border=True, height=460)
with chat_container:
    if not st.session_state.messages:
        st.markdown("Upload a PDF above, then ask a question to get started.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}** — page {source['page']}")

question = st.chat_input("Ask a question about your documents...")

if question:
    if not st.session_state.documents_ready or st.session_state.pipeline is None:
        st.error("Please upload a document before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            answer, sources = st.session_state.pipeline.generate_answer(question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()
