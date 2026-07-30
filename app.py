import os
import streamlit as st
from dotenv import load_dotenv
from document_loader import extract_pages_from_uploaded_files
from rag_pipeline import RAGPipeline

load_dotenv()

MAX_FILE_SIZE_MB = 20
ALLOWED_FILE_TYPES = ["pdf"]

st.set_page_config(page_title="Industry Data Chatbot", page_icon="📄", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_ready" not in st.session_state:
    st.session_state.documents_ready = False

st.title("Industry Data Chatbot")
st.caption("Ask questions about your uploaded PDF documents. Answers are grounded strictly in the uploaded content.")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Uploaded files:")
        for uploaded_file in uploaded_files:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"{uploaded_file.name} exceeds {MAX_FILE_SIZE_MB} MB limit.")
            else:
                st.write(f"- {uploaded_file.name} ({size_mb:.2f} MB)")

    process_clicked = st.button("Process Documents", use_container_width=True)
    clear_clicked = st.button("Clear Chat", use_container_width=True)

    if clear_clicked:
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info("Verify high-stakes information independently. Generated answers may be incomplete or incorrect.")

groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file.")
    elif not groq_api_key:
        st.sidebar.error("GROQ_API_KEY is not set. Add it to your environment variables.")
    else:
        valid_files = [f for f in uploaded_files if f.size / (1024 * 1024) <= MAX_FILE_SIZE_MB]
        with st.spinner("Extracting and indexing documents..."):
            pages = extract_pages_from_uploaded_files(valid_files)
            if not pages:
                st.sidebar.error("No readable text found in the uploaded documents.")
            else:
                pipeline = RAGPipeline(groq_api_key=groq_api_key, groq_model=groq_model)
                pipeline.index_documents(pages)
                st.session_state.pipeline = pipeline
                st.session_state.documents_ready = True
        st.sidebar.success("Documents processed successfully.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"{source['source']} - Page {source['page']}")

question = st.chat_input("Ask a question about your documents...")

if question:
    if not st.session_state.documents_ready or st.session_state.pipeline is None:
        st.error("Please upload and process documents before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = st.session_state.pipeline.generate_answer(question)
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"{source['source']} - Page {source['page']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
