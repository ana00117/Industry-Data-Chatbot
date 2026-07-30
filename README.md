# Industry Data Chatbot

A domain-specific Retrieval-Augmented Generation (RAG) chatbot that answers questions from user-uploaded PDF documents: course notes, company policies, manuals, legal documents, or training material. Answers are generated strictly from the retrieved document content, with source document and page references shown alongside every response.

## Overview

Large documents are difficult to search manually. This application lets users upload one or more PDFs and ask questions in natural language. It retrieves the most relevant passages using semantic search and generates a grounded answer using a large language model, citing the exact document and page the answer came from.

## Features

- Multi-file PDF upload with automatic, on-upload processing (no manual "process" step)
- Page-level text extraction with document and page metadata preserved
- Recursive text chunking with overlap to preserve context across boundaries
- Semantic embeddings via Sentence Transformers
- Fast similarity search using a FAISS vector index
- Grounded answer generation with a strict no-hallucination system prompt
- Automatic refusal when an answer is not present in the uploaded documents
- Source document and page number shown for every answer
- Persistent chat history within a session, with a one-click reset
- Clean, native Streamlit interface, no custom CSS


## Tech Stack

| Component            | Technology                          |
|-----------------------|--------------------------------------|
| Programming language  | Python                              |
| User interface        | Streamlit                           |
| PDF text extraction   | pypdf                               |
| Text splitting        | LangChain text splitters            |
| Embeddings             | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database        | FAISS                               |
| Language model         | Groq (Llama 3.1)                    |
| Environment management | python-dotenv                       |
| Deployment              | Streamlit Community Cloud           |

## Project Structure

```
domain_rag_chatbot/
├── app.py                  # Streamlit interface
├── rag_pipeline.py         # Chunking, retrieval, and generation logic
├── document_loader.py      # PDF text extraction
├── vector_store.py         # Embedding creation and FAISS index management
├── prompt.py                # System prompt and prompt construction
├── requirements.txt        # Python dependencies
├── .env                     # Environment variables (not committed)
├── .gitignore
├── .streamlit/
│   └── config.toml          # Application theme
├── documents/
│   └── sample.pdf           # Sample document for testing
└── tests/
    └── test_questions.csv   # Evaluation question set
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
git clone <your-repository-url>
cd domain_rag_chatbot
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Running Locally

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`), upload a PDF, and start asking questions.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub, with `app.py` and `requirements.txt` at the repository root.
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), create a new app pointing to the repository and `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   GROQ_MODEL = "llama-3.1-8b-instant"
   ```
4. Deploy. The `.streamlit/config.toml` theme is picked up automatically.

## How It Works

1. **Upload** — PDFs are uploaded through the interface and validated for type and size.
2. **Extraction** — Each page is read with `pypdf`; empty pages are skipped, and the source file name and page number are retained as metadata.
3. **Chunking** — Extracted text is split into overlapping chunks (700–1000 characters, 100–150 character overlap) to preserve context across chunk boundaries.
4. **Embedding** — Each chunk is converted into a vector using the `all-MiniLM-L6-v2` sentence embedding model.
5. **Indexing** — Embeddings and metadata are stored in an in-memory FAISS index for similarity search.
6. **Retrieval** — On each question, the top matching chunks are retrieved by cosine similarity.
7. **Generation** — The retrieved chunks and the question are sent to the language model with a strict prompt that forbids answering outside the given context.
8. **Response** — The answer is displayed along with the source document and page number for verification.

## Evaluation

`tests/test_questions.csv` contains a sample set of test questions with expected sources, used to manually verify:

- Retrieval accuracy — did the system find the correct document and page?
- Groundedness — does the answer avoid unsupported information?
- Refusal quality — does the chatbot correctly decline when the answer is unavailable?

## Responsible AI and Security

- API keys are stored as environment variables and are never committed to source control.
- The system prompt instructs the model to ignore any instructions embedded within uploaded documents.
- The application does not claim generated answers are automatically correct; users are advised to verify high-stakes information independently.
- Uploaded file types and sizes are validated before processing.

## License

This project is provided for educational purposes as part of a guided student project.
