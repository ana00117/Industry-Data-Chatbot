import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from prompt import SYSTEM_PROMPT, build_user_prompt
from vector_store import VectorStore

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "source": page["source"],
                "page": page["page"]
            })
    return chunks


class RAGPipeline:
    def __init__(self, groq_api_key, groq_model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=groq_api_key)
        self.model = groq_model
        self.vector_store = VectorStore()

    def index_documents(self, pages):
        chunks = chunk_pages(pages)
        self.vector_store.build(chunks)
        return len(chunks)

    def retrieve(self, question, top_k=TOP_K):
        return self.vector_store.search(question, top_k=top_k)

    def generate_answer(self, question):
        retrieved_chunks = self.retrieve(question)
        if not retrieved_chunks:
            return "I could not find this information in the uploaded documents.", []

        user_prompt = build_user_prompt(question, retrieved_chunks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        answer = response.choices[0].message.content
        return answer, retrieved_chunks

    def save_index(self, directory):
        self.vector_store.save(directory)

    def load_index(self, directory):
        self.vector_store.load(directory)
