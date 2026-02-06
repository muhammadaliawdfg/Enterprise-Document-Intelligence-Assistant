# app/services/rag.py
from typing import List, Dict

from dotenv import load_dotenv
from .vector_store import VectorStoreService
from .embeddings import EmbeddingService
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

# Initialize services
embed_service = EmbeddingService()
vector_store = VectorStoreService()

# Load environment
load_dotenv(dotenv_path=".env", override=True)

# OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks from vector store.
    """
    embedding = embed_service.embed_query(query)

    results = vector_store.similarity_search(
        query_embedding=embedding,
        top_k=top_k
    )

    return results


# -----------------------------
# BUILD PROMPT
# -----------------------------
def build_prompt(retrieved_chunks: List[Dict], query: str) -> str:
    docs_text = "\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""
You are a professional AI assistant.

IMPORTANT RULES:
- Use ONLY the provided document excerpts.
- If answer is not present, say: "Information not found in documents."
- Do NOT make assumptions.

Documents:
{docs_text}

Question:
{query}

Answer:
"""
    return prompt


# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    return answer


# -----------------------------
# EXTRACT SOURCES (MODULE 6 CORE)
# -----------------------------
def extract_sources(retrieved_chunks: List[Dict]) -> List[Dict]:

    seen = set()
    sources = []

    for c in retrieved_chunks:
        key = (c["metadata"].get("source"),
               c["metadata"].get("page_number"))

        if key not in seen:
            seen.add(key)

            sources.append({
                "document_name": c["metadata"].get("document_name"),
                "source": c["metadata"].get("source"),
                "page_number": c["metadata"].get("page_number"),
                "chunk_id": c["metadata"].get("chunk_id")
            })

    return sources


# -----------------------------
# FULL RAG PIPELINE
# -----------------------------
def rag_pipeline(query: str, top_k: int = 5) -> Dict:
    """
    Full RAG pipeline:
    retrieve -> prompt -> generate -> sources
    """

    logger.info(f"RAG pipeline started for query: {query}")

    # 1️⃣ Retrieve
    chunks = retrieve_chunks(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": []
        }

    # 2️⃣ Prompt
    prompt = build_prompt(chunks, query)

    # 3️⃣ Generate Answer
    answer = generate_answer(prompt)

    # 4️⃣ Extract Sources ⭐ Module 6
    sources = extract_sources(chunks)
    if "Information not found in documents." in answer:
        sources = []  # Clear sources if answer indicates no info found
    return {
        "answer": answer,
        "sources": sources
    }
