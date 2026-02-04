# app/services/rag.py
from typing import List, Dict

from dotenv import load_dotenv
from .vector_store import VectorStoreService
from .embeddings import EmbeddingService
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

# Initialize embedding service
embed_service = EmbeddingService()
vector_store = VectorStoreService()

# Initialize LLM client (reads API key from env var `OPENAI_API_KEY`)
load_dotenv(dotenv_path=".env", override=True)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks from vector store.
    """
    embedding = embed_service.embed_query(query)
    results = vector_store.similarity_search(query_embedding=embedding, top_k=top_k)
    return results


def build_prompt(retrieved_chunks: List[Dict], query: str) -> str:
    """
    Build a prompt template using retrieved chunks.
    """
    docs_text = "\n\n".join([c["text"] for c in retrieved_chunks])
    prompt = f"""
You are an AI assistant. Use ONLY the following document excerpts to answer the question.
Do not include any information not present in these documents.

Documents:
{docs_text}

Question:
{query}

Answer:
"""
    return prompt


def generate_answer(prompt: str) -> str:
    """
    Call OpenAI GPT model to generate an answer.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return answer


def rag_pipeline(query: str, top_k: int = 5) -> str:
    """
    Full RAG pipeline: retrieve -> prompt -> generate
    """
    logger.info(f"RAG pipeline started for query: {query}")

    # 1️⃣ Retrieve relevant chunks
    chunks = retrieve_chunks(query, top_k=top_k)
    if not chunks:
        return "No relevant documents found."

    # 2️⃣ Build prompt
    prompt = build_prompt(chunks, query)

    # 3️⃣ Generate answer
    answer = generate_answer(prompt)
    return answer
