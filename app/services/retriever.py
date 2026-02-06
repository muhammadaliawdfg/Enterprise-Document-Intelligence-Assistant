from typing import List, Dict
from app.services.embeddings import get_embedding
from app.services.vector_store import get_vector_store


class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.vector_db = get_vector_store()

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-k relevant chunks from vector database
        """

        # 1️⃣ Generate query embedding
        query_embedding = get_embedding(query)

        # 2️⃣ Search vector database
        results = self.vector_db.similarity_search_by_vector(
            query_embedding,
            k=self.top_k
        )

        # 3️⃣ Format results with metadata for RAG + Citation
        retrieved_chunks = []

        for r in results:
            retrieved_chunks.append({
                "text": r.page_content,
                "document_name": r.metadata.get("document_name"),
                "source": r.metadata.get("source"),
                "page_number": r.metadata.get("page_number"),
                "chunk_id": r.metadata.get("chunk_id"),
                "chunk_index": r.metadata.get("chunk_index"),
                "token_length": r.metadata.get("token_length")
            })

        return retrieved_chunks
