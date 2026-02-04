from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
import uuid

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedding model
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Input:
        [
            {
                "text": "...",
                "metadata": {
                    "document_name": "...",
                    "page_number": 1,
                    "chunk_id": "doc_p1_0"
                }
            }
        ]

        Output:
        [
            {
                "id": "unique_chunk_id",
                "embedding": [...],
                "text": "...",
                "metadata": {...}
            }
        ]
        """

        if not chunks:
            logger.warning("No chunks provided for embedding.")
            return []

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False
        ).tolist()

        embedded_chunks = []

        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})

            # Prefer existing chunk_id, else generate safe unique ID
            chunk_id = metadata.get(
                "chunk_id",
                f"{metadata.get('document_name', 'doc')}_{metadata.get('page_number', 0)}_{i}_{uuid.uuid4().hex[:8]}"
            )

            embedded_chunks.append({
                "id": chunk_id,
                "embedding": embeddings[i],
                "text": chunk["text"],
                "metadata": metadata
            })

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks.")
        return embedded_chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a user query for similarity search
        """
        return self.model.encode(query).tolist()
