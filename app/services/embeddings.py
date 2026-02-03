from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Input:
        [
            {
                "text": "...",
                "metadata": {...}
            }
        ]

        Output:
        [
            {
                "embedding": [...],
                "text": "...",
                "metadata": {...}
            }
        ]
        """
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False
        ).tolist()

        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunks.append({
                "embedding": embeddings[i],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            })

        return embedded_chunks
