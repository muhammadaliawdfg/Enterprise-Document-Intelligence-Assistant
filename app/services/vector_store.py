import chromadb
from chromadb.config import Settings
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Handles vector storage and similarity search using ChromaDB
    """

    def __init__(
        self,
        persist_directory: str = "storage/vectordb",
        collection_name: str = "enterprise_docs",
    ):
        """
        Initialize ChromaDB client with persistence
        """

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        logger.info(
            f"ChromaDB initialized | Collection='{collection_name}' | Path='{persist_directory}'"
        )

    def add_documents(self, chunks: List[Dict]):
        """
        Add embedded chunks to vector store

        chunks: [
            {
                "id": str,
                "text": str,
                "embedding": List[float],
                "metadata": dict
            }
        ]
        """

        if not chunks:
            logger.warning("No chunks provided. Skipping insert.")
            return

        # Validate and normalize incoming chunks to avoid index/key errors
        valid_chunks: List[Dict] = []
        for idx, c in enumerate(chunks):
            if not isinstance(c, dict):
                logger.warning("Skipping non-dict chunk at index %d", idx)
                continue

            cid = c.get("id")
            text = c.get("text")
            emb = c.get("embedding")
            meta = c.get("metadata", {})

            if cid is None or text is None or emb is None:
                logger.warning("Skipping chunk at index %d: missing id/text/embedding", idx)
                continue

            if not isinstance(emb, (list, tuple)):
                logger.warning("Skipping chunk '%s': embedding is not a list/tuple", cid)
                continue

            valid_chunks.append({
                "id": str(cid),
                "text": str(text),
                "embedding": list(emb),
                "metadata": meta if isinstance(meta, dict) else {"metadata": meta},
            })

        if not valid_chunks:
            logger.warning("No valid chunks to insert after validation. Skipping.")
            return

        try:
            self.collection.add(
                ids=[c["id"] for c in valid_chunks],
                documents=[c["text"] for c in valid_chunks],
                embeddings=[c["embedding"] for c in valid_chunks],
                metadatas=[c.get("metadata", {}) for c in valid_chunks],
            )

            total = self.collection.count()
            logger.info("Inserted %d chunks into vector store. Total vectors: %d", len(valid_chunks), total)

        except Exception:
            logger.exception("Failed to insert documents into ChromaDB")
            raise


    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Perform semantic similarity search

        Returns:
        [
            {
                "text": str,
                "metadata": dict,
                "score": float   # lower = more similar
            }
        ]
        """

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results.get("documents"):
                logger.info("No matching documents found.")
                return []

            matches = []
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(docs)):
                matches.append(
                    {
                        "text": docs[i],
                        "metadata": metas[i],
                        "score": distances[i],
                    }
                )

            return matches

        except Exception as e:
            logger.exception("Similarity search failed")
            return []
