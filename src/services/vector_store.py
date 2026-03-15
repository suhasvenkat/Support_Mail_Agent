"""Vector store service using Chromadb."""

from typing import List, Optional
import chromadb
from pathlib import Path
from src.core.config import get_settings


class VectorStore:
    """
    Vector store service for knowledge base operations using Chroma.

    Handles semantic search and document storage with embeddings.
    """

    def __init__(self):
        """Initialize vector store with Chroma client."""
        settings = get_settings()
        self.chroma_path = settings.chroma_path

        # Create data directory if it doesn't exist
        Path(self.chroma_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize Chroma persistent client
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.client.get_or_create_collection(
                name="support_docs",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print(f"[WARN] Chroma initialization failed: {e}")
            self.client = None
            self.collection = None

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search knowledge base for relevant documents using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of relevant documents with metadata and distance scores
        """
        if not self.collection:
            print("[WARN] Vector store not initialized")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            documents = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    documents.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "distance": results["distances"][0][i] if results["distances"] else 0,
                        }
                    )

            return documents

        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Add documents to knowledge base with automatic embedding.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional document IDs (auto-generated if not provided)

        Returns:
            True if successful, False otherwise
        """
        if not self.collection:
            print("[WARN] Vector store not initialized")
            return False

        if not documents:
            return False

        try:
            # Generate IDs if not provided
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # Add documents to collection (Chroma auto-embeds)
            self.collection.add(
                documents=documents,
                metadatas=metadatas or [{} for _ in documents],
                ids=ids,
            )

            print(f"[INFO] Added {len(documents)} documents to knowledge base")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            return False
