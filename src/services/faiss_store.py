"""FAISS-based vector store for semantic search."""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import faiss
from src.core.config import get_settings

# Try to use real embeddings, fall back to mock if API key unavailable
try:
    from langchain_openai import OpenAIEmbeddings
    USE_REAL_EMBEDDINGS = True
except ImportError:
    USE_REAL_EMBEDDINGS = False

from src.services.mock_embeddings import MockEmbeddings


class FAISSStore:
    """
    FAISS-based vector store for fast semantic similarity search.

    Features:
    - Semantic search using OpenAI embeddings
    - Efficient similarity search with FAISS
    - Persistent storage of index and metadata
    """

    def __init__(self, index_path: str = "data/faiss_index", use_mock: bool = None):
        """
        Initialize FAISS store.

        Args:
            index_path: Path to store FAISS index and metadata
            use_mock: Force mock embeddings (auto-detect if None)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Auto-detect or use provided mock setting
        if use_mock is None:
            # Check if MOCK_MODE environment variable is set
            use_mock = os.getenv("MOCK_MODE", "false").lower() == "true"
            # Also check settings
            settings = get_settings()
            use_mock = use_mock or settings.mock_mode

        self.use_mock = use_mock

        # Initialize embeddings (real or mock)
        if use_mock or not get_settings().openai_api_key:
            print("[INFO] Using MOCK embeddings (no API calls)")
            self.embeddings = MockEmbeddings()
        else:
            try:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=get_settings().openai_api_key,
                )
                print("[INFO] Using REAL OpenAI embeddings")
            except Exception as e:
                print(f"[WARN] OpenAI embeddings failed: {e}")
                print("[INFO] Falling back to MOCK embeddings")
                self.embeddings = MockEmbeddings()
                self.use_mock = True

        self.faiss_index = None
        self.document_store = []  # Store original documents
        self.index_file = self.index_path / "faiss.index"
        self.metadata_file = self.index_path / "metadata.pkl"

        # Load existing index if available
        self._load_index()

    def _load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.

        Returns:
            True if index loaded, False if new
        """
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                self.faiss_index = faiss.read_index(str(self.index_file))

                with open(self.metadata_file, "rb") as f:
                    self.document_store = pickle.load(f)

                print(f"[INFO] Loaded FAISS index with {len(self.document_store)} documents")
                return True
        except Exception as e:
            print(f"[WARN] Failed to load index: {e}")

        return False

    def _save_index(self) -> bool:
        """
        Save FAISS index and metadata to disk.

        Returns:
            True if successful
        """
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.index_file))

                with open(self.metadata_file, "wb") as f:
                    pickle.dump(self.document_store, f)

                print(f"[INFO] Saved FAISS index with {len(self.document_store)} documents")
                return True
        except Exception as e:
            print(f"[ERROR] Failed to save index: {e}")

        return False

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Add documents to FAISS index.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional document IDs (auto-generated if not provided)

        Returns:
            True if successful
        """
        if not documents:
            return False

        try:
            # Generate embeddings for all documents
            print(f"[INFO] Generating embeddings for {len(documents)} documents...")
            embeddings = self.embeddings.embed_documents(documents)
            embeddings_array = np.array(embeddings).astype("float32")

            # Initialize FAISS index if needed
            if self.faiss_index is None:
                dimension = embeddings_array.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)

            # Add vectors to index
            self.faiss_index.add(embeddings_array)

            # Store document metadata
            for i, doc in enumerate(documents):
                doc_id = ids[i] if ids else f"doc_{len(self.document_store) + i}"
                metadata = metadatas[i] if metadatas else {}

                self.document_store.append(
                    {
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "embedding": embeddings[i],
                    }
                )

            # Save to disk
            self._save_index()
            print(f"[INFO] Added {len(documents)} documents to FAISS index")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of similar documents with scores
        """
        if self.faiss_index is None:
            print("[WARN] FAISS index not initialized")
            return []

        if not self.document_store:
            print("[WARN] No documents in store")
            return []

        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            query_array = np.array([query_embedding]).astype("float32")

            # Search in FAISS
            distances, indices = self.faiss_index.search(query_array, min(top_k, len(self.document_store)))

            # Format results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.document_store):
                    doc = self.document_store[idx]
                    # Lower distance = higher similarity
                    similarity_score = 1 / (1 + distance)

                    results.append(
                        {
                            "id": doc["id"],
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "distance": float(distance),
                            "similarity": float(similarity_score),
                        }
                    )

            return results

        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []

    def delete_index(self) -> bool:
        """
        Delete the FAISS index and metadata files.

        Returns:
            True if successful
        """
        try:
            if self.index_file.exists():
                self.index_file.unlink()

            if self.metadata_file.exists():
                self.metadata_file.unlink()

            self.faiss_index = None
            self.document_store = []

            print("[INFO] FAISS index deleted")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to delete index: {e}")
            return False

    def get_stats(self) -> dict:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index stats
        """
        return {
            "total_documents": len(self.document_store),
            "index_initialized": self.faiss_index is not None,
            "index_file_size": (
                self.index_file.stat().st_size if self.index_file.exists() else 0
            ),
        }
