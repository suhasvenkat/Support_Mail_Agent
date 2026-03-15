"""Knowledge base document loader."""

from pathlib import Path
from typing import List
from src.services.vector_store import VectorStore


class KnowledgeBaseLoader:
    """
    Loader for FAQ and documentation files.

    TODO: Implement document loading from files and embedding into vector store.
    """

    def __init__(self):
        """Initialize loader with vector store."""
        self.vector_store = VectorStore()
        self.docs_path = Path(__file__).parent / "docs"

    def load_documents(self) -> List[str]:
        """
        Load all FAQ/doc files from docs/ directory.

        TODO: Implement file reading and parsing.

        Returns:
            List of loaded documents
        """
        # Placeholder
        print(f"[MOCK] Loading documents from {self.docs_path}")
        return []

    def ingest(self) -> bool:
        """
        Load and ingest all documents into vector store.

        TODO: Implement full ingest pipeline.

        Returns:
            True if successful
        """
        documents = self.load_documents()
        if not documents:
            print("[MOCK] No documents to ingest")
            return False

        return self.vector_store.add_documents(documents)
