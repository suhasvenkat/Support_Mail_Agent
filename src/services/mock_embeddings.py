"""Mock embeddings service - No API calls needed"""

import hashlib
import json
from typing import List


class MockEmbeddings:
    """
    Generates mock embeddings deterministically based on text.
    Perfect for testing without OpenAI API calls.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.embedding_dim = 1536  # Same as real embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings for documents."""
        return [self._text_to_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Generate deterministic mock embedding for query."""
        return self._text_to_embedding(text)

    def _text_to_embedding(self, text: str) -> List[float]:
        """
        Convert text to deterministic embedding vector.
        Same input always produces same output.
        """
        # Hash the text to get a seed
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)

        # Generate deterministic embedding
        embedding = []
        for i in range(self.embedding_dim):
            # Use hash to generate pseudo-random but deterministic values
            seed = (hash_int + i) % (2**32)
            # Generate value between -1 and 1
            value = ((seed % 1000) / 1000.0) - 0.5
            embedding.append(value)

        # Normalize to unit vector (like real embeddings)
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding
