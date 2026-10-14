
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingModel:
    """Generate local embeddings for documents and search queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if device is None:
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device

        print(f"Loading embedding model: {model_name}")
        print(f"Device: {device}")

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

        if self.embedding_dimension is None:
            raise RuntimeError("Could not determine embedding dimension")

        print(
            f"Embedding dimension: {self.embedding_dimension}"
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Embed document chunks in batches."""

        if isinstance(texts, str):
            raise TypeError(
                "Pass a sequence of texts, not one string"
            )

        texts = list(texts)

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every document must be a string")

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts cannot be empty")

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )

        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a user query for semantic search."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        query_text = QUERY_PREFIX + query.strip()

        vector = self.model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(vector, dtype=np.float32)
