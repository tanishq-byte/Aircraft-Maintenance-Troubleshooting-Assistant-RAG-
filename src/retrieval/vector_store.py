
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


DEFAULT_COLLECTION = "aviation_chunks"
DEFAULT_VECTOR_SIZE = 384


class VectorStore:
    """Persist and search aviation document embeddings locally."""

    def __init__(
        self,
        storage_path: str | Path = "storage/qdrant",
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.vector_size = vector_size

        self.client = QdrantClient(path=str(self.storage_path))

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Convert our chunk ID to a deterministic UUID accepted by Qdrant."""
        return str(uuid5(NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: Any,
    ) -> int:
        """Store chunk vectors and their text/metadata together."""

        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")

        points = []

        for chunk, vector in zip(chunks, vectors):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            source_id = chunk.get("source_record_id")

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Every chunk needs a non-empty chunk_id")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every chunk needs non-empty text")

            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Every chunk needs a source_record_id")

            
            if len(vector) != self.vector_size:
                raise ValueError(
                f"Invalid vector size: expected {self.vector_size}, "
                f"got {len(vector)}"
             )


            payload = {
                "chunk_id": chunk_id,
                "source_record_id": source_id,
                "chunk_index": chunk.get("chunk_index", 0),
                "text": text,
                "metadata": chunk.get("metadata", {}),
            }

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        return len(points)

    def search(
        self,
        query_vector: Any,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar chunks for an embedding vector."""

        if limit <= 0:
            raise ValueError("limit must be positive")

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Expected query vector size {self.vector_size}, "
                f"got {len(query_vector)}"
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "score": point.score,
                **(point.payload or {}),
            }
            for point in results
        ]

    def close(self) -> None:
        """Release the local database handle."""
        self.client.close()
