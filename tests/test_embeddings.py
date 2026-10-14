
import numpy as np
import pytest

from src.retrieval.embeddings import EmbeddingModel


@pytest.fixture(scope="module")
def embedder():
    return EmbeddingModel(batch_size=2)


def test_document_embedding_shape(embedder):
    texts = [
        "Hydraulic pressure fluctuated during retraction.",
        "The landing gear failed to extend.",
    ]

    vectors = embedder.embed_documents(texts)

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32


def test_embeddings_are_normalized(embedder):
    texts = [
        "Hydraulic pressure fluctuated during retraction.",
        "The landing gear failed to extend.",
    ]

    vectors = embedder.embed_documents(texts)
    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_query_embedding_shape(embedder):
    vector = embedder.embed_query(
        "Why did the landing gear fail to extend?"
    )

    assert vector.shape == (384,)
    assert np.isclose(
        np.linalg.norm(vector), 1.0, atol=1e-5
    )


def test_empty_document_list(embedder):
    vectors = embedder.embed_documents([])

    assert vectors.shape == (0, 384)


def test_invalid_query(embedder):
    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_empty_document_rejected(embedder):
    with pytest.raises(ValueError):
        embedder.embed_documents([""])
