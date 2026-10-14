
import numpy as np
import pytest

from src.retrieval.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    vector_store = VectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="test_aviation",
        vector_size=3,
    )

    yield vector_store
    vector_store.close()


def test_store_and_search(store):
    chunks = [
        {
            "chunk_id": "chunk-test-001",
            "source_record_id": "SDR-001",
            "chunk_index": 0,
            "text": "Landing gear failed to extend.",
            "metadata": {"aircraft_model": "767"},
        },
        {
            "chunk_id": "chunk-test-002",
            "source_record_id": "SDR-002",
            "chunk_index": 0,
            "text": "Cabin lighting failed.",
            "metadata": {"aircraft_model": "737"},
        },
    ]

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert store.upsert_chunks(chunks, vectors) == 2

    results = store.search(
        np.array([0.9, 0.1, 0.0], dtype=np.float32),
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-test-001"
    assert results[0]["source_record_id"] == "SDR-001"
    assert results[0]["metadata"]["aircraft_model"] == "767"


def test_empty_upsert(store):
    assert store.upsert_chunks([], np.empty((0, 3))) == 0


def test_mismatched_chunk_and_vector_count(store):
    chunks = [
        {
            "chunk_id": "chunk-test-003",
            "source_record_id": "SDR-003",
            "text": "Hydraulic pressure was low.",
        }
    ]

    with pytest.raises(ValueError, match="exactly one vector"):
        store.upsert_chunks(chunks, np.empty((0, 3)))


def test_invalid_vector_size(store):
    chunks = [
        {
            "chunk_id": "chunk-test-004",
            "source_record_id": "SDR-004",
            "text": "Brake pressure was low.",
        }
    ]

    with pytest.raises(ValueError, match="vector size"):
        store.upsert_chunks(
            chunks,
            np.array([[1.0, 0.0]], dtype=np.float32),
        )


def test_invalid_search_limit(store):
    with pytest.raises(ValueError, match="limit must be positive"):
        store.search(np.array([1.0, 0.0, 0.0]), limit=0)
