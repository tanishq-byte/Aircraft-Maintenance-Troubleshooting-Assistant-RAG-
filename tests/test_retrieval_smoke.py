
import json
import zipfile
from pathlib import Path

from src.data.prepare_documents import prepare_document
from src.data.chunking import create_chunks
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


ZIP_PATH = Path("aviation_maintenance_dataset_rag_ready_final.zip")
JSONL_NAME = "combined_rag_ready_final.jsonl"


def test_real_embeddings_retrieved_from_qdrant(tmp_path):
    assert ZIP_PATH.is_file(), f"Dataset ZIP not found: {ZIP_PATH}"

    documents = []

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        with archive.open(JSONL_NAME, "r") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line.decode("utf-8"))
                document = prepare_document(record)

                if document is not None:
                    documents.append(document)

                if len(documents) >= 20:
                    break

    assert documents, "No usable records found"

    chunks = []

    for document in documents:
        chunks.extend(
            create_chunks(
                text=document["text"],
                source_id=document["source_record_id"],
                metadata=document["metadata"],
            )
        )

    assert chunks, "No chunks were created"

    model = EmbeddingModel(batch_size=8)
    vectors = model.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    store = VectorStore(
        storage_path=tmp_path / "qdrant",
        collection_name="aviation_test",
        vector_size=384,
    )

    try:
        stored_count = store.upsert_chunks(chunks, vectors)
        assert stored_count == len(chunks)

        query_vector = model.embed_query(
            "Why might an aircraft battery charger show a fault?"
        )

        results = store.search(query_vector, limit=5)

        assert results
        assert all(result["text"] for result in results)
        assert all(result["source_record_id"] for result in results)

        print(f"\nDocuments loaded: {len(documents)}")
        print(f"Chunks indexed: {len(chunks)}")
        print("\nTop retrieval results:")

        for result in results:
            print(
                f"{result['score']:.4f} | "
                f"{result['source_record_id']} | "
                f"{result['text'][:150]}"
            )
    finally:
        store.close()
