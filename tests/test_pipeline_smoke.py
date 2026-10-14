
import json
import zipfile
from pathlib import Path

import numpy as np

from src.data.prepare_documents import prepare_document
from src.data.chunking import create_chunks
from src.retrieval.embeddings import EmbeddingModel


ZIP_PATH = Path("aviation_maintenance_dataset_rag_ready_final.zip")
JSONL_NAME = "combined_rag_ready_final.jsonl"


def test_five_real_records_end_to_end():
    assert ZIP_PATH.is_file(), f"ZIP file not found: {ZIP_PATH}"

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        with archive.open(JSONL_NAME, "r") as file:
            documents = []

            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line.decode("utf-8"))
                document = prepare_document(record)

                if document is not None:
                    documents.append(document)

                if len(documents) == 5:
                    break

    assert len(documents) == 5

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
    assert all(chunk["text"].strip() for chunk in chunks)

    model = EmbeddingModel(batch_size=8)
    vectors = model.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    assert vectors.shape == (len(chunks), 384)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()

    print(f"\nPrepared documents: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedding matrix shape: {vectors.shape}")
    print(f"First source ID: {chunks[0]['source_record_id']}")
