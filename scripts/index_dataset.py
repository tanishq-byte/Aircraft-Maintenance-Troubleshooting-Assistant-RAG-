
import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from src.data.prepare_documents import prepare_document
from src.data.chunking import create_chunks
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


DEFAULT_ZIP = "aviation_maintenance_dataset_rag_ready_final.zip"
DEFAULT_JSONL = "combined_rag_ready_final.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local Qdrant index from aviation records."
    )

    parser.add_argument("--zip-path", default=DEFAULT_ZIP)
    parser.add_argument("--jsonl-name", default=DEFAULT_JSONL)
    parser.add_argument("--storage-path", default="storage/qdrant_pilot")
    parser.add_argument("--collection", default="aviation_pilot")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()

    if args.max_records <= 0:
        parser.error("--max-records must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    return args


def iter_records(
    archive: zipfile.ZipFile,
    jsonl_name: str,
    max_records: int,
):
    """Stream records from a JSONL member of a ZIP archive."""

    available_files = archive.namelist()

    if jsonl_name not in available_files:
        raise FileNotFoundError(
            f"{jsonl_name!r} not found in ZIP. "
            f"Available files: {available_files}"
        )

    with archive.open(jsonl_name, "r") as file:
        records_read = 0

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid record at JSONL line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}"
                )

            records_read += 1
            yield record

            if records_read >= max_records:
                break


def flush_batch(
    chunks: list[dict[str, Any]],
    embedder: EmbeddingModel,
    store: VectorStore,
) -> int:
    """Embed and persist one batch of chunks."""

    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.embed_documents(texts)

    stored = store.upsert_chunks(chunks, vectors)

    if stored != len(chunks):
        raise RuntimeError(
            f"Expected to store {len(chunks)} chunks, stored {stored}"
        )

    return stored


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)

    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {zip_path.resolve()}"
        )

    print(f"Dataset: {zip_path}")
    print(f"Record limit: {args.max_records}")
    print(f"Embedding batch size: {args.batch_size}")
    print(f"Qdrant collection: {args.collection}")
    print(f"Qdrant storage: {args.storage_path}")

    embedder = EmbeddingModel(batch_size=args.batch_size)

    store = VectorStore(
        storage_path=args.storage_path,
        collection_name=args.collection,
        vector_size=embedder.embedding_dimension,
    )

    records_read = 0
    documents_prepared = 0
    records_skipped = 0
    chunks_created = 0
    chunks_stored = 0
    pending_chunks: list[dict[str, Any]] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for record in iter_records(
                archive,
                args.jsonl_name,
                args.max_records,
            ):
                records_read += 1
                document = prepare_document(record)

                if document is None:
                    records_skipped += 1
                    continue

                documents_prepared += 1

                document_chunks = create_chunks(
                    text=document["text"],
                    source_id=document["source_record_id"],
                    metadata=document["metadata"],
                )

                chunks_created += len(document_chunks)
                pending_chunks.extend(document_chunks)

                while len(pending_chunks) >= args.batch_size:
                    batch = pending_chunks[:args.batch_size]
                    del pending_chunks[:args.batch_size]

                    chunks_stored += flush_batch(
                        batch,
                        embedder,
                        store,
                    )

                    print(
                        f"\rRecords: {records_read} | "
                        f"Prepared: {documents_prepared} | "
                        f"Chunks stored: {chunks_stored}",
                        end="",
                        flush=True,
                    )

        if pending_chunks:
            chunks_stored += flush_batch(
                pending_chunks,
                embedder,
                store,
            )

        print("\n\nIndexing summary")
        print("----------------")
        print(f"Records read:        {records_read}")
        print(f"Documents prepared:  {documents_prepared}")
        print(f"Records skipped:     {records_skipped}")
        print(f"Chunks created:      {chunks_created}")
        print(f"Chunks stored:       {chunks_stored}")
        print(f"Storage path:        {Path(args.storage_path).resolve()}")
        print(f"Collection:          {args.collection}")

        if chunks_created != chunks_stored:
            raise RuntimeError(
                "Not all generated chunks were stored successfully."
            )

    finally:
        store.close()


if __name__ == "__main__":
    main()
