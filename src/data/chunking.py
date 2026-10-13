
import hashlib
from typing import Any

from transformers import AutoTokenizer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def create_chunk_id(source_id: str, chunk_index: int) -> str:
    raw_id = f"{source_id}:{chunk_index}"
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
    return f"chunk-{digest}"



def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(offsets), step):
        end = min(start + chunk_size, len(offsets))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        # Shrink the chunk if slicing changes tokenization at its boundary.
        while end > start:
            chunk = text[char_start:char_end].strip()
            token_count = len(
                tokenizer.encode(chunk, add_special_tokens=False)
            )

            if token_count <= chunk_size:
                break

            end -= 1
            if end > start:
                char_end = offsets[end - 1][1]

        if end <= start:
            raise ValueError("Unable to create a valid token-sized chunk")

        if chunk:
            chunks.append(chunk)

        if end == len(offsets):
            break

    return chunks


def create_chunks(
    text: str,
    source_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")

    metadata = metadata or {}
    text_chunks = chunk_text(text)
    records = []

    for index, chunk in enumerate(text_chunks):
        records.append(
            {
                "chunk_id": create_chunk_id(source_id, index),
                "source_record_id": source_id,
                "chunk_index": index,
                "text": chunk,
                "metadata": metadata.copy(),
            }
        )

    return records
