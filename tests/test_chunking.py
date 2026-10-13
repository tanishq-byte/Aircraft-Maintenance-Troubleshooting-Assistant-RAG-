
from src.data.chunking import chunk_text, create_chunks


def test_empty_text():
    assert chunk_text("") == []


def test_short_text():
    text = "Hydraulic pressure fluctuated during retraction."
    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert "Hydraulic pressure" in chunks[0]


def test_long_text():
    text = "hydraulic pressure fluctuation " * 1000
    chunks = chunk_text(text)

    assert len(chunks) > 1


def test_chunk_metadata():
    records = create_chunks(
        text="Hydraulic pressure fluctuated during retraction.",
        source_id="test-report-001",
        metadata={"source_type": "faa_sdr"},
    )

    assert len(records) == 1
    assert records[0]["source_record_id"] == "test-report-001"
    assert records[0]["metadata"]["source_type"] == "faa_sdr"
    assert records[0]["chunk_id"] == create_chunks(
        text="Hydraulic pressure fluctuated during retraction.",
        source_id="test-report-001",
        metadata={"source_type": "faa_sdr"},
    )[0]["chunk_id"]


from src.data.chunking import CHUNK_SIZE, CHUNK_OVERLAP, tokenizer


def test_chunk_token_limit():
    text = "Hydraulic pressure fluctuated during retraction. " * 1000
    chunks = chunk_text(text)

    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
        assert token_count <= CHUNK_SIZE



def test_chunks_have_overlap():
    text = "hydraulic pressure fluctuated during retraction. " * 1000
    chunks = chunk_text(text)

    assert len(chunks) > 1

    # The end of one chunk should also appear in the next chunk.
    overlapping_text = " ".join(chunks[0].split()[-5:])

    assert overlapping_text in chunks[1]

