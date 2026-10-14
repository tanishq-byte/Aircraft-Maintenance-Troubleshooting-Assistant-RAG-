
import json

import pytest

from src.data.prepare_documents import (
    build_document_text,
    iter_jsonl_records,
    iter_prepared_documents,
    prepare_document,
)



def test_build_document_text():
    record = {
        "record_id": "TEST-001",
        "source_system": "SDR",
        "report_type": "Service Difficulty Report",
        "aircraft_make": "BOEING",
        "aircraft_model": "7673Y0",
        "problem_description": "Landing gear did not extend.",
        "investigation_findings": "Hydraulic pressure was low.",
        "corrective_action": "Hydraulic line inspected.",
        "component_part_name": "LANDING GEAR",
        "component_part_condition": "FAULTY",
        "raw_text_narrative": "",
    }

    text = build_document_text(record)

    assert "Service Difficulty Report" in text
    assert "Aircraft make: BOEING" in text
    assert "Aircraft model: 7673Y0" in text
    assert "Landing gear did not extend." in text
    assert "Hydraulic pressure was low." in text
    assert "Hydraulic line inspected." in text
    assert "Component condition: FAULTY" in text


def test_empty_record_is_skipped():
    record = {
        "record_id": "TEST-001",
        "problem_description": "",
        "investigation_findings": "",
        "raw_text_narrative": "",
    }

    assert prepare_document(record) is None


def test_metadata_is_preserved():
    record = {
        "record_id": "TEST-002",
        "source_system": "FAA_SDR",
        "source_year": "2025",
        "problem_description": "Hydraulic leak reported.",
    }

    document = prepare_document(record)

    assert document is not None
    assert document["source_record_id"] == "TEST-002"
    assert document["metadata"]["source_system"] == "FAA_SDR"
    assert document["metadata"]["source_year"] == "2025"


def test_jsonl_streaming(tmp_path):
    dataset = tmp_path / "sample.jsonl"

    records = [
        {
            "record_id": "TEST-003",
            "problem_description": "Brake pressure was low.",
        },
        {
            "record_id": "TEST-004",
            "problem_description": "Fuel leak was reported.",
        },
    ]

    with dataset.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    loaded = list(iter_jsonl_records(dataset))
    prepared = list(iter_prepared_documents(dataset))

    assert len(loaded) == 2
    assert len(prepared) == 2
    assert prepared[0]["source_record_id"] == "TEST-003"


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl_records("does-not-exist.jsonl"))


def test_invalid_json(tmp_path):
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        list(iter_jsonl_records(dataset))
