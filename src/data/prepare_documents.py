
import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl_records(
    file_path: str | Path,
) -> Iterator[dict[str, Any]]:
    """Read a JSONL file one record at a time."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}"
                )

            yield record


def build_document_text(record: dict[str, Any]) -> str:
    """Build searchable text using the actual dataset schema."""

    fields = (
        ("Source", "source_system"),
        ("Report type", "report_type"),
        ("Aircraft make", "aircraft_make"),
        ("Aircraft model", "aircraft_model"),
        ("Aircraft serial number", "aircraft_serial_number"),
        ("Event date", "event_date"),
        ("Problem description", "problem_description"),
        ("Investigation findings", "investigation_findings"),
        ("Corrective action", "corrective_action"),
        ("Component", "component_part_name"),
        ("Component condition", "component_part_condition"),
        ("Aircraft location", "location_on_aircraft"),
        ("Severity", "severity_level"),
        ("Narrative", "raw_text_narrative"),
    )

    sections = []

    for label, field in fields:
        value = record.get(field)

        if value is None:
            continue

        value = str(value).strip()

        if value:
            sections.append(f"{label}: {value}")

    return "\n\n".join(sections)



def prepare_document(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    """Prepare one source record for chunking."""

    source_id = record.get("record_id")

    if not isinstance(source_id, str) or not source_id.strip():
        return None

    text = build_document_text(record)

    if not text:
        return None

   
    metadata_fields = (
        "source_system",
        "source_year",
        "event_date",
        "aircraft_make",
        "aircraft_model",
        "aircraft_serial_number",
        "component_part_name",
        "component_part_condition",
        "location_on_aircraft",
        "severity_level",
        "report_type",
        "geographic_location",
        "data_quality_flags",
        "source_record_hash",
    )


    metadata = {
        field: record[field]
        for field in metadata_fields
        if field in record
    }

    return {
        "source_record_id": source_id.strip(),
        "text": text,
        "metadata": metadata,
    }


def iter_prepared_documents(
    file_path: str | Path,
) -> Iterator[dict[str, Any]]:
    """Stream valid, text-bearing documents from a JSONL dataset."""

    for record in iter_jsonl_records(file_path):
        document = prepare_document(record)

        if document is not None:
            yield document
