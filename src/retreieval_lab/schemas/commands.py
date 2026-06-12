from __future__ import annotations

from pathlib import Path
from typing import Any

from retreieval_lab.artifacts import data_sha256, read_json, read_jsonl, write_json
from retreieval_lab.schemas.models import json_schema_for, schema_catalog, validate_records


DEFAULT_SCHEMA_CATALOG_PATH = Path(".tmp") / "retrieval_lab" / "schema_catalog.json"
DEFAULT_SCHEMA_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "schema_validation_report.json"


def schema_catalog_command(args: Any) -> dict[str, Any]:
    rows = schema_catalog(include_json_schema=bool(getattr(args, "include_json_schema", False)))
    report = {
        "method": "retrieval_lab_schema_catalog",
        "schema_count": len(rows),
        "schemas": rows,
        "fingerprint": data_sha256(rows),
    }
    output = getattr(args, "output", None)
    if output:
        write_json(Path(output), report)
    return {
        "method": "retrieval_lab_schema_catalog",
        "summary": {
            "schema_count": report["schema_count"],
            "output": str(output) if output else "",
            "fingerprint": report["fingerprint"],
        },
        "report": report,
    }


def schema_show_command(args: Any) -> dict[str, Any]:
    schema_name = str(getattr(args, "name"))
    schema = json_schema_for(schema_name)
    report = {
        "method": "retrieval_lab_schema_show",
        "schema_name": schema_name.replace("-", "_"),
        "json_schema": schema,
        "fingerprint": data_sha256(schema),
    }
    output = getattr(args, "output", None)
    if output:
        write_json(Path(output), report)
    return {
        "method": "retrieval_lab_schema_show",
        "summary": {
            "schema_name": report["schema_name"],
            "output": str(output) if output else "",
            "fingerprint": report["fingerprint"],
        },
        "report": report,
    }


def schema_validate_command(args: Any) -> dict[str, Any]:
    schema_name = str(getattr(args, "name"))
    input_path = Path(getattr(args, "input"))
    records = load_records_for_validation(input_path)
    validation = validate_records(schema_name, records, max_issues=int(getattr(args, "max_issues", 50)))
    report = {
        "method": "retrieval_lab_schema_validate",
        "input": str(input_path),
        "summary": validation_summary(validation),
        "validation": validation,
        "fingerprint": data_sha256({"input": str(input_path), "validation": validation}),
    }
    output = Path(getattr(args, "output", DEFAULT_SCHEMA_REPORT_PATH))
    write_json(output, report)
    report["summary"]["output"] = str(output)
    return {
        "method": "retrieval_lab_schema_validate",
        "summary": report["summary"],
        "report": report,
    }


def load_records_for_validation(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = read_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("records", "rows", "run_rows", "qrels", "cases", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [data]
    raise ValueError(f"schema validation input must be JSON object/list or JSONL: {path}")


def validation_summary(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": validation["schema_name"],
        "record_count": validation["record_count"],
        "valid_count": validation["valid_count"],
        "invalid_count": validation["invalid_count"],
        "valid_rate": validation["valid_rate"],
    }


__all__ = [
    "DEFAULT_SCHEMA_CATALOG_PATH",
    "DEFAULT_SCHEMA_REPORT_PATH",
    "load_records_for_validation",
    "schema_catalog_command",
    "schema_show_command",
    "schema_validate_command",
]
