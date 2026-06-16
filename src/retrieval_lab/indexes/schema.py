from __future__ import annotations

from typing import Any

from retrieval_lab.schemas import IndexManifestModel, json_schema_for, validate_record


INDEX_SCHEMA_NAME = "index_manifest"


def index_manifest_contract() -> dict[str, Any]:
    return {
        "schema_name": INDEX_SCHEMA_NAME,
        "purpose": "Prepared index and cache identity contract for reproducible retrieval runs.",
        "invariants": [
            "index_id and item_count are required",
            "channels records which semantic/lexical/signature channels were prepared",
            "fingerprint should change when indexed items, tokenization, embeddings, or channel config changes",
            "cache_paths are metadata only; they do not imply a cache is trusted without matching fingerprint",
        ],
        "json_schema": json_schema_for(INDEX_SCHEMA_NAME),
    }


def validate_index_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_record(INDEX_SCHEMA_NAME, payload)


def normalize_index_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    return IndexManifestModel.model_validate(payload).model_dump(mode="json", exclude_none=True)


__all__ = ["INDEX_SCHEMA_NAME", "index_manifest_contract", "normalize_index_manifest", "validate_index_manifest"]
