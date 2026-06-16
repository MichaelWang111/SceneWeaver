from __future__ import annotations

from typing import Any

from retrieval_lab.schemas import RetrievalRunConfigModel, RunRowModel, json_schema_for, validate_record


RUN_CONFIG_SCHEMA_NAME = "retrieval_run_config"
RUN_ROW_SCHEMA_NAME = "run_row"


def retrieval_run_config_contract() -> dict[str, Any]:
    return {
        "schema_name": RUN_CONFIG_SCHEMA_NAME,
        "purpose": "Reproducible retrieval workflow configuration for semantic, lexical, hybrid, and rerank runs.",
        "invariants": [
            "workflow and top_k are required",
            "query_planner records which query understanding strategy produced the plan",
            "index_id links a run to a prepared index manifest when available",
            "llm_enabled must be explicit; real LLM use is never implied by workflow name alone",
        ],
        "json_schema": json_schema_for(RUN_CONFIG_SCHEMA_NAME),
    }


def run_row_contract() -> dict[str, Any]:
    return {
        "schema_name": RUN_ROW_SCHEMA_NAME,
        "purpose": "One query result row used by artifact-first evaluation, qrels pooling, rerank, and failure analysis.",
        "invariants": [
            "case_id is required",
            "top_results are scored candidate records",
            "query_plan may be embedded for planner/debug analysis, but long prompt text should stay out of default artifacts",
            "target fields are optional for production-like search rows but required for generated eval rows",
        ],
        "json_schema": json_schema_for(RUN_ROW_SCHEMA_NAME),
    }


def validate_retrieval_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_record(RUN_CONFIG_SCHEMA_NAME, payload)


def validate_run_row(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_record(RUN_ROW_SCHEMA_NAME, payload)


def normalize_retrieval_run_config(payload: dict[str, Any]) -> dict[str, Any]:
    return RetrievalRunConfigModel.model_validate(payload).model_dump(mode="json", exclude_none=True)


def normalize_run_row(payload: dict[str, Any]) -> dict[str, Any]:
    return RunRowModel.model_validate(payload).model_dump(mode="json", exclude_none=True)


__all__ = [
    "RUN_CONFIG_SCHEMA_NAME",
    "RUN_ROW_SCHEMA_NAME",
    "normalize_retrieval_run_config",
    "normalize_run_row",
    "retrieval_run_config_contract",
    "run_row_contract",
    "validate_retrieval_run_config",
    "validate_run_row",
]
