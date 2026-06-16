from __future__ import annotations

from typing import Any

from retrieval_lab.schemas import LLMJudgementModel, json_schema_for, validate_record


LLM_JUDGEMENT_SCHEMA_NAME = "llm_judgement"


def llm_judgement_contract() -> dict[str, Any]:
    return {
        "schema_name": LLM_JUDGEMENT_SCHEMA_NAME,
        "purpose": "Small sampled LLM judgement contract for qrels adjudication, rerank checks, and constraint audits.",
        "invariants": [
            "LLM judgement rows are sampled evidence, not default retrieval behavior",
            "query_id and candidate_item_id are required",
            "prompt_fingerprint should identify prompt changes without storing long prompts in default artifacts",
            "payload_chars records cost/latency pressure without requiring token accounting",
        ],
        "json_schema": json_schema_for(LLM_JUDGEMENT_SCHEMA_NAME),
    }


def validate_llm_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_record(LLM_JUDGEMENT_SCHEMA_NAME, payload)


def normalize_llm_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    return LLMJudgementModel.model_validate(payload).model_dump(mode="json", exclude_none=True)


__all__ = ["LLM_JUDGEMENT_SCHEMA_NAME", "llm_judgement_contract", "normalize_llm_judgement", "validate_llm_judgement"]
