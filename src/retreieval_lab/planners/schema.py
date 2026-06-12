from __future__ import annotations

from typing import Any

from retreieval_lab.schemas import QueryPlanModel, json_schema_for, validate_record


PLAN_SCHEMA_NAME = "query_plan"


def query_plan_contract() -> dict[str, Any]:
    return {
        "schema_name": PLAN_SCHEMA_NAME,
        "purpose": "Stable query understanding contract shared by rule, multi-query, HyDE, and LLM planners.",
        "invariants": [
            "positive_query is required and must be free of explicit negative constraints",
            "forbidden_stage and negative_style are constraints, not positive retrieval text",
            "scene_signature is structured evidence for exact-scene ranking",
            "hard_constraints are veto/filter candidates; soft_constraints are penalties or review signals",
            "LLM planner outputs must validate against this contract before entering retrieval experiments",
        ],
        "json_schema": json_schema_for(PLAN_SCHEMA_NAME),
    }


def validate_query_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return validate_record(PLAN_SCHEMA_NAME, payload)


def normalize_query_plan(payload: dict[str, Any]) -> dict[str, Any]:
    return QueryPlanModel.model_validate(payload).model_dump(mode="json", exclude_none=True)


__all__ = ["PLAN_SCHEMA_NAME", "normalize_query_plan", "query_plan_contract", "validate_query_plan"]
