from __future__ import annotations

from pathlib import Path
from typing import Any

from retreieval_lab.artifacts import write_json
from retreieval_lab.planners.service import (
    DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT,
    DEFAULT_PLANNER_CACHE_PATH,
    DEFAULT_PLANNER_COMPARE_OUTPUT,
    DEFAULT_PLANNER_PLAN_OUTPUT,
    PLANNER_NAMES,
    audit_cache,
    compare_planners,
    load_queries,
    plan_many,
    write_planner_report,
    write_plans_jsonl,
)


def planner_plan_command(args: Any) -> dict[str, Any]:
    queries = load_queries(
        queries=list(getattr(args, "query", []) or []),
        input_file=getattr(args, "input_file", None),
        dataset_path=getattr(args, "dataset", None),
        split=str(getattr(args, "split", "test")),
        limit=int(getattr(args, "limit", 0)),
    )
    report = plan_many(
        queries,
        planner=str(getattr(args, "planner", "rule")),
        cache_path=getattr(args, "planner_cache", DEFAULT_PLANNER_CACHE_PATH),
        use_cache=not bool(getattr(args, "no_cache", False)),
        config={"command": "planner_plan"},
    )
    output = Path(getattr(args, "output", DEFAULT_PLANNER_PLAN_OUTPUT))
    write_planner_report(output, report)
    jsonl_output = getattr(args, "jsonl_output", None)
    if jsonl_output is not None:
        write_plans_jsonl(Path(jsonl_output), report["plans"])
    return {
        "method": "retrieval_lab_planner_plan",
        "output": str(output),
        "summary": {**report["summary"], "output": str(output)},
    }


def planner_compare_command(args: Any) -> dict[str, Any]:
    queries = load_queries(
        queries=list(getattr(args, "query", []) or []),
        input_file=getattr(args, "input_file", None),
        dataset_path=getattr(args, "dataset", None),
        split=str(getattr(args, "split", "test")),
        limit=int(getattr(args, "limit", 0)),
    )
    planners = parse_planner_names(str(getattr(args, "planners", "rule,multi_query,hyde_card")))
    report = compare_planners(
        queries,
        planners=planners,
        cache_path=getattr(args, "planner_cache", DEFAULT_PLANNER_CACHE_PATH),
        use_cache=not bool(getattr(args, "no_cache", False)),
    )
    output = Path(getattr(args, "output", DEFAULT_PLANNER_COMPARE_OUTPUT))
    write_planner_report(output, report)
    return {
        "method": "retrieval_lab_planner_compare",
        "output": str(output),
        "summary": {**report["summary"], "output": str(output)},
    }


def planner_audit_cache_command(args: Any) -> dict[str, Any]:
    report = audit_cache(getattr(args, "planner_cache", DEFAULT_PLANNER_CACHE_PATH))
    output = Path(getattr(args, "output", DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT))
    write_json(output, report)
    return {
        "method": "retrieval_lab_planner_cache_audit",
        "output": str(output),
        "summary": {**report["summary"], "output": str(output)},
    }


def parse_planner_names(value: str) -> list[str]:
    planners = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [planner for planner in planners if planner not in PLANNER_NAMES]
    if unknown:
        raise ValueError(f"unknown planner(s): {', '.join(unknown)}")
    return planners


__all__ = [
    "planner_audit_cache_command",
    "planner_compare_command",
    "planner_plan_command",
]
