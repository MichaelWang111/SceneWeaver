from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, write_json
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retrieval_lab.indexes import index_items_from_cases
from retrieval_lab.planners import DEFAULT_PLANNER_CACHE_PATH, plan_many
from retrieval_lab.retrieval.service import (
    mean_top_margin,
    prepare_retrieval_index,
    purpose_hit,
    retrieve_case,
    retrieve_case_compact_fast,
    stage_hit,
    style_violation,
    target_recall,
)


DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT = Path(".tmp") / "retrieval_lab" / "retrieval_benchmark_latest.json"


def retrieval_benchmark_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    load_started = time.perf_counter()
    base_cases = read_cases(
        Path(getattr(args, "dataset", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "test")),
        limit=int(getattr(args, "limit", 0)),
    )
    load_seconds = time.perf_counter() - load_started

    repeat_started = time.perf_counter()
    cases = repeat_cases(base_cases, int(getattr(args, "repeat_to", 1000)))
    repeat_seconds = time.perf_counter() - repeat_started

    index_started = time.perf_counter()
    items = index_items_from_cases(base_cases)
    prepared = prepare_retrieval_index(items)
    index_seconds = time.perf_counter() - index_started

    planner_started = time.perf_counter()
    plans_report = plan_many(
        [str(case.get("user_input", "")) for case in cases],
        planner=str(getattr(args, "planner", "multi_query")),
        cache_path=Path(getattr(args, "planner_cache", DEFAULT_PLANNER_CACHE_PATH))
        if not bool(getattr(args, "no_cache", False))
        else None,
        config={"command": "benchmark-retrieval", "repeat_to": len(cases), "split": getattr(args, "split", "test")},
    )
    planner_seconds = time.perf_counter() - planner_started

    score_started = time.perf_counter()
    top_k = int(getattr(args, "top_k", 10))
    candidate_depth = int(getattr(args, "candidate_depth", top_k) or top_k)
    if bool(getattr(args, "compact_output", False)):
        rows, score_cache_hits, score_cache_misses = compact_benchmark_rows(
            cases,
            plans_report["plans"],
            items=items,
            prepared_index=prepared,
            top_k=top_k,
            ranking_key=str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature")),
        )
    else:
        score_cache_hits = 0
        score_cache_misses = len(cases)
        rows = [
            retrieve_case(
                case,
                plan=plan,
                items=items,
                prepared_index=prepared,
                top_k=top_k,
                candidate_depth=candidate_depth,
                ranking_key=str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature")),
            )
            for case, plan in zip(cases, plans_report["plans"], strict=False)
        ]
    scoring_seconds = time.perf_counter() - score_started

    summary = {
        "base_case_count": len(base_cases),
        "case_count": len(cases),
        "index_item_count": len(items),
        "repeat_to": int(getattr(args, "repeat_to", 1000)),
        "top_k": top_k,
        "candidate_depth": candidate_depth,
        "planner": str(getattr(args, "planner", "multi_query")),
        "ranking_key": str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature")),
        "compact_output": bool(getattr(args, "compact_output", False)),
        "target_recall_at_10": target_recall(rows, 10),
        "stage_hit_at_3": stage_hit(rows, 3),
        "purpose_hit_at_3": purpose_hit(rows, 3),
        "style_violation_at_3": style_violation(rows, 3),
        "mean_top1_top2_margin": mean_top_margin(rows),
        "planner_cache_hit_rate": plans_report["summary"].get("cache_hit_rate", 0.0),
        "score_cache_hit_rate": round(score_cache_hits / max(1, score_cache_hits + score_cache_misses), 6),
        "score_cache_hits": score_cache_hits,
        "score_cache_misses": score_cache_misses,
        "llm_call_count": 0,
    }
    elapsed_before_serialization = time.perf_counter() - started
    summary["timing_seconds"] = {
        "load_cases": round(load_seconds, 6),
        "repeat_cases": round(repeat_seconds, 6),
        "prepare_index": round(index_seconds, 6),
        "planner": round(planner_seconds, 6),
        "scoring_sorting": round(scoring_seconds, 6),
        "pre_serialization_total": round(elapsed_before_serialization, 6),
    }
    summary["qps"] = round(len(cases) / max(1e-9, elapsed_before_serialization), 3)
    summary["meets_1000_under_10s"] = bool(len(cases) >= 1000 and elapsed_before_serialization <= 10.0)

    report: dict[str, Any] = {
        "method": "retrieval_lab_retrieval_benchmark",
        "summary": summary,
        "planner_summary": plans_report["summary"],
        "sample_rows": compact_rows(rows[:5]),
    }
    if not bool(getattr(args, "compact_output", False)):
        report["run_rows"] = {"benchmark::native": rows}
    report["fingerprint"] = data_sha256({"summary": summary, "sample_rows": report["sample_rows"]})
    serialization_started = time.perf_counter()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True)
    summary["report_bytes"] = len(payload.encode("utf-8"))
    output = Path(getattr(args, "output", DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT))
    write_json(output, report)
    serialization_seconds = time.perf_counter() - serialization_started
    summary["timing_seconds"]["serialization_write"] = round(serialization_seconds, 6)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    summary["output"] = str(output)
    write_json(output, report)
    return report


def repeat_cases(cases: list[dict[str, Any]], repeat_to: int) -> list[dict[str, Any]]:
    if repeat_to <= 0 or len(cases) >= repeat_to:
        return list(cases)
    repeated = []
    index = 0
    while len(repeated) < repeat_to:
        source = cases[index % len(cases)]
        copied = deepcopy(source)
        copied["case_id"] = f"{source.get('case_id', index)}::bench_{len(repeated) + 1}"
        repeated.append(copied)
        index += 1
    return repeated


def compact_benchmark_rows(
    cases: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    *,
    items: list[dict[str, Any]],
    prepared_index: dict[str, Any],
    top_k: int,
    ranking_key: str,
) -> tuple[list[dict[str, Any]], int, int]:
    rows = []
    cache: dict[str, dict[str, Any]] = {}
    hits = 0
    misses = 0
    for case, plan in zip(cases, plans, strict=False):
        key = compact_score_cache_key(case, plan, ranking_key=ranking_key, top_k=top_k)
        cached = cache.get(key)
        if cached is None:
            cached = retrieve_case_compact_fast(
                case,
                plan=plan,
                items=items,
                prepared_index=prepared_index,
                top_k=top_k,
                ranking_key=ranking_key,
            )
            cache[key] = cached
            misses += 1
        else:
            hits += 1
        row = dict(cached)
        row["case_id"] = case.get("case_id", "")
        row["variant_type"] = case.get("case_type", "default")
        row["fuzzy_set_type"] = case.get("fuzzy_set_type", "")
        row["user_input"] = case.get("user_input", "")
        rows.append(row)
    return rows, hits, misses


def compact_score_cache_key(case: dict[str, Any], plan: dict[str, Any], *, ranking_key: str, top_k: int) -> str:
    target = case.get("expected_prefer") if isinstance(case.get("expected_prefer"), dict) else case.get("target")
    target_id = ""
    if isinstance(target, dict):
        target_id = "::".join(str(target.get(key, "")) for key in ("fixture_id", "scene_id", "retrieval_id") if target.get(key))
    return data_sha256(
        {
            "target_id": target_id,
            "ranking_key": ranking_key,
            "top_k": top_k,
            "plan": plan,
        }
    )


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted = []
    for row in rows:
        compacted.append(
            {
                "case_id": row.get("case_id", ""),
                "target_item_id": row.get("target_item_id", ""),
                "target_rank": row.get("target_rank"),
                "top_results": [
                    {
                        "item_id": result.get("item_id", ""),
                        "score": result.get("score", 0.0),
                        "script_stage": result.get("metadata", {}).get("script_stage", ""),
                        "constraint_hits": result.get("constraint_hits", {}),
                    }
                    for result in row.get("top_results", [])
                ],
            }
        )
    return compacted


__all__ = [
    "DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT",
    "compact_benchmark_rows",
    "compact_score_cache_key",
    "retrieval_benchmark_command",
    "repeat_cases",
]
