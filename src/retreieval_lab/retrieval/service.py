from __future__ import annotations

from pathlib import Path
import math
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, read_json, write_json
from retreieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retreieval_lab.experiments.runs import cases_from_run_rows, run_artifact_summary
from retreieval_lab.indexes import index_items_from_cases, lexical_tokens, target_item_id
from retreieval_lab.planners import DEFAULT_PLANNER_CACHE_PATH, plan_many


DEFAULT_RETRIEVAL_RUN_OUTPUT = Path(".tmp") / "retrieval_lab" / "retrieval_run_latest.json"
DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT = Path(".tmp") / "retrieval_lab" / "retrieval_legacy_comparison_latest.json"
HARD_FORBIDDEN_STAGE_VETO = 1000.0


def retrieval_run(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    split: str = "test",
    limit: int = 0,
    planner: str = "multi_query",
    planner_cache: Path | None = DEFAULT_PLANNER_CACHE_PATH,
    top_k: int = 10,
    candidate_depth: int = 100,
    run_name: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    cases = read_cases(dataset_path, split=split, limit=limit)
    items = index_items_from_cases(cases)
    plans_report = plan_many(
        [str(case.get("user_input", "")) for case in cases],
        planner=planner,
        cache_path=planner_cache,
        config={"command": "retrieval_run", "split": split, "limit": limit},
    )
    rows = []
    for case, plan in zip(cases, plans_report["plans"], strict=False):
        rows.append(
            retrieve_case(
                case,
                plan=plan,
                items=items,
                top_k=top_k,
                candidate_depth=candidate_depth,
                ranking_key="native_lightweight",
            )
        )
    run_rows = {run_name or f"{planner}::native_lightweight": rows}
    summary = {
        **run_artifact_summary(run_rows, cases_from_run_rows(run_rows)),
        "dataset": str(dataset_path),
        "split": split,
        "limit": limit,
        "planner": planner,
        "top_k": top_k,
        "candidate_depth": candidate_depth,
        "index_item_count": len(items),
        "planner_negative_leak_rate": plans_report["summary"].get("negative_leak_rate", 0.0),
        "target_recall_at_10": target_recall(rows, 10),
        "stage_hit_at_3": stage_hit(rows, 3),
        "style_violation_at_3": style_violation(rows, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    artifact = {
        "method": "retrieval_lab_native_retrieval_run",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_config": {
            "workflow": "native_lightweight",
            "ranking_key": "native_lightweight",
            "query_planner": planner,
            "top_k": top_k,
            "candidate_depth": candidate_depth,
            "constraints_enabled": True,
            "llm_enabled": False,
            "parameters": {"split": split, "limit": limit},
        },
        "planner_summary": plans_report["summary"],
        "run_rows": run_rows,
        "cases": cases_from_run_rows(run_rows),
        "summary": summary,
        "fingerprint": data_sha256({"run_rows": run_rows, "summary": summary}),
    }
    return artifact


def retrieve_case(
    case: dict[str, Any],
    *,
    plan: dict[str, Any],
    items: list[dict[str, Any]],
    top_k: int,
    candidate_depth: int,
    ranking_key: str,
) -> dict[str, Any]:
    scored = [score_item(item, plan=plan) for item in items]
    ranked = sorted(scored, key=lambda row: (-float(row["final_score"]), row["item_id"]))
    all_results = ranked[:candidate_depth]
    top_results = ranked[:top_k]
    expected = expected_target(case)
    expected_id = target_item_id(expected) if expected else ""
    target_rank = next((index for index, item in enumerate(ranked, start=1) if item["item_id"] == expected_id), None)
    return {
        "case_id": case.get("case_id", ""),
        "variant_type": case.get("case_type", "default"),
        "user_input": case.get("user_input", ""),
        "query_plan": plan,
        "query_constraints": {
            "desired_stage": plan.get("desired_stage", []),
            "forbidden_stage": plan.get("forbidden_stage", []),
            "negative_style": plan.get("negative_style", []),
        },
        "target_item_id": expected_id,
        "target_stage": expected.get("script_stage", "") if expected else "",
        "target_purposes": expected.get("creative_purpose", []) if expected else [],
        "target_rank": target_rank,
        "ranking_key": ranking_key,
        "top_results": top_results,
        "all_results": all_results,
    }


def score_item(item: dict[str, Any], *, plan: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    query_texts = [str(plan.get("positive_query", ""))]
    query_texts.extend(str(row.get("text", "")) for row in plan.get("rewrites", []) if isinstance(row, dict))
    if plan.get("hyde_text"):
        query_texts.append(str(plan["hyde_text"]))
    query_tokens = lexical_tokens(" ".join(query_texts))
    item_tokens = item.get("tokens", []) or lexical_tokens(item.get("text", ""))
    lexical = lexical_overlap_score(query_tokens, item_tokens)
    constraint, hits = constraint_score(metadata, plan)
    purpose = purpose_score(metadata, plan)
    signature = lexical_overlap_score(lexical_tokens(signature_text(plan)), item_tokens)
    final_score = lexical + constraint + purpose + signature * 0.25
    result = {
        "item_id": item["item_id"],
        "score": round(final_score, 6),
        "final_score": round(final_score, 6),
        "lexical_score": round(lexical, 6),
        "constraint_score": round(constraint, 6),
        "signature_score": round(signature, 6),
        "metadata": metadata,
        "constraint_hits": hits,
        "explanation": explain_score(lexical, constraint, purpose, signature, hits),
    }
    return result


def lexical_overlap_score(query_tokens: list[str], item_tokens: list[str]) -> float:
    if not query_tokens or not item_tokens:
        return 0.0
    query = set(query_tokens)
    item = set(item_tokens)
    overlap = len(query & item)
    return overlap / math.sqrt(max(1, len(query)) * max(1, len(item)))


def constraint_score(metadata: dict[str, Any], plan: dict[str, Any]) -> tuple[float, dict[str, list[str]]]:
    score = 0.0
    hits: dict[str, list[str]] = {}
    stage = str(metadata.get("script_stage", ""))
    if stage in set(plan.get("forbidden_stage", []) or []):
        hits["forbidden_stage"] = [stage]
        return -HARD_FORBIDDEN_STAGE_VETO, hits
    if stage in set(plan.get("desired_stage", []) or []):
        score += 3.0
        hits["desired_stage"] = [stage]
    return score, hits


def purpose_score(metadata: dict[str, Any], plan: dict[str, Any]) -> float:
    item_purposes = set(str(value) for value in metadata.get("creative_purpose", []) or [])
    query_purposes = set(str(value) for value in plan.get("positive_purposes", []) or [])
    return 0.75 * len(item_purposes & query_purposes)


def signature_text(plan: dict[str, Any]) -> str:
    signature = plan.get("scene_signature", {}) if isinstance(plan.get("scene_signature"), dict) else {}
    parts = [str(signature.get("raw_positive_query", ""))]
    for key in ("people", "place", "actions", "objects", "emotion_function", "narrative_position", "camera_experience"):
        value = signature.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def explain_score(
    lexical: float,
    constraint: float,
    purpose: float,
    signature: float,
    hits: dict[str, list[str]],
) -> str:
    parts = [f"lexical={lexical:.3f}", f"constraint={constraint:.3f}", f"purpose={purpose:.3f}", f"signature={signature:.3f}"]
    if hits:
        parts.append(f"hits={hits}")
    return "; ".join(parts)


def expected_target(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_prefer")
    if isinstance(expected, dict):
        return expected
    target = case.get("target")
    return target if isinstance(target, dict) else {}


def target_recall(rows: list[dict[str, Any]], k: int) -> float:
    hits = sum(1 for row in rows if isinstance(row.get("target_rank"), int) and int(row["target_rank"]) <= k)
    return round(hits / max(1, len(rows)), 6)


def stage_hit(rows: list[dict[str, Any]], k: int) -> float:
    hits = 0
    for row in rows:
        target_stage = str(row.get("target_stage", ""))
        if any(result.get("metadata", {}).get("script_stage") == target_stage for result in row.get("top_results", [])[:k]):
            hits += 1
    return round(hits / max(1, len(rows)), 6)


def style_violation(rows: list[dict[str, Any]], k: int) -> float:
    violations = 0
    for row in rows:
        if any(result.get("constraint_hits", {}).get("negative_style") for result in row.get("top_results", [])[:k]):
            violations += 1
    return round(violations / max(1, len(rows)), 6)


def write_retrieval_run(path: Path, artifact: dict[str, Any]) -> None:
    write_json(path, artifact)


def compare_run_artifacts(native_path: Path, legacy_path: Path) -> dict[str, Any]:
    native = read_json(native_path)
    legacy = read_json(legacy_path)
    native_summary = normalize_metric_names(summary_like(native))
    legacy_summary = normalize_metric_names(summary_like(legacy))
    keys = sorted(set(native_summary) | set(legacy_summary))
    delta = {}
    for key in keys:
        left = native_summary.get(key)
        right = legacy_summary.get(key)
        if isinstance(left, int | float) and isinstance(right, int | float):
            delta[key] = round(float(left) - float(right), 6)
    return {
        "method": "retrieval_lab_compare_legacy_retrieval",
        "summary": {
            "native": str(native_path),
            "legacy": str(legacy_path),
            "numeric_delta_count": len(delta),
        },
        "native_summary": native_summary,
        "legacy_summary": legacy_summary,
        "delta": delta,
    }


def summary_like(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    for key in ("summary", "overall", "graded_metrics"):
        value = report.get(key)
        if isinstance(value, dict):
            return value
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("overall"), dict):
        return metrics["overall"]
    return {}


def normalize_metric_names(summary: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    aliases = {
        "target_recall_at_1": "recall_at_1",
        "target_recall_at_3": "recall_at_3",
        "target_recall_at_10": "recall_at_10",
        "stage_hit_at_1": "desired_stage_hit_at_1",
        "stage_hit_at_3": "desired_stage_hit_at_3",
    }
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


__all__ = [
    "DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT",
    "DEFAULT_RETRIEVAL_RUN_OUTPUT",
    "compare_run_artifacts",
    "retrieval_run",
    "score_item",
    "write_retrieval_run",
]
