from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, write_json
from retrieval_lab.experiments.runs import cases_from_run_rows
from retrieval_lab.qrels import canonical_stage, load_qrels, result_stage


DEFAULT_FAILURE_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "failure_analysis_latest.json"


def analyze_failures_from_runs_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    source = read_json(runs_path)
    run_rows = source.get("run_rows", {}) if isinstance(source, dict) else {}
    if not isinstance(run_rows, dict):
        raise ValueError("runs artifact must contain a run_rows mapping")
    qrels_path = Path(getattr(args, "qrels", ""))
    qrels = load_qrels(qrels_path) if str(qrels_path) else []
    top_k = int(getattr(args, "top_k", 10))
    candidate_depth = int(getattr(args, "candidate_depth", 100))
    failures = analyze_failure_rows(
        {str(name): list(rows) for name, rows in run_rows.items()},
        qrels=qrels,
        top_k=top_k,
        candidate_depth=candidate_depth,
    )
    counts = Counter(row["failure_type"] for row in failures)
    cases = cases_from_run_rows(run_rows)
    total_rows = sum(len(rows) for rows in run_rows.values())
    summary = {
        "run_count": len(run_rows),
        "case_count": len(cases),
        "row_count": total_rows,
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / max(1, total_rows), 6),
        "failure_type_counts": dict(sorted(counts.items())),
        "top_failure_type": counts.most_common(1)[0][0] if counts else None,
        "top_k": top_k,
        "candidate_depth": candidate_depth,
        "qrels_count": len(qrels),
        "has_qrels": bool(qrels),
    }
    report = {
        "method": "retrieval_lab_failure_analysis_from_runs",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_runs": str(runs_path),
        "qrels": str(qrels_path) if str(qrels_path) else "",
        "top_k": top_k,
        "candidate_depth": candidate_depth,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "failures": failures[: int(getattr(args, "max_failures", 200))],
        "fingerprint": data_sha256({"source": source.get("fingerprint"), "qrels": qrels, "failures": failures}),
    }
    output = Path(getattr(args, "output", DEFAULT_FAILURE_REPORT_PATH))
    write_json(output, report)
    return {
        "method": "retrieval_lab_analyze_failures_from_runs",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": report["fingerprint"]},
    }


def analyze_failure_rows(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    qrels: list[dict[str, Any]] | None = None,
    top_k: int,
    candidate_depth: int,
) -> list[dict[str, Any]]:
    qrel_map = {(str(row["query_id"]), str(row["item_id"])): int(row["grade"]) for row in qrels or []}
    failures = []
    for run_name, rows in run_rows.items():
        for row in rows:
            if not row_is_failure(row, qrel_map=qrel_map, top_k=top_k):
                continue
            failures.append(
                failure_analysis_row_from_artifact(
                    row,
                    run_name=run_name,
                    qrel_map=qrel_map,
                    top_k=top_k,
                    candidate_depth=candidate_depth,
                )
            )
    failures.sort(key=lambda row: (row["failure_type"], row["run_name"], row["case_id"]))
    return failures


def row_is_failure(row: dict[str, Any], *, qrel_map: dict[tuple[str, str], int], top_k: int) -> bool:
    target_rank = row.get("target_rank")
    if target_rank is None or int(target_rank) > top_k:
        return True
    if not qrel_map:
        return False
    query_id = str(row.get("case_id", ""))
    top_results = row.get("top_results", [])[:top_k]
    relevant_in_qrels = any(grade >= 2 for (q_id, _item_id), grade in qrel_map.items() if q_id == query_id)
    if not relevant_in_qrels or not top_results:
        return False
    best_top_grade = max(qrel_map.get((query_id, str(result.get("item_id"))), 0) for result in top_results)
    return best_top_grade < 2


def failure_analysis_row_from_artifact(
    row: dict[str, Any],
    *,
    run_name: str,
    qrel_map: dict[tuple[str, str], int],
    top_k: int,
    candidate_depth: int,
) -> dict[str, Any]:
    top_results = row.get("top_results", [])
    top1 = top_results[0] if top_results else {}
    target_result, target_position = find_target_result(row)
    failure_type = classify_failure_from_artifact(
        row,
        top1=top1,
        qrel_map=qrel_map,
        top_k=top_k,
        candidate_depth=candidate_depth,
        target_result=target_result,
        target_position=target_position,
    )
    target_scores = component_scores(target_result or {})
    top1_scores = component_scores(top1)
    return {
        "run_name": run_name,
        "case_id": row.get("case_id"),
        "user_input": row.get("user_input", ""),
        "ranking_key": row.get("ranking_key"),
        "failure_type": failure_type,
        "target_item_id": row.get("target_item_id"),
        "target_rank": row.get("target_rank"),
        "target_in_candidate_depth": target_in_candidate_depth(row, candidate_depth=candidate_depth),
        "target_position_in_artifact": target_position,
        "target_qrel_grade": qrel_grade(qrel_map, row, row.get("target_item_id")),
        "target_scores": target_scores,
        "top1_item_id": top1.get("item_id"),
        "top1_qrel_grade": qrel_grade(qrel_map, row, top1.get("item_id")),
        "top1_scores": top1_scores,
        "top1_metadata": top1.get("metadata", {}),
        "top1_reason": explain_top1_win_from_artifact(row, top1, target_scores, top1_scores),
        "qrels_relevant_count": relevant_qrel_count(qrel_map, str(row.get("case_id", ""))),
        "judged_top_k": judged_top_k(row, qrel_map, top_k=top_k),
        "suggested_next_action": suggested_next_action(failure_type),
    }


def classify_failure_from_artifact(
    row: dict[str, Any],
    *,
    top1: dict[str, Any],
    qrel_map: dict[tuple[str, str], int],
    top_k: int,
    candidate_depth: int,
    target_result: dict[str, Any] | None,
    target_position: int | None = None,
) -> str:
    target_rank = row.get("target_rank")
    candidate_rank = target_position if target_position is not None else target_rank
    if candidate_rank is None or int(candidate_rank) > candidate_depth:
        return "candidate_recall_failure"
    if top1.get("constraint_hits", {}).get("negative_style"):
        return "style_risk_miss"
    if top1.get("constraint_hits", {}).get("forbidden_stage"):
        return "constraint_failure"
    if has_ambiguous_valid_answer(row, top1=top1, qrel_map=qrel_map):
        return "ambiguous_multi_valid_answer"
    if query_understanding_failed(row, top1=top1):
        return "query_understanding_failure"
    if target_rank and int(target_rank) <= top_k:
        return "fusion_ranking_failure"
    if target_result is not None and component_scores(target_result).get("semantic", 0.0) <= 0:
        return "candidate_recall_failure"
    if candidate_rank and int(candidate_rank) <= candidate_depth:
        return "fusion_ranking_failure"
    return "weak_target_label"


def has_ambiguous_valid_answer(
    row: dict[str, Any],
    *,
    top1: dict[str, Any],
    qrel_map: dict[tuple[str, str], int],
) -> bool:
    query_id = str(row.get("case_id", ""))
    if qrel_map:
        top1_grade = qrel_map.get((query_id, str(top1.get("item_id"))), 0)
        target_grade = qrel_map.get((query_id, str(row.get("target_item_id"))), 0)
        relevant_count = relevant_qrel_count(qrel_map, query_id)
        if top1.get("item_id") != row.get("target_item_id") and top1_grade >= 2 and target_grade >= 2:
            return True
        if relevant_count >= 3 and top1_grade >= 2:
            return True
    target_purposes = set(row.get("target_purposes", []))
    metadata = top1.get("metadata", {}) if isinstance(top1.get("metadata"), dict) else {}
    purposes = set(metadata.get("creative_purpose", []))
    return bool(
        top1.get("item_id") != row.get("target_item_id")
        and result_stage(top1) == row.get("target_stage")
        and target_purposes
        and target_purposes & purposes
    )


def query_understanding_failed(row: dict[str, Any], *, top1: dict[str, Any]) -> bool:
    query_plan = row.get("query_plan", {}) if isinstance(row.get("query_plan"), dict) else {}
    ambiguity = query_plan.get("ambiguity", {}) if isinstance(query_plan.get("ambiguity"), dict) else {}
    if ambiguity.get("level") == "high":
        return True
    if float(row.get("planner_confidence", 1.0) or 1.0) < 0.6:
        return True
    target_stage = canonical_stage(row.get("target_stage", ""))
    if target_stage and top1 and result_stage(top1) != target_stage and not row.get("stage_hit_at_3", False):
        return True
    return False


def explain_top1_win_from_artifact(
    row: dict[str, Any],
    top1: dict[str, Any],
    target_scores: dict[str, float],
    top1_scores: dict[str, float],
) -> str:
    if top1.get("constraint_hits", {}).get("negative_style"):
        return "top1 still has negative style hits"
    if top1.get("constraint_hits", {}).get("forbidden_stage"):
        return "top1 violates forbidden stage constraint"
    if result_stage(top1) == row.get("target_stage") and top1_scores.get("signature", 0.0) >= target_scores.get(
        "signature",
        0.0,
    ):
        return "top1 matches target stage and has equal or better signature score"
    if top1_scores.get("rrf", 0.0) > target_scores.get("rrf", 0.0):
        return "top1 won by RRF fusion score"
    if top1_scores.get("semantic", 0.0) > target_scores.get("semantic", 0.0):
        return "top1 won by semantic score"
    return "top1 score margin is small or target label may be ambiguous"


def component_scores(result: dict[str, Any]) -> dict[str, float]:
    components = result.get("workflow_score_components", {}) if isinstance(result.get("workflow_score_components"), dict) else {}
    return {
        "semantic": numeric(result, "embedding_score", components, "semantic"),
        "lexical": numeric(result, "lexical_score", components, "lexical"),
        "rrf": numeric(result, "rrf_score", components, "rrf"),
        "constraint": numeric(result, "constraint_score", components, "constraint"),
        "signature": numeric(result, "signature_score", components, "signature"),
        "score": numeric(result, "score", components, "base_score"),
    }


def numeric(result: dict[str, Any], field: str, components: dict[str, Any], component_field: str) -> float:
    value = result.get(field, components.get(component_field, 0.0))
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def find_target_result(row: dict[str, Any]) -> tuple[dict[str, Any] | None, int | None]:
    target_id = row.get("target_item_id")
    for index, result in enumerate(row.get("all_results") or row.get("top_results") or [], start=1):
        if result.get("item_id") == target_id:
            return result, index
    return None, None


def target_in_candidate_depth(row: dict[str, Any], *, candidate_depth: int) -> bool:
    _target, position = find_target_result(row)
    if position is not None:
        return position <= candidate_depth
    rank = row.get("target_rank")
    if rank is None:
        return False
    return int(rank) <= candidate_depth


def qrel_grade(qrel_map: dict[tuple[str, str], int], row: dict[str, Any], item_id: Any) -> int | None:
    if not qrel_map or item_id is None:
        return None
    return qrel_map.get((str(row.get("case_id", "")), str(item_id)), 0)


def relevant_qrel_count(qrel_map: dict[tuple[str, str], int], query_id: str) -> int:
    return sum(1 for (q_id, _item_id), grade in qrel_map.items() if q_id == query_id and grade >= 2)


def judged_top_k(row: dict[str, Any], qrel_map: dict[tuple[str, str], int], *, top_k: int) -> float:
    if not qrel_map:
        return 0.0
    results = row.get("top_results", [])[:top_k]
    if not results:
        return 0.0
    query_id = str(row.get("case_id", ""))
    judged = sum(1 for result in results if (query_id, str(result.get("item_id"))) in qrel_map)
    return round(judged / len(results), 6)


def suggested_next_action(failure_type: str) -> str:
    return {
        "query_understanding_failure": "inspect planner output and add ambiguous query understanding cases",
        "candidate_recall_failure": "improve recall pool or query expansion before reranking",
        "fusion_ranking_failure": "tune fusion/signature weights or test.md reranker",
        "constraint_failure": "tighten hard filters and constraint evidence",
        "style_risk_miss": "expand style risk mining and negative-style penalties",
        "ambiguous_multi_valid_answer": "use graded qrels and avoid single-target-only scoring",
        "weak_target_label": "audit target label and add qrels votes",
    }.get(failure_type, "inspect case manually")


__all__ = [
    "DEFAULT_FAILURE_REPORT_PATH",
    "analyze_failure_rows",
    "analyze_failures_from_runs_command",
    "classify_failure_from_artifact",
    "component_scores",
    "explain_top1_win_from_artifact",
    "failure_analysis_row_from_artifact",
    "row_is_failure",
]
