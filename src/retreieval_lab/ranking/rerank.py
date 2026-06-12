from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, read_json, write_json
from retreieval_lab.experiments.runs import cases_from_run_rows, run_artifact_summary, unique_run_name
from retreieval_lab.qrels import canonical_stage, load_qrels


DEFAULT_RERANKED_RUN_ARTIFACT_PATH = Path(".tmp") / "retrieval_lab" / "reranked_run_rows_latest.json"


def rerank_run_artifact_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    method = str(getattr(args, "method", "rule"))
    rerank_depth = int(getattr(args, "rerank_depth", 20))
    top_k = int(getattr(args, "top_k", 10))
    source = read_json(runs_path)
    run_rows = source.get("run_rows", {}) if isinstance(source, dict) else {}
    if not isinstance(run_rows, dict):
        raise ValueError("runs artifact must contain a run_rows mapping")
    qrels = []
    if method in {"qrels_oracle", "qrels-oracle"}:
        qrels_path = Path(getattr(args, "qrels", ""))
        if not qrels_path:
            raise ValueError("--qrels is required for qrels_oracle rerank")
        qrels = load_qrels(qrels_path)
    reranked = rerank_run_rows(
        {str(name): list(rows) for name, rows in run_rows.items()},
        method=method,
        qrels=qrels,
        rerank_depth=rerank_depth,
        top_k=top_k,
    )
    cases = cases_from_run_rows(reranked)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        **run_artifact_summary(reranked, cases),
        "method": method,
        "source_runs": str(runs_path),
        "rerank_depth": rerank_depth,
        "top_k": top_k,
        "elapsed_seconds": elapsed_seconds,
    }
    artifact = {
        "method": "retrieval_lab_reranked_run_artifact",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rerank_method": method,
        "rerank_depth": rerank_depth,
        "top_k": top_k,
        "source_runs": str(runs_path),
        "run_rows": reranked,
        "cases": cases,
        "summary": summary,
        "fingerprint": data_sha256(reranked),
    }
    output = Path(getattr(args, "output", DEFAULT_RERANKED_RUN_ARTIFACT_PATH))
    write_json(output, artifact)
    return {
        "method": "retrieval_lab_rerank_run_artifact",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": artifact["fingerprint"]},
    }


def rerank_run_rows(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    method: str,
    qrels: list[dict[str, Any]] | None = None,
    rerank_depth: int,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for run_name, rows in run_rows.items():
        if method == "rule":
            reranked_rows = [rerank_row_by_rule(row, rerank_depth=rerank_depth, top_k=top_k) for row in rows]
            output_name = f"{run_name}::rule_rerank@{rerank_depth}"
        elif method in {"qrels_oracle", "qrels-oracle"}:
            reranked_rows = [
                rerank_row_by_qrels(row, qrels or [], rerank_depth=rerank_depth, top_k=top_k)
                for row in rows
            ]
            output_name = f"{run_name}::qrels_oracle_rerank@{rerank_depth}"
        else:
            raise ValueError(f"unknown rerank method: {method}")
        result[unique_run_name(result, output_name)] = reranked_rows
    return result


def rerank_row_by_rule(row: dict[str, Any], *, rerank_depth: int, top_k: int) -> dict[str, Any]:
    candidates = row.get("top_results", [])[:rerank_depth]
    reranked = sorted(candidates, key=rule_rerank_score, reverse=True)
    return row_with_reranked_results(row, reranked[:top_k], ranking_key=f"rule_rerank@{rerank_depth}")


def rule_rerank_score(result: dict[str, Any]) -> float:
    hits = result.get("constraint_hits", {})
    score = float(result.get("score", 0.0))
    score += 0.45 * float(result.get("signature_score", 0.0))
    score += 0.25 * float(result.get("constraint_score", 0.0))
    if hits.get("desired_stage"):
        score += 0.15
    if hits.get("positive_style"):
        score += 0.1 * len(hits["positive_style"])
    if hits.get("negative_style"):
        score -= 0.5 * len(hits["negative_style"])
    return score


def rerank_row_by_qrels(
    row: dict[str, Any],
    qrels: list[dict[str, Any]],
    *,
    rerank_depth: int,
    top_k: int,
) -> dict[str, Any]:
    qrel_map = {(qrel["query_id"], qrel["item_id"]): int(qrel["grade"]) for qrel in qrels}
    query_id = row["case_id"]
    candidates = row.get("top_results", [])[:rerank_depth]
    reranked = sorted(
        candidates,
        key=lambda result: (qrel_map.get((query_id, result["item_id"]), 0), float(result.get("score", 0.0))),
        reverse=True,
    )
    return row_with_reranked_results(row, reranked[:top_k], ranking_key=f"qrels_oracle_rerank@{rerank_depth}")


def row_with_reranked_results(row: dict[str, Any], top_results: list[dict[str, Any]], *, ranking_key: str) -> dict[str, Any]:
    target_id = row.get("target_item_id")
    target_rank = None
    copied_results = []
    for rank, result in enumerate(top_results, start=1):
        result = dict(result)
        result["ranking_key"] = ranking_key
        copied_results.append(result)
        if result.get("item_id") == target_id:
            target_rank = rank
    target_stage = row.get("target_stage")
    target_purposes = set(row.get("target_purposes", []))
    return {
        **row,
        "ranking_key": ranking_key,
        "target_rank": target_rank,
        "target_score": score_of(top_results, target_id),
        "stage_hit_at_1": bool(copied_results and result_stage(copied_results[0]) == target_stage),
        "stage_hit_at_3": any(result_stage(result) == target_stage for result in copied_results[:3]),
        "purpose_hit_at_3": purpose_hit_at(copied_results, target_purposes, 3),
        "top_results": copied_results,
    }


def result_stage(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    return canonical_stage(metadata.get("script_stage", ""))


def purpose_hit_at(results: list[dict[str, Any]], target_purposes: set[str], k: int) -> bool:
    if not target_purposes:
        return False
    for result in results[:k]:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
        purposes = set(metadata.get("creative_purpose", []))
        if target_purposes & purposes:
            return True
    return False


def score_of(rows: list[dict[str, Any]], item_id: str | None) -> float | None:
    if item_id is None:
        return None
    for row in rows:
        if row["item_id"] == item_id:
            return row["score"]
    return None


__all__ = [
    "DEFAULT_RERANKED_RUN_ARTIFACT_PATH",
    "purpose_hit_at",
    "rerank_row_by_qrels",
    "rerank_row_by_rule",
    "rerank_run_artifact_command",
    "rerank_run_rows",
    "result_stage",
    "row_with_reranked_results",
    "rule_rerank_score",
    "score_of",
]
