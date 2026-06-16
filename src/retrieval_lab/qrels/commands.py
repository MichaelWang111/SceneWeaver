from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import subprocess
import time
from typing import Any

from retrieval_lab.qrels.service import (
    active_qrels_samples,
    load_qrels,
    merge_adjudicated_qrels,
    pooled_qrels_from_run_rows,
    pooled_qrels_summary,
    qrel_confidence,
    qrel_relevance_vote,
    qrel_is_bootstrap_only,
    qrel_needs_adjudication,
    qrel_vote_judge_type,
    qrels_audit_summary,
    qrels_trust_level,
    reranker_candidate_summary,
    write_qrels,
)
from retrieval_lab.evaluators import graded_metrics


DEFAULT_POOLED_QRELS_PATH = Path(".tmp") / "pooled_qrels_next.jsonl"
DEFAULT_QRELS_AUDIT_REPORT_PATH = Path(".tmp") / "qrels_audit_next.json"
DEFAULT_ADJUDICATED_QRELS_PATH = Path(".tmp") / "pooled_qrels_adjudicated.jsonl"
DEFAULT_ACTIVE_QRELS_SAMPLE_PATH = Path(".tmp") / "active_qrels_next.jsonl"
DEFAULT_COVERAGE_QRELS_SAMPLE_PATH = Path(".tmp") / "coverage_qrels_next.jsonl"
DEFAULT_POOLED_QRELS_REPORT_PATH = Path(".tmp") / "pooled_qrels_next_report.json"


def audit_qrels_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    qrels_path = Path(getattr(args, "qrels", DEFAULT_POOLED_QRELS_PATH))
    qrels = load_qrels(qrels_path)
    summary = qrels_audit_summary(qrels)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    report = {
        "method": "retrieval_lab_qrels_audit",
        "qrels": str(qrels_path),
        "qrels_count": len(qrels),
        "elapsed_seconds": elapsed_seconds,
        "summary": summary,
        "low_confidence_examples": [row for row in qrels if qrel_confidence(row) < 0.6][:50],
    }
    report["experiment"] = {
        "command": getattr(args, "command", "audit-qrels"),
        "config": serializable_args(args),
        "git_sha": git_sha(),
        "elapsed_seconds": elapsed_seconds,
        "summary": summary,
    }
    return report


def merge_adjudicated_qrels_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    qrels_path = Path(getattr(args, "qrels", DEFAULT_POOLED_QRELS_PATH))
    adjudications_path = Path(getattr(args, "adjudications"))
    output_path = Path(getattr(args, "output", DEFAULT_ADJUDICATED_QRELS_PATH))
    existing_qrels = load_qrels(qrels_path)
    adjudication_votes = load_adjudication_votes(
        adjudications_path,
        default_judge_type=getattr(args, "default_judge_type", "human"),
        default_judge_id=getattr(args, "default_judge_id", "adjudicator"),
        judge_version=getattr(args, "judge_version", "v1"),
    )
    merged_qrels = merge_adjudicated_qrels(existing_qrels, adjudication_votes)
    write_qrels(output_path, merged_qrels)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        **qrels_audit_summary(merged_qrels),
        "input_qrels_count": len(existing_qrels),
        "adjudication_vote_count": len(adjudication_votes),
        "output": str(output_path),
        "elapsed_seconds": elapsed_seconds,
    }
    report = {
        "method": "retrieval_lab_merge_adjudicated_qrels",
        "qrels": str(qrels_path),
        "adjudications": str(adjudications_path),
        "output": str(output_path),
        "elapsed_seconds": elapsed_seconds,
        "summary": summary,
        "changed_examples": [
            row
            for row in merged_qrels
            if any(qrel_vote_judge_type(vote, row) in {"human", "llm"} for vote in row.get("grade_votes", []))
        ][:50],
    }
    report["experiment"] = {
        "command": getattr(args, "command", "merge-adjudicated-qrels"),
        "config": serializable_args(args),
        "git_sha": git_sha(),
        "elapsed_seconds": elapsed_seconds,
        "summary": summary,
    }
    return report


def pool_qrels_from_runs_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    qrels_output = Path(getattr(args, "qrels_output", DEFAULT_POOLED_QRELS_PATH))
    report_output = Path(getattr(args, "report_output", DEFAULT_POOLED_QRELS_REPORT_PATH))
    payload = load_run_rows_payload(runs_path)
    run_rows = payload["run_rows"]
    cases = payload["cases"] or cases_from_run_rows(run_rows)
    qrels = pooled_qrels_from_run_rows(run_rows)
    write_qrels(qrels_output, qrels)
    baseline_run = getattr(args, "baseline_run", "") or next(iter(run_rows), "")
    baseline_rows = run_rows.get(baseline_run, [])
    metrics = graded_metrics(baseline_rows, qrels, top_k=int(getattr(args, "top_k", 10))) if baseline_rows else {}
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        **metrics,
        **pooled_qrels_summary(qrels, cases, run_rows),
        "baseline_run": baseline_run,
        "qrels_output": str(qrels_output),
        "qrels_trust_level": qrels_trust_level(qrels),
        "elapsed_seconds": elapsed_seconds,
    }
    report = {
        "method": "retrieval_lab_pool_qrels_from_runs",
        "runs": str(runs_path),
        "baseline_run": baseline_run,
        "qrels_output": str(qrels_output),
        "elapsed_seconds": elapsed_seconds,
        "graded_metrics": metrics,
        "summary": summary,
        "experiment": {
            "command": getattr(args, "command", "pool-qrels-from-runs"),
            "config": serializable_args(args),
            "git_sha": git_sha(),
            "elapsed_seconds": elapsed_seconds,
            "summary": summary,
        },
    }
    write_json(report_output, report)
    return report


def sample_active_qrels_from_runs_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    output = Path(getattr(args, "output", DEFAULT_ACTIVE_QRELS_SAMPLE_PATH))
    payload = load_run_rows_payload(runs_path)
    run_rows = payload["run_rows"]
    existing_qrels_path = Path(getattr(args, "qrels", DEFAULT_POOLED_QRELS_PATH))
    existing_qrels = load_qrels(existing_qrels_path) if existing_qrels_path.exists() else []
    samples = active_qrels_samples(
        run_rows,
        existing_qrels=existing_qrels,
        sample_size=int(getattr(args, "sample_size", 80)),
        include_judged=bool(getattr(args, "include_judged", False)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        "sample_count": len(samples),
        "case_count": len(cases_from_run_rows(run_rows)),
        "run_count": len(run_rows),
        "existing_qrels_count": len(existing_qrels),
        "include_judged": bool(getattr(args, "include_judged", False)),
        "reason_counts": dict(Counter(reason for row in samples for reason in row.get("reasons", []))),
        "output": str(output),
        "elapsed_seconds": elapsed_seconds,
    }
    return {
        "method": "retrieval_lab_active_qrels_sampler_from_runs",
        "runs": str(runs_path),
        "qrels": str(existing_qrels_path),
        "output": str(output),
        "summary": summary,
        "examples": samples[:20],
        "experiment": {
            "command": getattr(args, "command", "sample-active-qrels-from-runs"),
            "config": serializable_args(args),
            "git_sha": git_sha(),
            "elapsed_seconds": elapsed_seconds,
            "summary": summary,
        },
    }


def sample_coverage_qrels_from_runs_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    output = Path(getattr(args, "output", DEFAULT_COVERAGE_QRELS_SAMPLE_PATH))
    payload = load_run_rows_payload(runs_path)
    run_rows = payload["run_rows"]
    existing_qrels_path = Path(getattr(args, "qrels", DEFAULT_POOLED_QRELS_PATH))
    existing_qrels = load_qrels(existing_qrels_path) if existing_qrels_path.exists() else []
    requested_sample_size = int(getattr(args, "sample_size", 0) or 0)
    sample_size = requested_sample_size or reviewed_gap_to_high(existing_qrels)
    samples = coverage_qrels_samples(run_rows, existing_qrels=existing_qrels, sample_size=sample_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    current_reviewed = reviewed_qrels_count(existing_qrels)
    projected_reviewed = current_reviewed + len(samples)
    summary = {
        "sample_count": len(samples),
        "case_count": len(cases_from_run_rows(run_rows)),
        "run_count": len(run_rows),
        "existing_qrels_count": len(existing_qrels),
        "bootstrap_only_qrels_count": sum(1 for row in existing_qrels if qrel_is_bootstrap_only(row)),
        "current_reviewed_count": current_reviewed,
        "high_reviewed_threshold": high_reviewed_threshold(existing_qrels),
        "additional_reviewed_needed_for_high": reviewed_gap_to_high(existing_qrels),
        "requested_sample_size": requested_sample_size,
        "projected_reviewed_count": projected_reviewed,
        "projected_reviewed_rate": round(projected_reviewed / max(1, len(existing_qrels)), 6),
        "current_qrels_trust_level": qrels_trust_level(existing_qrels),
        "reason_counts": dict(Counter(reason for row in samples for reason in row.get("reasons", []))),
        "output": str(output),
        "elapsed_seconds": elapsed_seconds,
    }
    return {
        "method": "retrieval_lab_coverage_qrels_sampler_from_runs",
        "runs": str(runs_path),
        "qrels": str(existing_qrels_path),
        "output": str(output),
        "summary": summary,
        "examples": samples[:20],
        "experiment": {
            "command": getattr(args, "command", "sample-coverage-qrels-from-runs"),
            "config": serializable_args(args),
            "git_sha": git_sha(),
            "elapsed_seconds": elapsed_seconds,
            "summary": summary,
        },
    }


def coverage_qrels_samples(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    existing_qrels: list[dict[str, Any]],
    sample_size: int,
) -> list[dict[str, Any]]:
    query_info, candidates, query_candidates, ranks = run_context_index(run_rows)
    rows = []
    for qrel in existing_qrels:
        if not qrel_is_bootstrap_only(qrel):
            continue
        query_id = str(qrel.get("query_id", ""))
        item_id = str(qrel.get("item_id", ""))
        info = query_info.get(query_id, {})
        candidate = candidates.get((query_id, item_id), {})
        item_ranks = ranks.get((query_id, item_id), [])
        reasons = coverage_sample_reasons(qrel, candidate, item_ranks)
        top_candidates = sorted(
            query_candidates.get(query_id, []),
            key=lambda row: (int(row.get("rank") or 999999), str(row.get("item_id", ""))),
        )[:5]
        row = {
            "query_id": query_id,
            "item_id": item_id,
            "query": info.get("user_input", ""),
            "user_input": info.get("user_input", ""),
            "target_item_id": info.get("target_item_id", ""),
            "target": {
                "item_id": info.get("target_item_id", ""),
                "stage": info.get("target_stage", ""),
                "purposes": info.get("target_purposes", []),
            },
            "metadata": candidate.get("metadata", {}),
            "constraint_hits": candidate.get("constraint_hits", {}),
            "scores": candidate.get("scores", {}),
            "pooled_from": candidate.get("pooled_from", qrel.get("pooled_from", [])),
            "candidate_summary": candidate.get("summary", {"item_id": item_id}),
            "top_candidates": top_candidates,
            "existing_qrel": compact_coverage_qrel(qrel),
            "suggested_grade": int(qrel.get("grade", 0)),
            "suggested_reason": str(qrel.get("reason", "")),
            "suggested_granularity": "existing_qrel_coverage",
            "reasons": reasons,
            "priority": coverage_sample_priority(qrel, reasons, item_ranks),
            "judged": False,
            "coverage_only": True,
        }
        rows.append(row)
    rows.sort(key=lambda row: (-float(row.get("priority", 0.0)), str(row.get("query_id", "")), str(row.get("item_id", ""))))
    if sample_size <= 0:
        return rows
    return rows[:sample_size]


def run_context_index(
    run_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[int]]]:
    query_info: dict[str, dict[str, Any]] = {}
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    query_candidates: dict[str, list[dict[str, Any]]] = {}
    ranks: dict[tuple[str, str], list[int]] = {}
    for run_name, rows in run_rows.items():
        for row in rows:
            query_id = str(row.get("case_id", ""))
            if not query_id:
                continue
            query_info.setdefault(
                query_id,
                {
                    "user_input": row.get("user_input", ""),
                    "target_item_id": row.get("target_item_id", ""),
                    "target_stage": row.get("target_stage", ""),
                    "target_purposes": list(row.get("target_purposes", [])),
                },
            )
            for rank, result in enumerate(row.get("top_results", []), start=1):
                item_id = str(result.get("item_id", ""))
                if not item_id:
                    continue
                key = (query_id, item_id)
                candidate = candidates.setdefault(
                    key,
                    {
                        "metadata": result.get("metadata", {}),
                        "constraint_hits": result.get("constraint_hits", {}),
                        "scores": {},
                        "pooled_from": [],
                    },
                )
                if not candidate.get("metadata"):
                    candidate["metadata"] = result.get("metadata", {})
                if not candidate.get("constraint_hits"):
                    candidate["constraint_hits"] = result.get("constraint_hits", {})
                candidate["scores"][run_name] = result.get("score")
                candidate["pooled_from"].append({"run": run_name, "rank": rank})
                ranks.setdefault(key, []).append(rank)
    for key, candidate in candidates.items():
        query_id, _item_id = key
        rank = min(ranks.get(key, [999999]))
        summary = reranker_candidate_summary({"item_id": key[1], **candidate}, rank=rank)
        candidate["summary"] = summary
        query_candidates.setdefault(query_id, []).append({**summary, "pooled_from": candidate.get("pooled_from", [])[:5]})
    return query_info, candidates, query_candidates, ranks


def coverage_sample_reasons(qrel: dict[str, Any], candidate: dict[str, Any], ranks: list[int]) -> list[str]:
    reasons = ["coverage_gap"]
    if qrel_needs_adjudication(qrel):
        reasons.append("needs_adjudication")
    if qrel_confidence(qrel) < 0.6:
        reasons.append("low_confidence_qrel")
    if int(qrel.get("grade", 0)) == 3:
        reasons.append("target_or_ideal_qrel")
    if ranks and min(ranks) <= 3:
        reasons.append("top3_candidate")
    if len(qrel.get("pooled_from", [])) <= 1:
        reasons.append("single_workflow_only")
    constraint_hits = candidate.get("constraint_hits", {}) if isinstance(candidate.get("constraint_hits", {}), dict) else {}
    if constraint_hits.get("negative_style") or constraint_hits.get("forbidden_stage"):
        reasons.append("style_or_stage_risk")
    if not candidate:
        reasons.append("missing_run_context")
    return reasons


def coverage_sample_priority(qrel: dict[str, Any], reasons: list[str], ranks: list[int]) -> float:
    weights = {
        "needs_adjudication": 5.0,
        "low_confidence_qrel": 4.0,
        "target_or_ideal_qrel": 3.0,
        "top3_candidate": 2.0,
        "single_workflow_only": 1.0,
        "style_or_stage_risk": 1.0,
        "coverage_gap": 0.5,
        "missing_run_context": -1.0,
    }
    rank_bonus = 1 / max(1, min(ranks or [100]))
    grade_bonus = 0.25 * int(qrel.get("grade", 0))
    return round(sum(weights.get(reason, 0.0) for reason in reasons) + rank_bonus + grade_bonus, 6)


def compact_coverage_qrel(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "grade": int(row.get("grade", 0)),
        "reason": row.get("reason", ""),
        "source": row.get("source", ""),
        "confidence": qrel_confidence(row),
        "needs_adjudication": qrel_needs_adjudication(row),
        "vote_count": len(row.get("grade_votes", [])),
    }


def reviewed_qrels_count(qrels: list[dict[str, Any]]) -> int:
    return sum(1 for row in qrels if not qrel_is_bootstrap_only(row))


def high_reviewed_threshold(qrels: list[dict[str, Any]]) -> int:
    return (len(qrels) + 1) // 2


def reviewed_gap_to_high(qrels: list[dict[str, Any]]) -> int:
    return max(0, high_reviewed_threshold(qrels) - reviewed_qrels_count(qrels))


def load_adjudication_votes(
    path: Path,
    *,
    default_judge_type: str,
    default_judge_id: str,
    judge_version: str,
) -> list[dict[str, Any]]:
    votes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        votes.extend(
            adjudication_votes_from_row(
                row,
                default_judge_type=default_judge_type,
                default_judge_id=default_judge_id,
                judge_version=judge_version,
            )
        )
    return votes


def load_run_rows_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "run_rows" in data:
        run_rows = data["run_rows"]
        cases = data.get("cases", [])
    elif isinstance(data, dict) and all(isinstance(value, list) for value in data.values()):
        run_rows = data
        cases = []
    else:
        raise ValueError("runs JSON must be either {'run_rows': {...}} or a run-name to rows mapping")
    if not isinstance(run_rows, dict):
        raise ValueError("run_rows must be a mapping of run name to ranked rows")
    return {
        "run_rows": {str(run_name): list(rows) for run_name, rows in run_rows.items()},
        "cases": list(cases or []),
    }


def cases_from_run_rows(run_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rows in run_rows.values():
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id and case_id not in by_id:
                by_id[case_id] = {"case_id": case_id, "user_input": row.get("user_input", "")}
    return list(by_id.values())


def adjudication_votes_from_row(
    row: dict[str, Any],
    *,
    default_judge_type: str,
    default_judge_id: str,
    judge_version: str,
) -> list[dict[str, Any]]:
    base = {
        "query_id": row.get("query_id"),
        "judge_type": row.get("judge_type", default_judge_type),
        "judge_id": row.get("judge_id", default_judge_id),
        "judge_version": row.get("judge_version", judge_version),
        "created_at": row.get("created_at"),
    }
    raw_votes = row.get("judgements") or row.get("judgments")
    if isinstance(raw_votes, list):
        votes = []
        for item in raw_votes:
            if not isinstance(item, dict):
                continue
            if "grade" not in item and "score" not in item:
                continue
            votes.append(
                qrel_relevance_vote(
                    query_id=str(item.get("query_id", base["query_id"])),
                    item_id=str(item.get("item_id", "")),
                    grade=int(item.get("grade", item.get("score", 0))),
                    reason=str(item.get("reason", row.get("reason", ""))),
                    judge_type=str(item.get("judge_type", base["judge_type"])),
                    judge_id=str(item.get("judge_id", base["judge_id"])),
                    judge_version=str(item.get("judge_version", base["judge_version"])),
                    confidence=optional_float(item.get("confidence", row.get("confidence"))),
                    created_at=str(item.get("created_at", base.get("created_at") or "")) or None,
                )
            )
        return [vote for vote in votes if vote["query_id"] and vote["item_id"]]
    if "grade" not in row and "score" not in row:
        return []
    return [
        qrel_relevance_vote(
            query_id=str(row.get("query_id", "")),
            item_id=str(row.get("item_id", "")),
            grade=int(row.get("grade", row.get("score", 0))),
            reason=str(row.get("reason", row.get("suggested_reason", ""))),
            judge_type=str(base["judge_type"]),
            judge_id=str(base["judge_id"]),
            judge_version=str(base["judge_version"]),
            confidence=optional_float(row.get("confidence")),
            created_at=str(base.get("created_at") or "") or None,
        )
    ]


def serializable_args(args: Any) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def git_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True)
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


__all__ = [
    "DEFAULT_ADJUDICATED_QRELS_PATH",
    "DEFAULT_ACTIVE_QRELS_SAMPLE_PATH",
    "DEFAULT_COVERAGE_QRELS_SAMPLE_PATH",
    "DEFAULT_POOLED_QRELS_PATH",
    "DEFAULT_POOLED_QRELS_REPORT_PATH",
    "DEFAULT_QRELS_AUDIT_REPORT_PATH",
    "adjudication_votes_from_row",
    "audit_qrels_command",
    "cases_from_run_rows",
    "load_adjudication_votes",
    "load_run_rows_payload",
    "merge_adjudicated_qrels_command",
    "pool_qrels_from_runs_command",
    "sample_active_qrels_from_runs_command",
    "sample_coverage_qrels_from_runs_command",
]
