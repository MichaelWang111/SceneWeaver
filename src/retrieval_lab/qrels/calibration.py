from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_jsonl, write_json
from retrieval_lab.qrels.service import (
    load_qrels,
    qrel_has_vote_conflict,
    qrel_needs_adjudication,
    qrel_vote_judge_type,
    qrels_audit_summary,
)


DEFAULT_JUDGE_CALIBRATION_REPORT = Path(".tmp") / "retrieval_lab" / "judge_calibration.json"


def judge_calibration_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    samples_path = optional_path(getattr(args, "samples", None))
    qrels_path = Path(getattr(args, "qrels"))
    qrels = load_qrels(qrels_path)
    samples = read_jsonl(samples_path) if samples_path and samples_path.exists() else []
    summary = judge_calibration_summary(qrels, samples=samples)
    summary.update(
        {
            "samples": str(samples_path or ""),
            "qrels": str(qrels_path),
            "require_llm": bool(getattr(args, "require_llm", False)),
            "repeat_count": int(getattr(args, "repeat_count", 0) or 0),
            "shuffle_candidate_order": bool(getattr(args, "shuffle_candidate_order", False)),
            "llm_call_count": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    if summary["require_llm"]:
        summary["status"] = "llm_calibration_requested_but_not_executed_in_offline_command"
    report = {
        "method": "retrieval_lab_qrels_judge_calibration",
        "summary": summary,
        "slice_metrics": qrels_trust_by_slice(qrels),
        "recommended_next_adjudication_queue": recommended_queue(samples, qrels)[:100],
        "fingerprint": data_sha256({"summary": summary, "slice_metrics": qrels_trust_by_slice(qrels)}),
    }
    output = Path(getattr(args, "output", DEFAULT_JUDGE_CALIBRATION_REPORT))
    write_json(output, report)
    markdown_output = getattr(args, "markdown_output", None)
    if markdown_output is not None:
        Path(markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_output).write_text(judge_calibration_markdown(report), encoding="utf-8")
    return {"method": report["method"], "output": str(output), "summary": summary}


def judge_calibration_summary(qrels: list[dict[str, Any]], *, samples: list[dict[str, Any]]) -> dict[str, Any]:
    vote_groups = [vote_grades(row) for row in qrels if vote_grades(row)]
    repeated = [grades for grades in vote_groups if len(grades) >= 2]
    agreement = [max(Counter(grades).values()) / len(grades) for grades in repeated]
    variances = [grade_variance(grades) for grades in repeated]
    audit = qrels_audit_summary(qrels)
    judge_versions = Counter(
        str(vote.get("judge_version", "unknown"))
        for row in qrels
        for vote in row.get("grade_votes", [])
        if isinstance(vote, dict)
    )
    judge_types = Counter(
        qrel_vote_judge_type(vote, row)
        for row in qrels
        for vote in row.get("grade_votes", [])
        if isinstance(vote, dict)
    )
    return {
        **audit,
        "sample_count": len(samples),
        "same_sample_repeat_count": len(repeated),
        "same_sample_agreement_rate": round(sum(agreement) / max(1, len(agreement)), 6),
        "mean_grade_variance": round(sum(variances) / max(1, len(variances)), 6),
        "position_bias_rate": 0.0,
        "position_bias_note": "offline analysis has no shuffled repeated judge calls; run with --require-llm to measure.",
        "judge_prompt_version_distribution": dict(sorted(judge_versions.items())),
        "judge_type_distribution": dict(sorted(judge_types.items())),
        "recommended_queue_count": sum(1 for row in qrels if qrel_needs_adjudication(row)),
        "status": "offline_analysis",
    }


def qrels_trust_by_slice(qrels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        query_id = str(row.get("query_id", ""))
        slice_name = "default"
        if "natural" in query_id:
            slice_name = "natural_fuzzy"
        elif "fuzzy" in query_id:
            slice_name = "all_fuzzy"
        elif "style" in query_id:
            slice_name = "style_negative"
        grouped[slice_name].append(row)
    return {name: qrels_audit_summary(rows) for name, rows in sorted(grouped.items())}


def recommended_queue(samples: list[dict[str, Any]], qrels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    needs = {(str(row.get("query_id", "")), str(row.get("item_id", ""))) for row in qrels if qrel_needs_adjudication(row)}
    conflicts = {(str(row.get("query_id", "")), str(row.get("item_id", ""))) for row in qrels if qrel_has_vote_conflict(row)}
    scored = []
    for sample in samples:
        key = (str(sample.get("query_id", "")), str(sample.get("item_id", "")))
        priority = 0
        if key in conflicts:
            priority += 3
        if key in needs:
            priority += 2
        if sample.get("low_confidence_reason"):
            priority += 1
        if priority:
            row = dict(sample)
            row["calibration_priority"] = priority
            scored.append(row)
    return sorted(scored, key=lambda row: (-int(row.get("calibration_priority", 0)), str(row.get("query_id", ""))))


def vote_grades(row: dict[str, Any]) -> list[int]:
    return [int(vote.get("grade", row.get("grade", 0))) for vote in row.get("grade_votes", []) if isinstance(vote, dict)]


def grade_variance(grades: list[int]) -> float:
    if not grades:
        return 0.0
    mean = sum(grades) / len(grades)
    return sum((grade - mean) ** 2 for grade in grades) / len(grades)


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if str(path) in {"", "."}:
        return None
    return path


def judge_calibration_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = ["# Judge Calibration Report", "", "## Summary", ""]
    for key in (
        "qrels_count",
        "llm_count",
        "bootstrap_only_count",
        "vote_conflict_rate",
        "same_sample_agreement_rate",
        "mean_grade_variance",
        "position_bias_rate",
        "recommended_queue_count",
        "status",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Slices", "", "| slice | qrels | trust | conflict | bootstrap_only |", "|---|---:|---|---:|---:|"])
    for name, row in report.get("slice_metrics", {}).items():
        lines.append(
            f"| {name} | {row.get('qrels_count')} | {row.get('qrels_trust_level')} | {row.get('vote_conflict_rate')} | {row.get('bootstrap_only_count')} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["DEFAULT_JUDGE_CALIBRATION_REPORT", "judge_calibration_command", "judge_calibration_summary"]
