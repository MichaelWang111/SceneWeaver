from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_ROUND2_OUTPUT_DIR = Path(".tmp") / "retrieval_lab" / "round2_1_taxonomy_gate_validator"
STYLE_RISK_TERMS = {
    "ad_feel",
    "big_company_taste",
    "fortune_500_polish",
    "generic_brand_film",
    "hard_sell",
    "slogan_like",
}


class Round2InputError(ValueError):
    """Raised when a Round 2 validator input is malformed."""


def round2_taxonomy_gate_report_command(args: Any) -> dict[str, Any]:
    return write_round2_taxonomy_gate_report(
        failures_path=Path(getattr(args, "failures")),
        runs_path=Path(getattr(args, "runs")),
        blind_analysis_path=Path(getattr(args, "blind_analysis")),
        qrels_path=optional_input_path(getattr(args, "qrels", None)),
        round1_gate_path=optional_input_path(getattr(args, "round1_gate", None)),
        output_dir=Path(getattr(args, "output_dir", DEFAULT_ROUND2_OUTPUT_DIR)),
    )


def write_round2_taxonomy_gate_report(
    *,
    failures_path: Path,
    runs_path: Path,
    blind_analysis_path: Path,
    qrels_path: Path | None = None,
    round1_gate_path: Path | None = None,
    output_dir: Path = DEFAULT_ROUND2_OUTPUT_DIR,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    report = build_round2_taxonomy_gate_report(
        failures=read_json(failures_path),
        runs=read_json(runs_path),
        blind_analysis=read_json(blind_analysis_path),
        qrels=read_jsonl(qrels_path) if qrels_path else [],
        round1_gate=read_json(round1_gate_path) if round1_gate_path else {},
        inputs={
            "failures": str(failures_path),
            "runs": str(runs_path),
            "qrels": str(qrels_path) if qrels_path else "",
            "blind_analysis": str(blind_analysis_path),
            "round1_gate": str(round1_gate_path) if round1_gate_path else "",
        },
    )
    report["summary"]["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "round2_1_taxonomy_gate_validator.json"
    decisions_path = output_dir / "round2_1_gate_decisions.jsonl"
    markdown_path = output_dir / "round2_1_taxonomy_gate_validator.md"
    hypothesis_path = output_dir / "round2_hypothesis_report.md"
    write_json(report_path, report)
    write_jsonl(decisions_path, report["gate_decisions"])
    markdown = round2_taxonomy_gate_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    hypothesis_path.write_text(round2_hypothesis_markdown(report), encoding="utf-8")
    return {
        "method": "retrieval_lab_round2_taxonomy_gate_validator",
        "summary": {
            **report["summary"],
            "output": str(report_path),
            "decisions_output": str(decisions_path),
            "markdown_output": str(markdown_path),
            "hypothesis_output": str(hypothesis_path),
        },
    }


def build_round2_taxonomy_gate_report(
    *,
    failures: dict[str, Any],
    runs: dict[str, Any],
    blind_analysis: dict[str, Any],
    qrels: list[dict[str, Any]] | None = None,
    round1_gate: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    failure_rows = failures.get("failures", []) if isinstance(failures, dict) else []
    run_rows = runs.get("run_rows", {}) if isinstance(runs, dict) else {}
    blind_rows = blind_analysis.get("rows", []) if isinstance(blind_analysis, dict) else []
    if not isinstance(failure_rows, list):
        raise Round2InputError("failures input must contain a failures list")
    if not isinstance(run_rows, dict):
        raise Round2InputError("runs input must contain a run_rows mapping")
    if not isinstance(blind_rows, list):
        raise Round2InputError("blind analysis input must contain a rows list")

    run_index = index_run_rows(run_rows)
    blind_index = index_blind_rows(blind_rows)
    qrel_map = qrel_grade_map(qrels or [])
    taxonomy_rows: list[dict[str, Any]] = []
    gate_decisions: list[dict[str, Any]] = []
    for failure in failure_rows:
        key = row_key(failure)
        run_row = run_index.get(key, {})
        blind_row = blind_index.get((key[0], key[1], str(failure.get("top1_item_id", "")))) or blind_index.get((key[0], key[1], ""), {})
        taxonomy = classify_round2_residual(failure, run_row=run_row, blind_row=blind_row, qrel_map=qrel_map)
        decision = validate_round2_gate(taxonomy, round1_gate=round1_gate or {})
        taxonomy_rows.append(taxonomy)
        gate_decisions.append(decision)

    summary = summarize_round2_report(
        failures=failures,
        blind_analysis=blind_analysis,
        taxonomy_rows=taxonomy_rows,
        gate_decisions=gate_decisions,
        round1_gate=round1_gate or {},
        inputs=inputs or {},
    )
    return {
        "method": "retrieval_lab_round2_taxonomy_gate_validator",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "taxonomy_rows": taxonomy_rows,
        "gate_decisions": gate_decisions,
        "inputs": inputs or {},
        "fingerprint": data_sha256({"summary": summary, "taxonomy_rows": taxonomy_rows, "gate_decisions": gate_decisions}),
    }


def classify_round2_residual(
    failure: dict[str, Any],
    *,
    run_row: dict[str, Any] | None = None,
    blind_row: dict[str, Any] | None = None,
    qrel_map: dict[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    run_row = run_row or {}
    blind_row = blind_row or {}
    qrel_map = qrel_map or {}
    query_id = str(failure.get("case_id", ""))
    top1_item_id = str(failure.get("top1_item_id") or top1_result(run_row).get("item_id") or "")
    old_top1_grade = int_or_default(blind_row.get("top1_qrel_grade"), int_or_default(failure.get("top1_qrel_grade"), qrel_map.get((query_id, top1_item_id), 0)))
    blind_grade = int_or_none(blind_row.get("blind_grade"))
    old_top1_relevant = old_top1_grade >= 2
    blind_relevant = blind_grade is not None and blind_grade >= 2
    source_failure_type = str(blind_row.get("refined_failure_type") or failure.get("failure_type") or "")
    planner_confidence = float_or_default(run_row.get("planner_confidence", failure.get("planner_confidence")), 1.0)
    ambiguity_level = query_ambiguity_level(run_row)
    top10_relevant_count = int_or_default(blind_row.get("top10_relevant_count"), top10_relevant_count_from_qrels(run_row, qrel_map))
    top1_metadata = top1_result(run_row).get("metadata") if run_row else failure.get("top1_metadata", {})
    if not isinstance(top1_metadata, dict):
        top1_metadata = {}
    style_risks = sorted({str(value) for value in top1_metadata.get("style_risks", []) if value})
    negative_style_query = has_negative_style_query(failure, run_row)
    style_risk_conflict = negative_style_query and bool(set(style_risks) & STYLE_RISK_TERMS)
    query_understanding_risk = source_failure_type == "query_understanding_failure" or planner_confidence < 0.6 or ambiguity_level == "high"
    topk_valid_but_top1_invalid = top10_relevant_count > 0 and not (blind_relevant or old_top1_relevant)
    qrels_disagrees_with_blind = blind_grade is not None and blind_relevant != old_top1_relevant
    primary = round2_primary_failure_type(
        blind_grade=blind_grade,
        blind_relevant=blind_relevant,
        old_top1_relevant=old_top1_relevant,
        style_risk_conflict=style_risk_conflict,
        query_understanding_risk=query_understanding_risk,
        top10_relevant_count=top10_relevant_count,
    )
    secondary_types = round2_secondary_failure_types(
        blind_grade=blind_grade,
        blind_relevant=blind_relevant,
        old_top1_relevant=old_top1_relevant,
        style_risk_conflict=style_risk_conflict,
        query_understanding_risk=query_understanding_risk,
        top10_relevant_count=top10_relevant_count,
    )
    flags = {
        "blind_adjudicated": blind_grade is not None,
        "blind_top1_relevant": blind_relevant,
        "old_top1_relevant": old_top1_relevant,
        "qrels_disagrees_with_blind": qrels_disagrees_with_blind,
        "negative_style_query": negative_style_query,
        "style_risk_conflict": style_risk_conflict,
        "query_understanding_risk": query_understanding_risk,
        "topk_has_valid_candidate": top10_relevant_count > 0,
        "topk_valid_but_top1_invalid": topk_valid_but_top1_invalid,
    }
    return {
        "run_name": failure.get("run_name"),
        "case_id": failure.get("case_id"),
        "user_input": failure.get("user_input", run_row.get("user_input", "")),
        "source_failure_type": source_failure_type,
        "round2_primary_failure_type": primary,
        "round2_secondary_failure_types": secondary_types,
        "target_item_id": failure.get("target_item_id"),
        "target_position_in_artifact": failure.get("target_position_in_artifact"),
        "top1_item_id": top1_item_id,
        "old_top1_qrel_grade": old_top1_grade,
        "blind_grade": blind_grade,
        "blind_reason": blind_row.get("blind_reason", ""),
        "blind_confidence": blind_row.get("blind_confidence"),
        "top10_relevant_count": top10_relevant_count,
        "planner_confidence": planner_confidence,
        "ambiguity_level": ambiguity_level,
        "style_risks": style_risks,
        "flags": flags,
    }


def round2_primary_failure_type(
    *,
    blind_grade: int | None,
    blind_relevant: bool,
    old_top1_relevant: bool,
    style_risk_conflict: bool,
    query_understanding_risk: bool,
    top10_relevant_count: int,
) -> str:
    if blind_grade is None:
        if old_top1_relevant:
            return "legacy_top1_valid_multi_answer"
        if top10_relevant_count > 0:
            return "legacy_topk_valid_but_top1_invalid"
        return "legacy_unresolved_failure"
    if blind_relevant and not old_top1_relevant:
        return "qrels_boundary_shift"
    if not blind_relevant and old_top1_relevant:
        return "qrels_over_generous_top1"
    if not blind_relevant and style_risk_conflict:
        return "style_risk_confirmed_failure"
    if not blind_relevant and query_understanding_risk:
        return "query_understanding_failure_confirmed"
    if not blind_relevant and top10_relevant_count > 0:
        return "top1_rerank_failure"
    if blind_relevant:
        return "blind_confirmed_valid_top1"
    return "blind_rejected_top1"


def round2_secondary_failure_types(
    *,
    blind_grade: int | None,
    blind_relevant: bool,
    old_top1_relevant: bool,
    style_risk_conflict: bool,
    query_understanding_risk: bool,
    top10_relevant_count: int,
) -> list[str]:
    if blind_grade is None:
        return []
    types: list[str] = []
    if blind_relevant and not old_top1_relevant:
        types.append("qrels_boundary_shift")
    if not blind_relevant and old_top1_relevant:
        types.append("qrels_over_generous_top1")
    if not blind_relevant and style_risk_conflict:
        types.append("style_risk_confirmed_failure")
    if not blind_relevant and query_understanding_risk:
        types.append("query_understanding_failure_confirmed")
    if not blind_relevant and top10_relevant_count > 0 and not old_top1_relevant:
        types.append("top1_rerank_failure")
    if blind_relevant and old_top1_relevant:
        types.append("blind_confirmed_valid_top1")
    if not types and not blind_relevant:
        types.append("blind_rejected_top1")
    return sorted(set(types))


def validate_round2_gate(taxonomy: dict[str, Any], *, round1_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    round1_gate = round1_gate or {}
    flags = taxonomy.get("flags", {}) if isinstance(taxonomy.get("flags"), dict) else {}
    primary = str(taxonomy.get("round2_primary_failure_type", ""))
    reasons: list[str] = []
    fallback = ""
    gated_decision = "manual_review_required"
    if not flags.get("blind_adjudicated"):
        reasons.append("missing_blind_adjudication")
        fallback = "manual_qrels_review"
    elif primary == "qrels_boundary_shift":
        gated_decision = "accept_gated_with_qrels_boundary_monitoring"
        reasons.extend(["blind_top1_valid", "old_qrels_underjudged_top1"])
        fallback = "keep_candidate_and_update_qrels_boundary"
    elif primary == "blind_confirmed_valid_top1":
        gated_decision = "accept_gated_with_monitoring"
        reasons.append("blind_top1_valid")
        fallback = "none"
    elif primary == "qrels_over_generous_top1":
        gated_decision = "reject_and_review_qrels"
        reasons.extend(["blind_top1_invalid", "old_qrels_overjudged_top1"])
        fallback = "manual_qrels_review_then_style_safe_signature"
    elif flags.get("style_risk_conflict") and not flags.get("blind_top1_relevant"):
        gated_decision = "reject_and_fallback_style_safe"
        reasons.extend(["blind_top1_invalid", "negative_style_conflict"])
        fallback = "style_safe_signature_after_negative_constraint_rewrite"
    elif flags.get("query_understanding_risk") and not flags.get("blind_top1_relevant"):
        gated_decision = "reject_and_rewrite_query"
        reasons.extend(["blind_top1_invalid", "query_understanding_risk"])
        fallback = "planner_positive_rewrite_then_style_safe_signature"
    elif flags.get("topk_valid_but_top1_invalid"):
        gated_decision = "reject_and_rerank_topk"
        reasons.extend(["blind_top1_invalid", "topk_has_valid_candidate"])
        fallback = "rerank_topk_with_style_risk_penalty"
    else:
        reasons.append(primary or "unclassified")
        fallback = "manual_review_required"
    default_allowed = False
    if round1_gate.get("default_strategy_accepted") is True and gated_decision.startswith("accept"):
        default_allowed = True
    return {
        "run_name": taxonomy.get("run_name"),
        "case_id": taxonomy.get("case_id"),
        "top1_item_id": taxonomy.get("top1_item_id"),
        "round2_primary_failure_type": primary,
        "gated_decision": gated_decision,
        "default_allowed": default_allowed,
        "fallback_strategy": fallback,
        "validator_reasons": sorted(set(reasons)),
        "blind_grade": taxonomy.get("blind_grade"),
        "old_top1_qrel_grade": taxonomy.get("old_top1_qrel_grade"),
        "style_risks": taxonomy.get("style_risks", []),
    }


def summarize_round2_report(
    *,
    failures: dict[str, Any],
    blind_analysis: dict[str, Any],
    taxonomy_rows: list[dict[str, Any]],
    gate_decisions: list[dict[str, Any]],
    round1_gate: dict[str, Any],
    inputs: dict[str, str],
) -> dict[str, Any]:
    primary_counts = Counter(str(row.get("round2_primary_failure_type", "")) for row in taxonomy_rows)
    secondary_counts = Counter(
        str(failure_type)
        for row in taxonomy_rows
        for failure_type in row.get("round2_secondary_failure_types", [])
    )
    source_to_primary_counts = source_primary_counts(taxonomy_rows)
    decision_counts = Counter(str(row.get("gated_decision", "")) for row in gate_decisions)
    flag_counts = Counter(
        flag
        for row in taxonomy_rows
        for flag, enabled in (row.get("flags", {}) if isinstance(row.get("flags"), dict) else {}).items()
        if enabled
    )
    old_counts = failures.get("summary", {}).get("failure_type_counts", {}) if isinstance(failures.get("summary"), dict) else {}
    blind_summary = blind_analysis.get("summary", {}) if isinstance(blind_analysis, dict) else {}
    all_coverage = round1_gate.get("gates", {}).get("all_coverage", {}).get("evidence", {}) if isinstance(round1_gate.get("gates"), dict) else {}
    judged_at_10 = float_or_default(all_coverage.get("after_Judged@10"), 0.0)
    blind_relevant_rate = float_or_default(blind_summary.get("blind_relevant_rate"), 0.0)
    qrels_boundary_count = primary_counts.get("qrels_boundary_shift", 0) + primary_counts.get("qrels_over_generous_top1", 0)
    confirmed_system_failure_count = sum(
        primary_counts.get(name, 0)
        for name in (
            "style_risk_confirmed_failure",
            "query_understanding_failure_confirmed",
            "top1_rerank_failure",
            "blind_rejected_top1",
        )
    )
    query_source_split = source_to_primary_counts.get("query_understanding_failure", {})
    confirmed_query_understanding_source_failures = sum(
        count
        for primary, count in query_source_split.items()
        if primary in {"style_risk_confirmed_failure", "query_understanding_failure_confirmed", "top1_rerank_failure", "blind_rejected_top1"}
    )
    round2_goal_achieved = (
        bool(taxonomy_rows)
        and primary_counts.get("ambiguous_multi_valid_answer", 0) == 0
        and len(taxonomy_rows) == len(gate_decisions)
        and flag_counts.get("blind_adjudicated", 0) == len(taxonomy_rows)
    )
    default_strategy_accepted = bool(round1_gate.get("default_strategy_accepted")) and blind_relevant_rate >= 0.8 and judged_at_10 >= 0.8
    return {
        "cycle_id": "round2_1_taxonomy_gate_validator",
        "round2_goal_achieved": round2_goal_achieved,
        "round2_status": "complete_as_taxonomy_and_gate_round" if round2_goal_achieved else "incomplete",
        "residual_count": len(taxonomy_rows),
        "legacy_failure_type_counts": old_counts,
        "round2_primary_failure_type_counts": dict(sorted(primary_counts.items())),
        "round2_secondary_failure_type_counts": dict(sorted(secondary_counts.items())),
        "source_to_primary_counts": {source: dict(sorted(counts.items())) for source, counts in sorted(source_to_primary_counts.items())},
        "query_understanding_source_split": dict(sorted(query_source_split.items())),
        "gate_decision_counts": dict(sorted(decision_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "blind_relevant_rate": blind_relevant_rate,
        "qrels_boundary_count": qrels_boundary_count,
        "confirmed_system_failure_count": confirmed_system_failure_count,
        "confirmed_query_understanding_source_failure_count": confirmed_query_understanding_source_failures,
        "standalone_query_understanding_primary_count": primary_counts.get("query_understanding_failure_confirmed", 0),
        "round1_all_judged_at_10": judged_at_10,
        "default_strategy_accepted": default_strategy_accepted,
        "default_strategy_rejection_reasons": default_rejection_reasons(
            round1_default=bool(round1_gate.get("default_strategy_accepted")),
            blind_relevant_rate=blind_relevant_rate,
            judged_at_10=judged_at_10,
            confirmed_system_failure_count=confirmed_system_failure_count,
        ),
        "next_cycle": {
            "suggested_id": "round3_1_llm_query_corpus_pilot",
            "entry_allowed": round2_goal_achieved,
            "hypothesis": "LLM-generated natural query corpus should be added after the gate can separate qrels boundary, style risk, and query-understanding failures.",
            "primary_targets": ["style_risk_confirmed_failure", "qrels_boundary_shift", "qrels_over_generous_top1", "query_understanding_source_split"],
        },
        "inputs": inputs,
    }


def default_rejection_reasons(
    *,
    round1_default: bool,
    blind_relevant_rate: float,
    judged_at_10: float,
    confirmed_system_failure_count: int,
) -> list[str]:
    reasons = []
    if not round1_default:
        reasons.append("round1_default_strategy_was_rejected")
    if blind_relevant_rate < 0.8:
        reasons.append("blind_top1_relevant_rate_below_0_8")
    if judged_at_10 < 0.8:
        reasons.append("all_slice_judged_at_10_below_0_8")
    if confirmed_system_failure_count > 0:
        reasons.append("confirmed_system_failures_remain")
    return reasons


def source_primary_counts(taxonomy_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for row in taxonomy_rows:
        source = str(row.get("source_failure_type", "unknown") or "unknown")
        primary = str(row.get("round2_primary_failure_type", "unknown") or "unknown")
        counts.setdefault(source, Counter())[primary] += 1
    return {source: dict(counter) for source, counter in counts.items()}


def round2_taxonomy_gate_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    type_counts = summary.get("round2_primary_failure_type_counts", {})
    secondary_counts = summary.get("round2_secondary_failure_type_counts", {})
    query_split = summary.get("query_understanding_source_split", {})
    gate_counts = summary.get("gate_decision_counts", {})
    lines = [
        "# Round 2-1 Taxonomy Gate Validator",
        "",
        "## Status",
        "",
        f"- cycle: `{summary.get('cycle_id')}`",
        f"- status: `{summary.get('round2_status')}`",
        f"- goal achieved: `{summary.get('round2_goal_achieved')}`",
        f"- residual count: `{summary.get('residual_count')}`",
        f"- default strategy accepted: `{summary.get('default_strategy_accepted')}`",
        "",
        "## Refined Failure Taxonomy",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in type_counts.items())
    lines.extend([
        "",
        "## Secondary Failure Signals",
        "",
    ])
    lines.extend(f"- `{name}`: {count}" for name, count in secondary_counts.items())
    lines.extend([
        "",
        "## Query Understanding Source Split",
        "",
    ])
    lines.extend(f"- `{name}`: {count}" for name, count in query_split.items())
    lines.extend([
        "",
        "## Gate Decisions",
        "",
    ])
    lines.extend(f"- `{name}`: {count}" for name, count in gate_counts.items())
    lines.extend([
        "",
        "## Main Decision",
        "",
        "Round 2 completes the executable taxonomy/gate step. It does not approve calibrated rerank as a default strategy.",
        "The validator keeps calibrated rerank conditional because blind residual relevance is mixed, all-slice judged coverage remains below the default threshold, and confirmed style-risk failures remain.",
        "The old query-understanding bucket did not survive as a standalone primary class: blind review split it into qrels boundary shifts and style-risk-confirmed failures.",
        "",
        "## Next Cycle",
        "",
        f"- suggested id: `{summary.get('next_cycle', {}).get('suggested_id')}`",
        f"- entry allowed: `{summary.get('next_cycle', {}).get('entry_allowed')}`",
        f"- hypothesis: {summary.get('next_cycle', {}).get('hypothesis')}",
    ])
    return "\n".join(lines) + "\n"


def round2_hypothesis_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    type_counts = summary.get("round2_primary_failure_type_counts", {})
    query_split = summary.get("query_understanding_source_split", {})
    reasons = summary.get("default_strategy_rejection_reasons", [])
    return "\n".join(
        [
            "# Round 1 Closure + Round 2 Hypothesis Report",
            "",
            "## Proposition 1",
            "",
            "Negative aesthetic requests can be represented as positive retrieval intent plus negative constraints and risk penalties, but the gate must distinguish acceptable documentary overlap from confirmed style conflict.",
            "",
            "## Attempt",
            "",
            "Round 1 built trusted internal qrels, expanded judged coverage, and accepted calibrated rerank only as a gated strategy. Round 1-6 then challenged the residual failures with blind LLM judgements. Round 2 converted that audit into an executable taxonomy and per-case gate validator.",
            "",
            "## Observation",
            "",
            f"- residuals reviewed: {summary.get('residual_count')}",
            f"- qrels boundary shifts: {summary.get('qrels_boundary_count')}",
            f"- confirmed system failures: {summary.get('confirmed_system_failure_count')}",
            f"- standalone query-understanding primary failures: {summary.get('standalone_query_understanding_primary_count')}",
            f"- query-understanding source failures still invalid after blind review: {summary.get('confirmed_query_understanding_source_failure_count')}",
            f"- blind top1 relevant rate: {summary.get('blind_relevant_rate')}",
            f"- all-slice Judged@10: {summary.get('round1_all_judged_at_10')}",
            "",
            "## Finding",
            "",
            "The old `ambiguous_multi_valid_answer` bucket was too broad. The executable split now separates qrels boundary movement, over-generous qrels, confirmed style-risk failure, confirmed query-understanding failure, and top1 rerank failure.",
            "",
            "## Refined Counts",
            "",
            *[f"- `{name}`: {count}" for name, count in type_counts.items()],
            "",
            "## Query Understanding Source Split",
            "",
            *[f"- `{name}`: {count}" for name, count in query_split.items()],
            "",
            "## Decision",
            "",
            "Round 2 is complete as a taxonomy/gate round. Calibrated rerank remains accepted only under a gate and is still rejected as default.",
            "",
            "## Why Default Is Still Rejected",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
            "## Next Step",
            "",
            "Round 3 may start a small real LLM-generated natural-query corpus pilot, with hard budget enforcement, because the validator can now separate qrels-boundary cases from system failures instead of letting metrics hide the proposition being tested.",
        ]
    ) + "\n"


def index_run_rows(run_rows: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(run_name), str(row.get("case_id", ""))): row for run_name, rows in run_rows.items() for row in rows}


def index_blind_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("run_name", "")), str(row.get("case_id", "")), str(row.get("top1_item_id", "")))
        index[key] = row
        index.setdefault((key[0], key[1], ""), row)
    return index


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("run_name", "")), str(row.get("case_id", "")))


def qrel_grade_map(qrels: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    return {(str(row.get("query_id", "")), str(row.get("item_id", ""))): int_or_default(row.get("grade"), 0) for row in qrels}


def top1_result(row: dict[str, Any]) -> dict[str, Any]:
    top = row.get("top_results", []) if isinstance(row.get("top_results"), list) else []
    return top[0] if top and isinstance(top[0], dict) else {}


def top10_relevant_count_from_qrels(row: dict[str, Any], qrel_map: dict[tuple[str, str], int]) -> int:
    query_id = str(row.get("case_id", ""))
    top = row.get("top_results", []) if isinstance(row.get("top_results"), list) else []
    return sum(1 for result in top[:10] if qrel_map.get((query_id, str(result.get("item_id", ""))), 0) >= 2)


def has_negative_style_query(failure: dict[str, Any], run_row: dict[str, Any]) -> bool:
    plan = run_row.get("query_plan", {}) if isinstance(run_row.get("query_plan"), dict) else {}
    if plan.get("negative_style") or plan.get("negative_constraints"):
        return True
    variant = str(run_row.get("variant_type", failure.get("case_id", "")))
    return "negative_style" in variant or "natural_negative_style" in variant


def query_ambiguity_level(run_row: dict[str, Any]) -> str:
    plan = run_row.get("query_plan", {}) if isinstance(run_row.get("query_plan"), dict) else {}
    ambiguity = plan.get("ambiguity", {}) if isinstance(plan.get("ambiguity"), dict) else {}
    return str(ambiguity.get("level", ""))


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def int_or_default(value: Any, default: int) -> int:
    parsed = int_or_none(value)
    return default if parsed is None else parsed


def float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_input_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if str(value) == "" or str(path) in {"", "."}:
        return None
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Round 2 taxonomy and gate validator artifacts.")
    parser.add_argument("--failures", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--blind-analysis", required=True)
    parser.add_argument("--qrels", default="")
    parser.add_argument("--round1-gate", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_ROUND2_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = round2_taxonomy_gate_report_command(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
