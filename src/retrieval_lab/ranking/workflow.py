from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, write_json
from retrieval_lab.evaluators.run_eval import evaluate_run_rows
from retrieval_lab.experiments.runs import cases_from_run_rows, run_artifact_summary, unique_run_name
from retrieval_lab.qrels import load_qrels
from retrieval_lab.qrels import canonical_stage
from retrieval_lab.ranking.rerank import row_with_reranked_results


DEFAULT_WORKFLOW_RUN_ARTIFACT_PATH = Path(".tmp") / "retrieval_lab" / "workflow_run_rows_latest.json"
DEFAULT_WORKFLOW_COMPARISON_PATH = Path(".tmp") / "retrieval_lab" / "workflow_comparison_latest.json"
HARD_FORBIDDEN_STAGE_VETO = 1000.0
SIGNATURE_SCORE_WEIGHT = 0.35
STYLE_SAFE_RISK_PENALTY = 50.0

CONSTRAINT_RANKING_KEYS = {
    "final_score",
    "constraints_only",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
    "feature_rerank_signature",
    "lexical_constraints",
    "lexical_constraints_signature",
    "style_safe_hybrid",
    "style_safe_signature",
    "style_safe_adaptive",
}

WORKFLOW_RANKING_KEYS = {
    "final_score",
    "embedding_only",
    "semantic_only",
    "lexical_only",
    "lexical_constraints",
    "lexical_constraints_signature",
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "script_use_only",
    "visual_tags_only",
    "experience_only",
    "combined_only",
    "constraints_only",
    "signature_only",
    "semantic_signature",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
    "feature_rerank_signature",
    "style_safe_hybrid",
    "style_safe_signature",
    "style_safe_adaptive",
}


def workflow_run_artifact_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    ranking_key = str(getattr(args, "ranking_key", "hybrid_rrf_constraints"))
    top_k = int(getattr(args, "top_k", 10))
    source = read_json(runs_path)
    run_rows = source.get("run_rows", {}) if isinstance(source, dict) else {}
    if not isinstance(run_rows, dict):
        raise ValueError("runs artifact must contain a run_rows mapping")
    reranked = rerank_run_rows_by_workflow(
        {str(name): list(rows) for name, rows in run_rows.items()},
        ranking_key=ranking_key,
        top_k=top_k,
    )
    cases = cases_from_run_rows(reranked)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        **run_artifact_summary(reranked, cases),
        "ranking_key": ranking_key,
        "source_runs": str(runs_path),
        "top_k": top_k,
        "elapsed_seconds": elapsed_seconds,
    }
    artifact = {
        "method": "retrieval_lab_workflow_run_artifact",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ranking_key": ranking_key,
        "top_k": top_k,
        "source_runs": str(runs_path),
        "run_rows": reranked,
        "cases": cases,
        "summary": summary,
        "fingerprint": data_sha256(reranked),
    }
    output = Path(getattr(args, "output", DEFAULT_WORKFLOW_RUN_ARTIFACT_PATH))
    write_json(output, artifact)
    return {
        "method": "retrieval_lab_workflow_run_artifact",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": artifact["fingerprint"]},
    }


def compare_workflow_runs_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    ranking_keys = parse_ranking_keys(str(getattr(args, "ranking_keys", "hybrid_rrf_constraints")))
    top_k = int(getattr(args, "top_k", 10))
    source = read_json(runs_path)
    source_rows = source.get("run_rows", {}) if isinstance(source, dict) else {}
    if not isinstance(source_rows, dict):
        raise ValueError("runs artifact must contain a run_rows mapping")
    workflow_runs: dict[str, list[dict[str, Any]]] = {}
    workflow_summaries = {}
    for ranking_key in ranking_keys:
        reranked = rerank_run_rows_by_workflow(
            {str(name): list(rows) for name, rows in source_rows.items()},
            ranking_key=ranking_key,
            top_k=top_k,
        )
        for run_name, rows in reranked.items():
            workflow_runs[run_name] = rows
        workflow_summaries[ranking_key] = run_artifact_summary(reranked, cases_from_run_rows(reranked))
    qrels_path = Path(getattr(args, "qrels", "")) if getattr(args, "qrels", "") else None
    run_metrics = {}
    if qrels_path and qrels_path.exists():
        qrels = load_qrels(qrels_path)
        if qrels:
            run_metrics = evaluate_run_rows(workflow_runs, qrels=qrels, top_k=top_k)
    best_run = select_best_workflow_run(run_metrics) if run_metrics else ""
    cases = cases_from_run_rows(workflow_runs)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        **run_artifact_summary(workflow_runs, cases),
        "source_runs": str(runs_path),
        "ranking_keys": ranking_keys,
        "top_k": top_k,
        "best_run": best_run,
        "qrels": str(qrels_path) if qrels_path else "",
        "elapsed_seconds": elapsed_seconds,
    }
    report = {
        "method": "retrieval_lab_workflow_comparison",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_runs": str(runs_path),
        "ranking_keys": ranking_keys,
        "top_k": top_k,
        "run_rows": workflow_runs,
        "cases": cases,
        "workflow_summaries": workflow_summaries,
        "run_metrics": run_metrics,
        "summary": summary,
        "fingerprint": data_sha256({"run_rows": workflow_runs, "metrics": run_metrics}),
    }
    output = Path(getattr(args, "output", DEFAULT_WORKFLOW_COMPARISON_PATH))
    write_json(output, report)
    return {
        "method": "retrieval_lab_compare_workflow_runs",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": report["fingerprint"]},
    }


def rerank_run_rows_by_workflow(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    ranking_key: str,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    if ranking_key not in WORKFLOW_RANKING_KEYS:
        raise ValueError(f"unknown workflow ranking key: {ranking_key}")
    result: dict[str, list[dict[str, Any]]] = {}
    for run_name, rows in run_rows.items():
        output_name = unique_run_name(result, f"{run_name}::{ranking_key}")
        result[output_name] = [rerank_row_by_workflow(row, ranking_key=ranking_key, top_k=top_k) for row in rows]
    return result


def parse_ranking_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in WORKFLOW_RANKING_KEYS]
    if unknown:
        raise ValueError(f"unknown workflow ranking key(s): {', '.join(unknown)}")
    return keys


def select_best_workflow_run(run_metrics: dict[str, dict[str, float]]) -> str:
    if not run_metrics:
        return ""
    return max(
        run_metrics,
        key=lambda name: (
            float(run_metrics[name].get("nDCG@10", 0.0)),
            float(run_metrics[name].get("MRR@10", 0.0)),
            float(run_metrics[name].get("Recall@10", 0.0)),
        ),
    )


def rerank_row_by_workflow(row: dict[str, Any], *, ranking_key: str, top_k: int) -> dict[str, Any]:
    if ranking_key not in WORKFLOW_RANKING_KEYS:
        raise ValueError(f"unknown workflow ranking key: {ranking_key}")
    candidates = list(row.get("all_results") or row.get("top_results") or [])
    scored = []
    for original_index, result in enumerate(candidates):
        score = workflow_score(result, ranking_key=ranking_key, row=row)
        scored.append((score, original_index, result))
    reranked = []
    for score, _original_index, result in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]:
        copied = dict(result)
        copied["score"] = round(score, 6)
        copied["final_score"] = round(score, 6)
        copied["ranking_key"] = ranking_key
        copied["workflow_score_components"] = workflow_score_components(result, row=row, ranking_key=ranking_key)
        copied.setdefault("style_risk_score", style_risk_score(result))
        copied.setdefault("style_guardrail_action", style_guardrail_action(result))
        copied.setdefault("risk_evidence", style_risk_evidence(result))
        reranked.append(copied)
    return row_with_reranked_results(row, reranked, ranking_key=ranking_key)


def workflow_score(result: dict[str, Any], *, ranking_key: str, row: dict[str, Any] | None = None) -> float:
    components = workflow_score_components(result, row=row or {}, ranking_key=ranking_key)
    score = float(components["base_score"])
    if components.get("forbidden_stage_veto"):
        score -= HARD_FORBIDDEN_STAGE_VETO
    return round(score, 6)


def workflow_score_components(
    result: dict[str, Any],
    *,
    row: dict[str, Any],
    ranking_key: str,
) -> dict[str, Any]:
    semantic = numeric_field(result, "embedding_score", fallback_keys=("semantic_score", "score"))
    lexical = numeric_field(result, "lexical_score")
    rrf = numeric_field(result, "rrf_score")
    constraint = numeric_field(result, "constraint_score")
    signature = numeric_field(result, "signature_score")
    channel_scores = result.get("channel_scores", {}) if isinstance(result.get("channel_scores"), dict) else {}

    if ranking_key == "final_score":
        base = numeric_field(result, "final_score", fallback_keys=("score",))
    elif ranking_key in {"embedding_only", "semantic_only"}:
        base = semantic
    elif ranking_key == "lexical_only":
        base = lexical
    elif ranking_key == "lexical_constraints":
        base = lexical + constraint
    elif ranking_key == "lexical_constraints_signature":
        base = lexical + constraint + signature * SIGNATURE_SCORE_WEIGHT
    elif ranking_key == "hybrid_rrf":
        base = rrf
    elif ranking_key in {"hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}:
        base = rrf + constraint
    elif ranking_key == "constraints_only":
        base = constraint
    elif ranking_key == "signature_only":
        base = signature
    elif ranking_key == "semantic_signature":
        base = semantic + signature * SIGNATURE_SCORE_WEIGHT
    elif ranking_key == "hybrid_rrf_constraints_signature":
        base = rrf + constraint + signature * SIGNATURE_SCORE_WEIGHT
    elif ranking_key == "adaptive_signature":
        base = rrf + constraint + signature * adaptive_signature_weight(row)
    elif ranking_key == "feature_rerank_signature":
        base = feature_rerank_signature_score(result, row=row)
    elif ranking_key == "style_safe_hybrid":
        base = rrf + constraint + style_safe_adjustment(result)
    elif ranking_key == "style_safe_signature":
        base = rrf + constraint + signature * SIGNATURE_SCORE_WEIGHT + style_safe_adjustment(result)
    elif ranking_key == "style_safe_adaptive":
        base = rrf + constraint + signature * adaptive_signature_weight(row) + style_safe_adjustment(result)
    elif ranking_key.endswith("_only"):
        base = float(channel_scores.get(ranking_key.removesuffix("_only"), 0.0) or 0.0)
    else:
        raise ValueError(f"unknown workflow ranking key: {ranking_key}")

    return {
        "ranking_key": ranking_key,
        "base_score": round(float(base), 6),
        "semantic": round(semantic, 6),
        "lexical": round(lexical, 6),
        "rrf": round(rrf, 6),
        "constraint": round(constraint, 6),
        "signature": round(signature, 6),
        "style": round(numeric_field(result, "style_score"), 6),
        "style_safe_adjustment": round(style_safe_adjustment(result), 6),
        "feature_rerank_signature": round(feature_rerank_signature_score(result, row=row), 6),
        "style_risk_score": style_risk_score(result),
        "style_guardrail_action": style_guardrail_action(result),
        "risk_evidence": style_risk_evidence(result),
        "signature_weight": adaptive_signature_weight(row)
        if ranking_key in {"adaptive_signature", "style_safe_adaptive"}
        else (SIGNATURE_SCORE_WEIGHT if "signature" in ranking_key else 0.0),
        "forbidden_stage_veto": has_forbidden_stage_hit(row, result) if ranking_key in CONSTRAINT_RANKING_KEYS else False,
    }


def numeric_field(result: dict[str, Any], key: str, *, fallback_keys: tuple[str, ...] = ()) -> float:
    for candidate in (key, *fallback_keys):
        value = result.get(candidate)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def feature_rerank_signature_score(result: dict[str, Any], *, row: dict[str, Any]) -> float:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    score = numeric_field(result, "rrf_score")
    score += 0.7 * numeric_field(result, "constraint_score")
    score += 0.55 * numeric_field(result, "signature_score")
    score += 0.35 * numeric_field(result, "purpose_score")
    score += 0.2 * max(numeric_field(result, "style_score"), 0.0)
    if hits.get("desired_stage"):
        score += 0.18
    if hits.get("positive_style"):
        score += 0.08 * len(hits["positive_style"])
    if hits.get("negative_style"):
        score -= 0.85 * len(hits["negative_style"])
    if has_forbidden_stage_hit(row, result):
        score -= HARD_FORBIDDEN_STAGE_VETO
    return round(score, 6)


def adaptive_signature_weight(row: dict[str, Any]) -> float:
    query_plan = row.get("query_plan", {}) if isinstance(row.get("query_plan"), dict) else {}
    ambiguity = query_plan.get("ambiguity", {}) if isinstance(query_plan.get("ambiguity"), dict) else {}
    if not ambiguity and isinstance(row.get("ambiguity"), dict):
        ambiguity = row["ambiguity"]
    level = str(ambiguity.get("level", "") or "").lower()
    if level == "high":
        return 0.55
    if level == "medium":
        return 0.45
    return 0.3


def style_safe_adjustment(result: dict[str, Any]) -> float:
    risk_count = style_risk_score(result)
    style = numeric_field(result, "style_score")
    if risk_count:
        return min(style, 0.0) - STYLE_SAFE_RISK_PENALTY * risk_count
    return max(style, 0.0)


def style_risk_score(result: dict[str, Any]) -> int:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    values = hits.get("negative_style", [])
    return len(values) if isinstance(values, list) else 0


def style_guardrail_action(result: dict[str, Any]) -> str:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    if hits.get("negative_style"):
        return "strong_style_penalty"
    if hits.get("positive_style"):
        return "positive_style_bonus"
    return "none"


def style_risk_evidence(result: dict[str, Any]) -> list[dict[str, Any]]:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    negative = hits.get("negative_style", [])
    if not isinstance(negative, list):
        return []
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    item_risks = metadata.get("style_risks", []) if isinstance(metadata.get("style_risks", []), list) else []
    return [
        {
            "style": str(style),
            "source": "constraint_hits.negative_style",
            "metadata_style_risks": [str(value) for value in item_risks],
        }
        for style in negative
    ]


def has_forbidden_stage_hit(row: dict[str, Any], result: dict[str, Any]) -> bool:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    if hits.get("forbidden_stage"):
        return True
    query_constraints = row.get("query_constraints", {}) if isinstance(row.get("query_constraints"), dict) else {}
    forbidden = {canonical_stage(stage) for stage in query_constraints.get("forbidden_stage", [])}
    if not forbidden:
        return False
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    return canonical_stage(metadata.get("script_stage", "")) in forbidden


__all__ = [
    "CONSTRAINT_RANKING_KEYS",
    "DEFAULT_WORKFLOW_COMPARISON_PATH",
    "DEFAULT_WORKFLOW_RUN_ARTIFACT_PATH",
    "compare_workflow_runs_command",
    "HARD_FORBIDDEN_STAGE_VETO",
    "SIGNATURE_SCORE_WEIGHT",
    "STYLE_SAFE_RISK_PENALTY",
    "WORKFLOW_RANKING_KEYS",
    "adaptive_signature_weight",
    "feature_rerank_signature_score",
    "has_forbidden_stage_hit",
    "parse_ranking_keys",
    "rerank_row_by_workflow",
    "rerank_run_rows_by_workflow",
    "workflow_run_artifact_command",
    "workflow_score",
    "workflow_score_components",
    "style_guardrail_action",
    "style_risk_evidence",
    "style_risk_score",
    "style_safe_adjustment",
]
