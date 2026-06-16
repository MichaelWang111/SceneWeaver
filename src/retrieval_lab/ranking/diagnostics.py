from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, read_jsonl, write_json, write_jsonl
from retrieval_lab.evaluators.metrics import graded_metrics
from retrieval_lab.experiments.runs import cases_from_run_rows, extract_run_rows_from_report, run_artifact_summary, unique_run_name
from retrieval_lab.qrels import canonical_stage, load_qrels, qrel_confidence, qrel_has_vote_conflict, qrel_needs_adjudication
from retrieval_lab.ranking.rerank import row_with_reranked_results
from retrieval_lab.ranking.workflow import (
    has_forbidden_stage_hit,
    style_risk_score,
    workflow_score_components,
)


DEFAULT_RERANK_FEATURES_PATH = Path(".tmp") / "retrieval_lab" / "rerank_features.jsonl"
DEFAULT_RERANK_FEATURE_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "rerank_features_report.json"
DEFAULT_CALIBRATED_MODEL_PATH = Path(".tmp") / "retrieval_lab" / "calibrated_rerank_model.json"
DEFAULT_CALIBRATED_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "calibrated_rerank_report.json"
DEFAULT_CALIBRATED_RUNS_PATH = Path(".tmp") / "retrieval_lab" / "calibrated_reranked_runs.json"
DEFAULT_RERANK_ATTRIBUTION_PATH = Path(".tmp") / "retrieval_lab" / "rerank_attribution.json"

FEATURE_PROFILE_DIAGNOSTIC = "diagnostic"
FEATURE_PROFILE_PRODUCTION_SAFE = "production_safe"
FEATURE_PROFILES = {FEATURE_PROFILE_DIAGNOSTIC, FEATURE_PROFILE_PRODUCTION_SAFE}
PRODUCTION_SAFE_EXCLUDED_FEATURES = {"target_like_score"}
SPLIT_STRATEGIES = {"none", "query_hash", "fixture_holdout"}

FEATURE_COLUMNS = (
    "original_score",
    "lexical_score",
    "semantic_score",
    "rrf_score",
    "constraint_score",
    "signature_score",
    "stage_match",
    "purpose_match",
    "style_risk_score",
    "style_violation",
    "negative_style_hit",
    "forbidden_stage_hit",
    "target_like_score",
    "graph_neighbor_score",
)

DEFAULT_LINEAR_WEIGHTS = {
    "original_score": 0.35,
    "rrf_score": 0.45,
    "constraint_score": 0.35,
    "signature_score": 0.45,
    "stage_match": 0.20,
    "purpose_match": 0.25,
    "target_like_score": 0.35,
    "graph_neighbor_score": 0.15,
    "style_risk_score": -0.75,
    "style_violation": -1.00,
    "negative_style_hit": -0.85,
    "forbidden_stage_hit": -3.00,
}


def export_rerank_features_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    qrels_path = optional_path(getattr(args, "qrels", None))
    feature_profile = normalize_feature_profile(getattr(args, "feature_profile", FEATURE_PROFILE_DIAGNOSTIC))
    source = read_json(runs_path)
    extraction = extract_run_rows_from_report(source, source_path=runs_path)
    run_rows = extraction["run_rows"]
    qrels = load_qrels(qrels_path) if qrels_path and qrels_path.exists() else []
    feature_rows = feature_rows_from_runs(run_rows, qrels=qrels, feature_profile=feature_profile)
    output = Path(getattr(args, "output", DEFAULT_RERANK_FEATURES_PATH))
    write_jsonl(output, feature_rows)
    summary = feature_export_summary(feature_rows, qrels=qrels)
    compact_summary = compact_workflow_summary(source)
    summary.update(
        {
            "feature_profile": feature_profile,
            "excluded_features": sorted(profile_excluded_features(feature_profile)),
            "runs": str(runs_path),
            "qrels": str(qrels_path or ""),
            "output": str(output),
            "row_level_available": bool(feature_rows),
            "source_run_count": len(run_rows),
            "source_row_count": sum(len(rows) for rows in run_rows.values()),
            "compact_workflow_count": len(compact_summary),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    report = {
        "method": "retrieval_lab_rerank_feature_export",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "features_preview": feature_rows[:20],
        "compact_workflows": compact_summary,
        "fingerprint": data_sha256({"features": feature_rows, "summary": summary}),
    }
    report_output = Path(getattr(args, "report_output", DEFAULT_RERANK_FEATURE_REPORT_PATH))
    write_json(report_output, report)
    return {"method": report["method"], "output": str(output), "summary": summary}


def calibrate_rerank_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    features_path = Path(getattr(args, "features"))
    rows = read_jsonl(features_path) if features_path.exists() else []
    method = str(getattr(args, "method", "linear"))
    feature_profile = normalize_feature_profile(infer_feature_profile(rows))
    explicit_excluded = csv_set(str(getattr(args, "exclude_features", "") or ""))
    excluded_features = profile_excluded_features(feature_profile) | explicit_excluded
    split_strategy = normalize_split_strategy(getattr(args, "split_strategy", "none"))
    train_ratio = bounded_ratio(float(getattr(args, "train_ratio", 1.0) or 1.0))
    train_rows, eval_rows, split_summary = split_feature_rows(rows, strategy=split_strategy, train_ratio=train_ratio)
    baseline_weights = {"original_score": 1.0}
    linear_weights = weights_without_features(DEFAULT_LINEAR_WEIGHTS, excluded_features)
    feature_columns = [feature for feature in FEATURE_COLUMNS if feature not in excluded_features]
    if method == "coordinate_search" and train_rows:
        weights = coordinate_search_weights(train_rows, initial=linear_weights, feature_columns=feature_columns)
    elif method == "qrels_oracle_upper":
        weights = {"qrel_grade": 1.0}
    else:
        weights = linear_weights
    train_baseline_metrics = metrics_for_feature_rows(train_rows, baseline_weights)
    train_calibrated_metrics = metrics_for_feature_rows(train_rows, weights)
    train_oracle_metrics = metrics_for_feature_rows(train_rows, {"qrel_grade": 1.0})
    eval_baseline_metrics = metrics_for_feature_rows(eval_rows, baseline_weights)
    eval_calibrated_metrics = metrics_for_feature_rows(eval_rows, weights)
    eval_oracle_metrics = metrics_for_feature_rows(eval_rows, {"qrel_grade": 1.0})
    baseline_metrics = eval_baseline_metrics
    calibrated_metrics = eval_calibrated_metrics
    oracle_metrics = eval_oracle_metrics
    oracle_gap = round(float(oracle_metrics.get("nDCG@10", 0.0)) - float(baseline_metrics.get("nDCG@10", 0.0)), 6)
    realized_gain = round(float(calibrated_metrics.get("nDCG@10", 0.0)) - float(baseline_metrics.get("nDCG@10", 0.0)), 6)
    gap_closure = round(realized_gain / oracle_gap, 6) if oracle_gap > 0 else 0.0
    model = {
        "method": "retrieval_lab_calibrated_rerank_model",
        "model_type": method,
        "weights": weights,
        "feature_columns": list(feature_columns) if method != "qrels_oracle_upper" else ["qrel_grade"],
        "feature_profile": feature_profile,
        "excluded_features": sorted(excluded_features),
        "split_strategy": split_strategy,
        "train_ratio": train_ratio,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_features": str(features_path),
        "fingerprint": data_sha256({"method": method, "weights": weights, "features": str(features_path)}),
    }
    output = Path(getattr(args, "output", DEFAULT_CALIBRATED_MODEL_PATH))
    write_json(output, model)
    summary = {
        "feature_row_count": len(rows),
        "judged_row_count": sum(1 for row in rows if row.get("is_judged")),
        "method": method,
        "feature_profile": feature_profile,
        "excluded_features": sorted(excluded_features),
        "split_strategy": split_strategy,
        "train_ratio": train_ratio,
        **split_summary,
        "model": str(output),
        "baseline_nDCG@10": baseline_metrics.get("nDCG@10", 0.0),
        "calibrated_nDCG@10": calibrated_metrics.get("nDCG@10", 0.0),
        "oracle_rerank_nDCG@10": oracle_metrics.get("nDCG@10", 0.0),
        "train_baseline_nDCG@10": train_baseline_metrics.get("nDCG@10", 0.0),
        "train_calibrated_nDCG@10": train_calibrated_metrics.get("nDCG@10", 0.0),
        "train_oracle_rerank_nDCG@10": train_oracle_metrics.get("nDCG@10", 0.0),
        "eval_baseline_nDCG@10": eval_baseline_metrics.get("nDCG@10", 0.0),
        "eval_calibrated_nDCG@10": eval_calibrated_metrics.get("nDCG@10", 0.0),
        "eval_oracle_rerank_nDCG@10": eval_oracle_metrics.get("nDCG@10", 0.0),
        "train_eval_gap_ndcg_at_10": round(float(train_calibrated_metrics.get("nDCG@10", 0.0)) - float(eval_calibrated_metrics.get("nDCG@10", 0.0)), 6),
        "rerank_oracle_gap_ndcg_at_10": oracle_gap,
        "rerank_realized_gain_ndcg_at_10": realized_gain,
        "rerank_gap_closure_rate": gap_closure,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report = {
        "method": "retrieval_lab_calibrated_rerank",
        "summary": summary,
        "baseline_metrics": baseline_metrics,
        "calibrated_metrics": calibrated_metrics,
        "oracle_metrics": oracle_metrics,
        "train_metrics": {"baseline": train_baseline_metrics, "calibrated": train_calibrated_metrics, "oracle": train_oracle_metrics},
        "eval_metrics": {"baseline": eval_baseline_metrics, "calibrated": eval_calibrated_metrics, "oracle": eval_oracle_metrics},
        "weights": weights,
        "feature_ablation": feature_ablation(eval_rows, weights, baseline=calibrated_metrics),
        "fingerprint": data_sha256({"summary": summary, "weights": weights}),
    }
    report_output = Path(getattr(args, "report_output", DEFAULT_CALIBRATED_REPORT_PATH))
    write_json(report_output, report)
    return {"method": report["method"], "output": str(output), "summary": summary}


def apply_calibrated_rerank_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    model_path = Path(getattr(args, "model"))
    top_k = int(getattr(args, "top_k", 10))
    source = read_json(runs_path)
    extraction = extract_run_rows_from_report(source, source_path=runs_path)
    model = read_json(model_path)
    weights = {str(key): float(value) for key, value in dict(model.get("weights", {})).items()}
    feature_profile = normalize_feature_profile(model.get("feature_profile", FEATURE_PROFILE_DIAGNOSTIC))
    reranked: dict[str, list[dict[str, Any]]] = {}
    for run_name, rows in extraction["run_rows"].items():
        output_name = unique_run_name(reranked, f"{run_name}::calibrated_{model.get('model_type', 'linear')}")
        reranked[output_name] = [rerank_row_with_weights(row, weights=weights, top_k=top_k, feature_profile=feature_profile) for row in rows]
    cases = cases_from_run_rows(reranked)
    summary = {
        **run_artifact_summary(reranked, cases),
        "runs": str(runs_path),
        "model": str(model_path),
        "top_k": top_k,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    artifact = {
        "method": "retrieval_lab_calibrated_reranked_runs",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_runs": str(runs_path),
        "model": str(model_path),
        "run_rows": reranked,
        "cases": cases,
        "summary": summary,
        "fingerprint": data_sha256({"run_rows": reranked, "model": model.get("fingerprint")}),
    }
    output = Path(getattr(args, "output", DEFAULT_CALIBRATED_RUNS_PATH))
    write_json(output, artifact)
    return {"method": artifact["method"], "output": str(output), "summary": {**summary, "output": str(output)}}


def attribute_rerank_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    qrels_path = optional_path(getattr(args, "qrels", None))
    features_path = optional_path(getattr(args, "features", None))
    source = read_json(runs_path)
    extraction = extract_run_rows_from_report(source, source_path=runs_path)
    run_rows = extraction["run_rows"]
    qrels = load_qrels(qrels_path) if qrels_path and qrels_path.exists() else []
    feature_rows = read_jsonl(features_path) if features_path and features_path.exists() else []
    qrels_by_key = qrels_index(qrels)
    features_by_key = feature_index(feature_rows)
    failures = []
    for run_name, rows in run_rows.items():
        for row in rows:
            failure = attribution_for_row(run_name, row, qrels_by_key=qrels_by_key, features_by_key=features_by_key)
            if failure:
                failures.append(failure)
    counts = Counter(failure["failure_category"] for failure in failures)
    summary = {
        "runs": str(runs_path),
        "qrels": str(qrels_path or ""),
        "features": str(features_path or ""),
        "row_level_available": bool(run_rows),
        "feature_row_count": len(feature_rows),
        "failure_count": len(failures),
        "top_failure_types": dict(counts.most_common()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report = {
        "method": "retrieval_lab_rerank_attribution",
        "summary": summary,
        "failures": failures[: int(getattr(args, "max_failures", 200) or 200)],
        "fingerprint": data_sha256({"summary": summary, "failures": failures}),
    }
    output = Path(getattr(args, "output", DEFAULT_RERANK_ATTRIBUTION_PATH))
    write_json(output, report)
    markdown_output = getattr(args, "markdown_output", None)
    if markdown_output is not None:
        Path(markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_output).write_text(rerank_attribution_markdown(report), encoding="utf-8")
    return {"method": report["method"], "output": str(output), "summary": summary}


def feature_rows_from_runs(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    qrels: list[dict[str, Any]],
    feature_profile: str = FEATURE_PROFILE_DIAGNOSTIC,
) -> list[dict[str, Any]]:
    feature_profile = normalize_feature_profile(feature_profile)
    qrels_by_key = qrels_index(qrels)
    rows = []
    for run_name, run in run_rows.items():
        for query_row in run:
            query_id = str(query_row.get("case_id", ""))
            candidates = list(query_row.get("all_results") or query_row.get("top_results") or [])
            for rank, result in enumerate(candidates, start=1):
                item_id = str(result.get("item_id", ""))
                qrel = qrels_by_key.get((query_id, item_id), {})
                rows.append(feature_row(run_name, query_row, result, rank=rank, qrel=qrel, feature_profile=feature_profile))
    return rows


def feature_row(
    run_name: str,
    query_row: dict[str, Any],
    result: dict[str, Any],
    *,
    rank: int,
    qrel: dict[str, Any],
    feature_profile: str = FEATURE_PROFILE_DIAGNOSTIC,
) -> dict[str, Any]:
    feature_profile = normalize_feature_profile(feature_profile)
    item_id = str(result.get("item_id", ""))
    query_id = str(query_row.get("case_id", ""))
    components = result.get("workflow_score_components") if isinstance(result.get("workflow_score_components"), dict) else {}
    if not components:
        try:
            components = workflow_score_components(result, row=query_row, ranking_key=str(result.get("ranking_key") or query_row.get("ranking_key") or "hybrid_rrf_constraints_signature"))
        except Exception:
            components = {}
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    stage_match = candidate_stage_match(query_row, metadata, hits, feature_profile=feature_profile)
    purpose_match = candidate_purpose_match(query_row, metadata, feature_profile=feature_profile)
    negative_hit = int(bool(hits.get("negative_style")))
    forbidden_hit = int(bool(hits.get("forbidden_stage")) or has_forbidden_stage_hit(query_row, result))
    diagnostic_is_target = item_id == str(query_row.get("target_item_id", ""))
    is_target = diagnostic_is_target if feature_profile == FEATURE_PROFILE_DIAGNOSTIC else False
    grade = int(qrel.get("grade", -1)) if qrel else -1
    original_score = numeric(result.get("final_score", result.get("score", 0.0)))
    target_like = target_like_score(is_target=is_target, stage_match=stage_match, purpose_match=purpose_match) if feature_profile == FEATURE_PROFILE_DIAGNOSTIC else 0.0
    return {
        "query_id": query_id,
        "item_id": item_id,
        "fixture_id": fixture_id_from_item_id(item_id),
        "query_fixture_id": fixture_id_from_item_id(str(query_row.get("target_item_id", ""))),
        "rank": rank,
        "feature_profile": feature_profile,
        "qrel_grade": grade if grade >= 0 else None,
        "qrel_source": str(qrel.get("source", "")) if qrel else "",
        "qrel_confidence": qrel_confidence(qrel) if qrel else None,
        "qrel_conflict": qrel_has_vote_conflict(qrel) if qrel else False,
        "workflow": run_name,
        "ranking_key": str(result.get("ranking_key", "")),
        "original_score": original_score,
        "lexical_score": numeric(result.get("lexical_score", components.get("lexical", 0.0))),
        "semantic_score": numeric(result.get("embedding_score", result.get("semantic_score", components.get("semantic", 0.0)))),
        "rrf_score": numeric(result.get("rrf_score", components.get("rrf", 0.0))),
        "constraint_score": numeric(result.get("constraint_score", components.get("constraint", 0.0))),
        "signature_score": numeric(result.get("signature_score", components.get("signature", 0.0))),
        "stage_match": stage_match,
        "purpose_match": purpose_match,
        "style_risk_score": float(style_risk_score(result)),
        "style_violation": float(negative_hit > 0),
        "negative_style_hit": float(negative_hit),
        "forbidden_stage_hit": float(forbidden_hit),
        "target_like_score": target_like,
        "graph_neighbor_score": graph_neighbor_score(query_row, metadata),
        "is_target": is_target,
        "is_judged": bool(qrel),
    }


def feature_export_summary(rows: list[dict[str, Any]], *, qrels: list[dict[str, Any]]) -> dict[str, Any]:
    grade_counts = Counter(str(row.get("qrel_grade")) for row in rows if row.get("qrel_grade") is not None)
    source_counts = Counter(str(row.get("qrel_source", "")) for row in rows if row.get("qrel_source"))
    missing_counts = {feature: sum(1 for row in rows if row.get(feature) is None) for feature in FEATURE_COLUMNS}
    stats = {feature: numeric_stats([numeric(row.get(feature)) for row in rows]) for feature in FEATURE_COLUMNS}
    return {
        "feature_row_count": len(rows),
        "judged_row_count": sum(1 for row in rows if row.get("is_judged")),
        "feature_profiles": dict(sorted(Counter(str(row.get("feature_profile", "")) for row in rows).items())),
        "positive_grade_count": sum(1 for row in rows if numeric(row.get("qrel_grade"), -1) > 0),
        "negative_grade_count": sum(1 for row in rows if numeric(row.get("qrel_grade"), -1) == 0),
        "grade_counts": dict(sorted(grade_counts.items())),
        "qrels_count": len(qrels),
        "qrels_source_distribution": dict(sorted(source_counts.items())),
        "missing_feature_counts": missing_counts,
        "feature_stats": stats,
    }


def compact_workflow_summary(source: dict[str, Any]) -> dict[str, Any]:
    workflows = source.get("workflows", {}) if isinstance(source, dict) else {}
    if not isinstance(workflows, dict):
        return {}
    result = {}
    for name, payload in workflows.items():
        if isinstance(payload, dict):
            result[str(name)] = payload.get("summary", {}) or payload.get("metrics", {}).get("overall", {})
    return result


def coordinate_search_weights(
    rows: list[dict[str, Any]],
    *,
    initial: dict[str, float],
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, float]:
    weights = dict(initial)
    baseline = metrics_for_feature_rows(rows, weights).get("nDCG@10", 0.0)
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for feature in feature_columns:
        best_value = weights.get(feature, 0.0)
        best_score = baseline
        for value in candidates:
            trial = dict(weights)
            trial[feature] = value
            score = metrics_for_feature_rows(rows, trial).get("nDCG@10", 0.0)
            if score > best_score:
                best_score = score
                best_value = value
        weights[feature] = best_value
        baseline = best_score
    return {key: round(value, 6) for key, value in weights.items() if abs(value) > 1e-9}


def metrics_for_feature_rows(rows: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qrels = []
    for row in rows:
        query_id = str(row.get("query_id", ""))
        item_id = str(row.get("item_id", ""))
        if not query_id or not item_id:
            continue
        scored = dict(row)
        scored["score"] = feature_score(row, weights)
        grouped[query_id].append(scored)
        if row.get("qrel_grade") is not None:
            qrels.append({"query_id": query_id, "item_id": item_id, "grade": int(row.get("qrel_grade", 0) or 0)})
    run_rows = []
    for query_id, query_rows in grouped.items():
        ranked = sorted(query_rows, key=lambda row: (-numeric(row.get("score")), int(row.get("rank", 999999))))
        run_rows.append({"case_id": query_id, "top_results": [{"item_id": row["item_id"]} for row in ranked]})
    if not run_rows or not qrels:
        return {"nDCG@3": 0.0, "nDCG@10": 0.0, "MRR@10": 0.0, "Recall@10": 0.0, "case_count": len(run_rows), "qrels_count": len(qrels)}
    metrics = graded_metrics(run_rows, qrels, top_k=10)
    metrics["case_count"] = len(run_rows)
    metrics["qrels_count"] = len(qrels)
    return metrics


def feature_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    return round(sum(float(weight) * numeric(row.get(feature)) for feature, weight in weights.items()), 6)


def feature_ablation(rows: list[dict[str, Any]], weights: dict[str, float], *, baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    result = {}
    base_ndcg = float(baseline.get("nDCG@10", 0.0))
    for feature in sorted(weights):
        trial = dict(weights)
        trial.pop(feature, None)
        metrics = metrics_for_feature_rows(rows, trial)
        result[feature] = {
            "nDCG@10": metrics.get("nDCG@10", 0.0),
            "delta": round(float(metrics.get("nDCG@10", 0.0)) - base_ndcg, 6),
        }
    return result


def rerank_row_with_weights(
    row: dict[str, Any],
    *,
    weights: dict[str, float],
    top_k: int,
    feature_profile: str = FEATURE_PROFILE_DIAGNOSTIC,
) -> dict[str, Any]:
    candidates = list(row.get("all_results") or row.get("top_results") or [])
    scored = []
    for index, result in enumerate(candidates):
        features = feature_row("calibrated", row, result, rank=index + 1, qrel={}, feature_profile=feature_profile)
        score = feature_score(features, weights)
        copied = dict(result)
        copied["score"] = score
        copied["final_score"] = score
        copied["ranking_key"] = "calibrated_rerank"
        copied["calibrated_features"] = {feature: features.get(feature) for feature in FEATURE_COLUMNS}
        scored.append((score, index, copied))
    ranked = [result for _score, _index, result in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]]
    return row_with_reranked_results(row, ranked, ranking_key="calibrated_rerank")


def attribution_for_row(
    run_name: str,
    row: dict[str, Any],
    *,
    qrels_by_key: dict[tuple[str, str], dict[str, Any]],
    features_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    query_id = str(row.get("case_id", ""))
    results = list(row.get("top_results", []))
    if not query_id or not results:
        return None
    top1 = results[0]
    top1_qrel = qrels_by_key.get((query_id, str(top1.get("item_id", ""))), {})
    judged = [(result, qrels_by_key.get((query_id, str(result.get("item_id", ""))), {})) for result in results]
    judged = [(result, qrel) for result, qrel in judged if qrel]
    if not judged:
        return None
    oracle_result, oracle_qrel = max(judged, key=lambda pair: (int(pair[1].get("grade", 0)), -rank_of(results, pair[0])))
    if int(top1_qrel.get("grade", -1)) >= int(oracle_qrel.get("grade", 0)):
        return None
    top1_item = str(top1.get("item_id", ""))
    oracle_item = str(oracle_result.get("item_id", ""))
    top1_features = features_by_key.get((run_name, query_id, top1_item), {})
    oracle_features = features_by_key.get((run_name, query_id, oracle_item), {})
    return {
        "query_id": query_id,
        "run": run_name,
        "top1_item_id": top1_item,
        "top1_grade": int(top1_qrel.get("grade", -1)) if top1_qrel else None,
        "oracle_item_id": oracle_item,
        "oracle_grade": int(oracle_qrel.get("grade", 0)),
        "target_item_id": row.get("target_item_id"),
        "target_rank": row.get("target_rank"),
        "best_judged_candidate_rank": rank_of(results, oracle_result),
        "top1_why_won": top_feature_reasons(top1_features),
        "target_why_lost": feature_delta(top1_features, oracle_features),
        "feature_deltas": feature_delta(top1_features, oracle_features),
        "qrels_confidence": qrel_confidence(oracle_qrel),
        "qrels_conflict": qrel_has_vote_conflict(oracle_qrel) or qrel_has_vote_conflict(top1_qrel),
        "failure_category": classify_attribution_failure(top1_features, oracle_features, top1_qrel, oracle_qrel),
    }


def classify_attribution_failure(top1_features: dict[str, Any], oracle_features: dict[str, Any], top1_qrel: dict[str, Any], oracle_qrel: dict[str, Any]) -> str:
    if qrel_has_vote_conflict(top1_qrel) or qrel_has_vote_conflict(oracle_qrel):
        return "qrels_conflict"
    top1_grade = int(top1_qrel.get("grade", -1)) if top1_qrel else -1
    oracle_grade = int(oracle_qrel.get("grade", 0))
    if top1_grade >= 2 and oracle_grade > top1_grade:
        return "qrels_preference_boundary"
    if not top1_features or not oracle_features:
        return "feature_missing"
    if numeric(top1_features.get("style_violation")) > numeric(oracle_features.get("style_violation")):
        return "style_penalty_over_or_under_applied"
    if numeric(oracle_features.get("stage_match")) > numeric(top1_features.get("stage_match")) or numeric(oracle_features.get("purpose_match")) > numeric(top1_features.get("purpose_match")):
        return "stage_purpose_mismatch"
    if oracle_grade <= 2:
        return "ambiguous_multi_valid_answer"
    return "feature_weight_misaligned"


def rerank_attribution_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = ["# Rerank Attribution Report", ""]
    lines.extend(["## Summary", ""])
    for key in ("failure_count", "feature_row_count", "row_level_available", "elapsed_seconds"):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "## Top Failure Types", "", "| type | count |", "|---|---:|"])
    for key, value in dict(summary.get("top_failure_types", {})).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Examples", ""])
    for failure in report.get("failures", [])[:10]:
        lines.append(f"- `{failure.get('query_id')}` top1 `{failure.get('top1_item_id')}` lost to `{failure.get('oracle_item_id')}` ({failure.get('failure_category')})")
    return "\n".join(lines) + "\n"


def qrels_index(qrels: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row.get("query_id", "")), str(row.get("item_id", ""))): row for row in qrels}


def feature_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(str(row.get("workflow", "")), str(row.get("query_id", "")), str(row.get("item_id", ""))): row for row in rows}


def candidate_stage_match(
    query_row: dict[str, Any],
    metadata: dict[str, Any],
    hits: dict[str, Any],
    *,
    feature_profile: str = FEATURE_PROFILE_DIAGNOSTIC,
) -> float:
    if hits.get("desired_stage"):
        return 1.0
    if feature_profile == FEATURE_PROFILE_PRODUCTION_SAFE:
        plan = query_row.get("query_plan", {}) if isinstance(query_row.get("query_plan"), dict) else {}
        desired = {canonical_stage(str(value)) for value in plan.get("desired_stage", []) or []}
        result_stage = canonical_stage(str(metadata.get("script_stage", "")))
        return 1.0 if result_stage and result_stage in desired else 0.0
    target_stage = canonical_stage(str(query_row.get("target_stage", "")))
    result_stage = canonical_stage(str(metadata.get("script_stage", "")))
    return 1.0 if target_stage and target_stage == result_stage else 0.0


def candidate_purpose_match(
    query_row: dict[str, Any],
    metadata: dict[str, Any],
    *,
    feature_profile: str = FEATURE_PROFILE_DIAGNOSTIC,
) -> float:
    if feature_profile == FEATURE_PROFILE_PRODUCTION_SAFE:
        plan = query_row.get("query_plan", {}) if isinstance(query_row.get("query_plan"), dict) else {}
        target = {str(value) for value in plan.get("positive_purposes", []) or []}
    else:
        target = {str(value) for value in query_row.get("target_purposes", []) or []}
    result = {str(value) for value in metadata.get("creative_purpose", []) or []}
    if not target or not result:
        return 0.0
    return round(len(target & result) / max(1, len(target)), 6)


def target_like_score(*, is_target: bool, stage_match: float, purpose_match: float) -> float:
    return round((1.0 if is_target else 0.0) + 0.3 * stage_match + 0.4 * purpose_match, 6)


def graph_neighbor_score(query_row: dict[str, Any], metadata: dict[str, Any]) -> float:
    plan = query_row.get("query_plan", {}) if isinstance(query_row.get("query_plan"), dict) else {}
    plan_signature = plan.get("scene_signature", {}) if isinstance(plan.get("scene_signature"), dict) else {}
    item_signature = metadata.get("scene_signature", {}) if isinstance(metadata.get("scene_signature"), dict) else {}
    overlaps = 0
    total = 0
    for key, values in plan_signature.items():
        plan_values = {str(value).lower() for value in values} if isinstance(values, list) else {str(values).lower()}
        item_values_raw = item_signature.get(key, []) if isinstance(item_signature, dict) else []
        item_values = {str(value).lower() for value in item_values_raw} if isinstance(item_values_raw, list) else {str(item_values_raw).lower()}
        if plan_values:
            total += 1
            if plan_values & item_values:
                overlaps += 1
    return round(overlaps / total, 6) if total else 0.0


def top_feature_reasons(features: dict[str, Any]) -> list[str]:
    if not features:
        return []
    candidates = []
    for feature in ("original_score", "rrf_score", "constraint_score", "signature_score", "stage_match", "purpose_match", "graph_neighbor_score"):
        value = numeric(features.get(feature))
        if value > 0:
            candidates.append((value, feature))
    return [feature for _value, feature in sorted(candidates, reverse=True)[:4]]


def feature_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    keys = set(FEATURE_COLUMNS) | {"qrel_grade"}
    return {key: round(numeric(right.get(key)) - numeric(left.get(key)), 6) for key in sorted(keys) if left or right}


def rank_of(results: list[dict[str, Any]], target: dict[str, Any]) -> int:
    item_id = str(target.get("item_id", ""))
    for index, result in enumerate(results, start=1):
        if str(result.get("item_id", "")) == item_id:
            return index
    return 999999


def numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def numeric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": round(min(values), 6), "max": round(max(values), 6), "mean": round(sum(values) / len(values), 6)}


def normalize_feature_profile(value: Any) -> str:
    profile = str(value or FEATURE_PROFILE_DIAGNOSTIC).strip()
    if profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown feature profile: {profile}")
    return profile


def profile_excluded_features(feature_profile: str) -> set[str]:
    if normalize_feature_profile(feature_profile) == FEATURE_PROFILE_PRODUCTION_SAFE:
        return set(PRODUCTION_SAFE_EXCLUDED_FEATURES)
    return set()


def infer_feature_profile(rows: list[dict[str, Any]]) -> str:
    profiles = {str(row.get("feature_profile", "")) for row in rows if row.get("feature_profile")}
    if len(profiles) == 1:
        return next(iter(profiles))
    return FEATURE_PROFILE_DIAGNOSTIC


def normalize_split_strategy(value: Any) -> str:
    strategy = str(value or "none").strip()
    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split strategy: {strategy}")
    return strategy


def bounded_ratio(value: float) -> float:
    return min(1.0, max(0.05, float(value)))


def csv_set(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def weights_without_features(weights: dict[str, float], excluded_features: set[str]) -> dict[str, float]:
    return {feature: weight for feature, weight in weights.items() if feature not in excluded_features}


def split_feature_rows(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    train_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    strategy = normalize_split_strategy(strategy)
    if strategy == "none" or not rows:
        groups = {str(row.get("query_id", "")) for row in rows if row.get("query_id")}
        return list(rows), list(rows), {
            "train_row_count": len(rows),
            "eval_row_count": len(rows),
            "split_group_count": len(groups),
            "train_group_count": len(groups),
            "eval_group_count": len(groups),
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[feature_split_group(row, strategy)].append(row)
    keys = sorted(grouped, key=lambda key: stable_hash(key))
    if len(keys) <= 1:
        train_keys = set(keys)
        eval_keys = set(keys)
    else:
        train_count = int(round(len(keys) * train_ratio))
        train_count = min(len(keys) - 1, max(1, train_count))
        train_keys = set(keys[:train_count])
        eval_keys = set(keys[train_count:])
    train_rows = [row for key in keys if key in train_keys for row in grouped[key]]
    eval_rows = [row for key in keys if key in eval_keys for row in grouped[key]]
    return train_rows, eval_rows, {
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "split_group_count": len(keys),
        "train_group_count": len(train_keys),
        "eval_group_count": len(eval_keys),
    }


def feature_split_group(row: dict[str, Any], strategy: str) -> str:
    if strategy == "fixture_holdout":
        return str(row.get("query_fixture_id") or row.get("fixture_id") or row.get("query_id") or "")
    return str(row.get("query_id") or "")


def stable_hash(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def fixture_id_from_item_id(item_id: str) -> str:
    return str(item_id).split("::", 1)[0] if "::" in str(item_id) else ""


def optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if str(path) in {"", "."}:
        return None
    return path


__all__ = [
    "DEFAULT_CALIBRATED_MODEL_PATH",
    "DEFAULT_CALIBRATED_REPORT_PATH",
    "DEFAULT_CALIBRATED_RUNS_PATH",
    "DEFAULT_RERANK_ATTRIBUTION_PATH",
    "DEFAULT_RERANK_FEATURE_REPORT_PATH",
    "DEFAULT_RERANK_FEATURES_PATH",
    "apply_calibrated_rerank_command",
    "attribute_rerank_command",
    "calibrate_rerank_command",
    "export_rerank_features_command",
    "feature_rows_from_runs",
    "metrics_for_feature_rows",
]
