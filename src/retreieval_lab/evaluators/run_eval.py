from __future__ import annotations

from pathlib import Path
from collections import Counter
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, read_json, write_json
from retreieval_lab.evaluators.metrics import graded_metrics
from retreieval_lab.experiments.runs import cases_from_run_rows, run_artifact_summary
from retreieval_lab.qrels import load_qrels


DEFAULT_RUN_EVAL_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "run_eval_latest.json"


def evaluate_run_artifact_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    runs_path = Path(getattr(args, "runs"))
    qrels_path = Path(getattr(args, "qrels"))
    top_k = int(getattr(args, "top_k", 10))
    source = read_json(runs_path)
    run_rows = source.get("run_rows", {}) if isinstance(source, dict) else {}
    if not isinstance(run_rows, dict):
        raise ValueError("runs artifact must contain a run_rows mapping")
    qrels = load_qrels(qrels_path)
    if not qrels:
        raise ValueError("--qrels must point to a non-empty qrels JSONL file")
    run_metrics = evaluate_run_rows({str(name): list(rows) for name, rows in run_rows.items()}, qrels=qrels, top_k=top_k)
    baseline_run = select_baseline_run(run_metrics, str(getattr(args, "baseline_run", "") or ""))
    best_run = max(run_metrics, key=lambda name: run_metric_selection_score(run_metrics[name])) if run_metrics else ""
    deltas = {
        name: run_metric_delta(metrics, run_metrics.get(baseline_run, {}))
        for name, metrics in run_metrics.items()
    }
    cases = cases_from_run_rows(run_rows)
    variant_metrics = run_metrics_by_variant(
        {str(name): list(rows) for name, rows in run_rows.items()},
        qrels=qrels,
        top_k=top_k,
    )
    summary = {
        **run_artifact_summary(run_rows, cases),
        "qrels_count": len(qrels),
        "top_k": top_k,
        "baseline_run": baseline_run,
        "best_run": best_run,
        "best_selection_score": run_metric_selection_score(run_metrics.get(best_run, {})) if best_run else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    report = {
        "method": "retrieval_lab_run_evaluation",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_runs": str(runs_path),
        "qrels": str(qrels_path),
        "top_k": top_k,
        "summary": summary,
        "graded_metrics": run_metrics.get(best_run, {}),
        "run_metrics": run_metrics,
        "delta_vs_baseline": deltas,
        "by_variant_type": variant_metrics,
        "fingerprint": data_sha256({"run_rows": run_rows, "qrels": qrels, "top_k": top_k}),
    }
    output = Path(getattr(args, "output", DEFAULT_RUN_EVAL_REPORT_PATH))
    write_json(output, report)
    return {
        "method": "retrieval_lab_evaluate_run_artifact",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": report["fingerprint"]},
    }


def evaluate_run_rows(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    qrels: list[dict[str, Any]],
    top_k: int,
) -> dict[str, dict[str, float]]:
    return {
        run_name: graded_metrics(rows, qrels, top_k=top_k)
        for run_name, rows in run_rows.items()
    }


def run_metrics_by_variant(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    qrels: list[dict[str, Any]],
    top_k: int,
) -> dict[str, dict[str, dict[str, float]]]:
    variants = sorted(
        {
            str(row.get("variant_type", "default") or "default")
            for rows in run_rows.values()
            for row in rows
        }
    )
    if variants == ["default"]:
        return {}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for run_name, rows in run_rows.items():
        by_variant = {}
        for variant in variants:
            variant_rows = [row for row in rows if str(row.get("variant_type", "default") or "default") == variant]
            if variant_rows:
                by_variant[variant] = graded_metrics(variant_rows, qrels, top_k=top_k)
        result[run_name] = by_variant
    return result


def select_baseline_run(run_metrics: dict[str, dict[str, float]], baseline_run: str) -> str:
    if baseline_run in run_metrics:
        return baseline_run
    return next(iter(run_metrics), "")


def run_metric_selection_score(metrics: dict[str, Any]) -> float:
    parts = [
        float(metrics.get("nDCG@10", 0.0)),
        float(metrics.get("MRR@10", 0.0)),
        float(metrics.get("Recall@10", 0.0)),
        1.0 - float(metrics.get("Unjudged@10", 0.0)),
    ]
    return round(sum(parts) / len(parts), 6)


def run_metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("nDCG@3", "nDCG@10", "ERR@10", "MRR@10", "Judged@10", "Unjudged@10", "Recall@10")
    return {
        key: round(float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6)
        for key in keys
    }


def run_eval_failure_summary(run_metrics: dict[str, dict[str, float]]) -> dict[str, int]:
    counts = Counter()
    for metrics in run_metrics.values():
        if float(metrics.get("Judged@10", 0.0)) < 0.5:
            counts["low_judged_coverage"] += 1
        if float(metrics.get("Recall@10", 0.0)) < 0.5:
            counts["low_recall"] += 1
        if float(metrics.get("nDCG@10", 0.0)) < 0.5:
            counts["low_ndcg"] += 1
    return dict(sorted(counts.items()))


__all__ = [
    "DEFAULT_RUN_EVAL_REPORT_PATH",
    "evaluate_run_artifact_command",
    "evaluate_run_rows",
    "run_metric_delta",
    "run_metric_selection_score",
    "run_metrics_by_variant",
]
