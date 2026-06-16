from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, write_json


DEFAULT_EXPERIMENT_COMPARISON_PATH = Path(".tmp") / "retrieval_lab" / "experiment_comparison_latest.json"

POSITIVE_METRICS = (
    "nDCG@10",
    "nDCG@3",
    "MRR@10",
    "ERR@10",
    "Recall@10",
    "scene_level_recall_at_10",
    "target_recall_at_10",
    "stage_level_hit_at_3",
    "purpose_level_hit_at_3",
    "hard_negative_expected_prefer_accuracy",
)
NEGATIVE_METRICS = (
    "style_violation_at_3",
    "forbidden_stage_violation_at_3",
    "Unjudged@10",
    "failure_rate",
    "low_confidence_rate",
    "negative_leak_rate",
)
KEY_DELTA_METRICS = (
    "nDCG@10",
    "MRR@10",
    "scene_level_recall_at_10",
    "target_recall_at_10",
    "stage_level_hit_at_3",
    "purpose_level_hit_at_3",
    "style_violation_at_3",
    "failure_rate",
)


def compare_experiments_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    reports = [experiment_report_row(Path(path), label=str(index + 1)) for index, path in enumerate(getattr(args, "reports", []))]
    baseline = select_baseline(reports, str(getattr(args, "baseline", "") or ""))
    best = max(reports, key=lambda row: row["selection_score"]) if reports else None
    for row in reports:
        row["delta_vs_baseline"] = metric_delta(row.get("metrics", {}), baseline.get("metrics", {}) if baseline else {})
    summary = {
        "report_count": len(reports),
        "methods": [row.get("method") for row in reports],
        "baseline_path": baseline.get("path") if baseline else "",
        "best_path": best.get("path") if best else "",
        "best_method": best.get("method") if best else "",
        "best_selection_score": best.get("selection_score") if best else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    report = {
        "method": "retrieval_lab_experiment_comparison",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "report_count": len(reports),
        "reports": reports,
        "summary": summary,
        "fingerprint": data_sha256(reports),
    }
    output = Path(getattr(args, "output", DEFAULT_EXPERIMENT_COMPARISON_PATH))
    write_json(output, report)
    return {
        "method": "retrieval_lab_compare_experiments",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": report["fingerprint"]},
    }


def experiment_report_row(path: Path, *, label: str = "") -> dict[str, Any]:
    data = read_json(path)
    metrics = extract_report_metrics(data)
    return {
        "label": label,
        "path": str(path),
        "method": data.get("method", ""),
        "summary": data.get("summary", data.get("metrics", {})),
        "experiment": data.get("experiment", {}),
        "metrics": metrics,
        "selection_score": experiment_selection_score(metrics),
    }


def extract_report_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    containers = [
        report.get("summary", {}),
        report.get("metrics", {}),
        report.get("graded_metrics", {}),
        report.get("fuzzy_metrics", {}),
    ]
    overall = report.get("metrics", {}).get("overall") if isinstance(report.get("metrics"), dict) else None
    if isinstance(overall, dict):
        containers.append(overall)
    for container in containers:
        if isinstance(container, dict):
            absorb_numeric_metrics(metrics, container)
    capabilities = report.get("capabilities", {})
    if isinstance(capabilities, dict):
        for name, row in capabilities.items():
            if isinstance(row, dict) and "score" in row:
                metrics[f"capability.{name}"] = round(float(row.get("score", 0.0)), 6)
    return dict(sorted(metrics.items()))


def absorb_numeric_metrics(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, bool) or isinstance(value, (dict, list)):
            continue
        try:
            target[str(key)] = round(float(value), 6)
        except (TypeError, ValueError):
            continue


def experiment_selection_score(metrics: dict[str, float]) -> float:
    score = 0.0
    weights = 0.0
    for key in POSITIVE_METRICS:
        if key in metrics:
            score += metrics[key]
            weights += 1.0
    for key in NEGATIVE_METRICS:
        if key in metrics:
            score += 1.0 - metrics[key]
            weights += 1.0
    if "capability.retrieval_quality" in metrics:
        score += metrics["capability.retrieval_quality"] / 100.0
        weights += 1.0
    if "capability.qrels_trust" in metrics:
        score += metrics["capability.qrels_trust"] / 100.0
        weights += 0.5
    return round(score / max(1.0, weights), 6)


def select_baseline(reports: list[dict[str, Any]], baseline: str) -> dict[str, Any] | None:
    if not reports:
        return None
    if not baseline:
        return reports[0]
    for row in reports:
        if baseline in {row.get("label"), row.get("path"), row.get("method")}:
            return row
    return reports[0]


def metric_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    keys = [key for key in KEY_DELTA_METRICS if key in metrics or key in baseline]
    if not keys:
        keys = sorted((set(metrics) | set(baseline)) & (set(POSITIVE_METRICS) | set(NEGATIVE_METRICS)))
    return {
        key: round(float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6)
        for key in keys
    }


__all__ = [
    "DEFAULT_EXPERIMENT_COMPARISON_PATH",
    "compare_experiments_command",
    "experiment_report_row",
    "experiment_selection_score",
    "extract_report_metrics",
    "metric_delta",
]
