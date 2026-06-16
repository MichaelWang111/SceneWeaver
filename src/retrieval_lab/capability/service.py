from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any


DEFAULT_CAPABILITY_REGISTRY_PATH = Path(".tmp") / "capability_cycles.jsonl"
DEFAULT_CAPABILITY_CYCLE_PATH = Path(".tmp") / "capability_cycle_latest.json"
DEFAULT_CAPABILITY_REPORT_PATH = Path(".tmp") / "capability_report.md"
DEFAULT_CAPABILITY_CHART_DIR = Path(".tmp") / "capability_charts"
DEFAULT_CORE_METRICS_TREND_HTML = Path(".tmp") / "core_metrics_trend.html"
DEFAULT_DIAGNOSTIC_METRICS_TREND_HTML = Path(".tmp") / "diagnostic_metrics_trend.html"


@dataclass(frozen=True)
class CapabilityCycleArgs:
    cycle_id: str = ""
    label: str = ""
    reports: list[Path] | None = None
    registry: Path = DEFAULT_CAPABILITY_REGISTRY_PATH
    output: Path = DEFAULT_CAPABILITY_CYCLE_PATH
    as_origin: bool = False


@dataclass(frozen=True)
class CapabilityReportArgs:
    registry: Path = DEFAULT_CAPABILITY_REGISTRY_PATH
    output: Path = DEFAULT_CAPABILITY_REPORT_PATH
    chart_dir: Path = DEFAULT_CAPABILITY_CHART_DIR


CAPABILITY_METRIC_ALIASES = {
    "nDCG@10": "ndcg_at_10",
    "MRR@10": "mrr_at_10",
    "Recall@10": "recall_at_10",
    "scene_level_recall_at_10": "scene_level_recall_at_10",
    "target_recall_at_10": "target_recall_at_10",
    "stage_level_hit_at_3": "stage_level_hit_at_3",
    "stage_hit_at_3": "stage_level_hit_at_3",
    "purpose_level_hit_at_3": "purpose_level_hit_at_3",
    "purpose_hit_at_3": "purpose_level_hit_at_3",
    "style_violation_at_3": "style_violation_at_3",
    "failure_rate": "failure_rate",
    "rerank_opportunity_nDCG@10": "rerank_opportunity_ndcg_at_10",
    "oracle_rerank_nDCG@10": "oracle_rerank_ndcg_at_10",
    "baseline_nDCG@10": "baseline_ndcg_at_10",
    "calibrated_nDCG@10": "calibrated_ndcg_at_10",
    "rule_rerank_nDCG@10": "rule_rerank_ndcg_at_10",
    "llm_rerank_nDCG@10": "llm_rerank_ndcg_at_10",
    "gated_sample_llm_nDCG@10": "llm_rerank_ndcg_at_10",
    "gated_sample_oracle_nDCG@10": "oracle_rerank_ndcg_at_10",
    "llm_delta_nDCG@10": "rerank_realized_gain_ndcg_at_10",
    "gated_sample_llm_delta_nDCG@10": "rerank_realized_gain_ndcg_at_10",
    "oracle_delta_nDCG@10": "rerank_oracle_gap_ndcg_at_10",
    "rerank_oracle_gap_ndcg_at_10": "rerank_oracle_gap_ndcg_at_10",
    "rerank_realized_gain_ndcg_at_10": "rerank_realized_gain_ndcg_at_10",
    "rerank_gap_closure_rate": "rerank_gap_closure_rate",
    "qrels_trust_level": "qrels_trust_level",
    "manual_or_llm_count": "manual_or_llm_count",
    "manual_count": "manual_count",
    "llm_count": "llm_count",
    "bootstrap_only_count": "bootstrap_only_count",
    "needs_adjudication_count": "needs_adjudication_count",
    "vote_conflict_rate": "vote_conflict_rate",
    "qrels_count": "qrels_count",
    "sample_count": "active_sample_count",
    "judgement_count": "llm_judgement_count",
    "natural_style_safe_metadata_leak_rate": "natural_fuzzy_metadata_leak_rate",
    "natural_style_safe_style_violation_at_3": "natural_fuzzy_style_violation_at_3",
    "natural_style_safe_scene_recall_at_10": "natural_fuzzy_scene_recall_at_10",
    "llm_call_count": "llm_call_count",
}

CORE_TREND_METRICS = (
    "ndcg_at_10",
    "mrr_at_10",
    "recall_at_10",
    "scene_level_recall_at_10",
    "target_recall_at_10",
    "stage_level_hit_at_3",
    "purpose_level_hit_at_3",
    "style_violation_at_3",
    "failure_rate",
    "qrels_trust_level",
)

CORE_TREND_LABELS = {
    "ndcg_at_10": "nDCG@10",
    "mrr_at_10": "MRR@10",
    "recall_at_10": "Recall@10",
    "scene_level_recall_at_10": "scene_level_recall_at_10",
    "target_recall_at_10": "target_recall_at_10",
    "stage_level_hit_at_3": "stage_level_hit_at_3",
    "purpose_level_hit_at_3": "purpose_level_hit_at_3",
    "style_violation_at_3": "style_violation_at_3",
    "failure_rate": "failure_rate",
    "qrels_trust_level": "qrels_trust_level",
}

DIAGNOSTIC_TREND_METRICS = (
    "ndcg_headroom_at_10",
    "rerank_oracle_gap_ndcg_at_10",
    "rerank_realized_gain_ndcg_at_10",
    "rerank_gap_closure_rate",
    "bootstrap_only_rate",
    "needs_adjudication_rate",
    "vote_conflict_rate",
    "llm_coverage_rate",
    "llm_seconds_per_judgement",
    "style_violation_gap",
    "all_fuzzy_ndcg_at_10",
    "natural_fuzzy_ndcg_at_10",
    "all_fuzzy_style_violation_at_3",
    "natural_fuzzy_style_violation_at_3",
)

DIAGNOSTIC_TREND_LABELS = {
    "ndcg_headroom_at_10": "nDCG@10 headroom",
    "rerank_oracle_gap_ndcg_at_10": "rerank oracle gap nDCG@10",
    "rerank_realized_gain_ndcg_at_10": "rerank realized gain nDCG@10",
    "rerank_gap_closure_rate": "rerank gap closure rate",
    "bootstrap_only_rate": "bootstrap-only qrels rate",
    "needs_adjudication_rate": "needs adjudication rate",
    "vote_conflict_rate": "vote conflict rate",
    "llm_coverage_rate": "LLM qrels coverage rate",
    "llm_seconds_per_judgement": "LLM seconds per judgement",
    "style_violation_gap": "style violation gap",
    "all_fuzzy_ndcg_at_10": "all fuzzy nDCG@10",
    "natural_fuzzy_ndcg_at_10": "natural fuzzy nDCG@10",
    "all_fuzzy_style_violation_at_3": "all fuzzy style violation@3",
    "natural_fuzzy_style_violation_at_3": "natural fuzzy style violation@3",
}

DIAGNOSTIC_DIRECTIONS = {
    "ndcg_headroom_at_10": "lower_is_better",
    "rerank_oracle_gap_ndcg_at_10": "opportunity",
    "rerank_realized_gain_ndcg_at_10": "higher_is_better",
    "rerank_gap_closure_rate": "higher_is_better",
    "bootstrap_only_rate": "lower_is_better",
    "needs_adjudication_rate": "lower_is_better",
    "vote_conflict_rate": "lower_is_better",
    "llm_coverage_rate": "higher_is_better",
    "llm_seconds_per_judgement": "lower_is_better",
    "style_violation_gap": "lower_is_better",
    "all_fuzzy_ndcg_at_10": "higher_is_better",
    "natural_fuzzy_ndcg_at_10": "higher_is_better",
    "all_fuzzy_style_violation_at_3": "lower_is_better",
    "natural_fuzzy_style_violation_at_3": "lower_is_better",
}

DIAGNOSTIC_DIRECTION_DETAILS = {
    "higher_is_better": "Higher values indicate stronger observed capability or coverage.",
    "lower_is_better": "Lower values indicate less debt, risk, latency, or remaining headroom.",
    "opportunity": "Higher values indicate useful room to investigate; this is headroom, not automatically a regression.",
}

DIAGNOSTIC_METRIC_DETAILS = {
    "ndcg_headroom_at_10": {
        "use": "Shows how much ranking-quality room remains before nDCG@10 reaches 1.0.",
        "meaning": "A falling line means the quality gap is closing; a rising line means the latest judged ranking is weaker or more uncertain.",
    },
    "rerank_oracle_gap_ndcg_at_10": {
        "use": "Estimates the maximum nDCG@10 lift still available if top candidates were perfectly reordered by qrels.",
        "meaning": "Large values justify rerank feature work; tiny values point back to recall, query understanding, or qrels coverage.",
    },
    "rerank_realized_gain_ndcg_at_10": {
        "use": "Tracks how much of the baseline-to-rerank improvement the real or calibrated reranker actually captured.",
        "meaning": "Higher is better; negative values mean the attempted rerank hurt the measured ranking.",
    },
    "rerank_gap_closure_rate": {
        "use": "Compares realized rerank gain against the oracle rerank gap.",
        "meaning": "Near 1.0 means the current reranker captured most available rerank headroom; near 0 means the gap is still unused.",
    },
    "bootstrap_only_rate": {
        "use": "Measures how much of the qrels pool still depends only on bootstrap labels.",
        "meaning": "Lower is better because more labels have manual or LLM adjudication evidence.",
    },
    "needs_adjudication_rate": {
        "use": "Tracks the share of qrels that still need judge attention.",
        "meaning": "Lower is better; high values warn that evaluation conclusions are label-limited.",
    },
    "vote_conflict_rate": {
        "use": "Tracks disagreement between qrels votes or sources.",
        "meaning": "Lower is better; conflict means the benchmark may be ambiguous or under-specified.",
    },
    "llm_coverage_rate": {
        "use": "Shows the share of qrels covered by LLM adjudication.",
        "meaning": "Higher is better for benchmark trust, while cost and latency must stay controlled.",
    },
    "llm_seconds_per_judgement": {
        "use": "Monitors LLM judge throughput after batching, concurrency, cache, and timeout changes.",
        "meaning": "Lower is better; this is an operations metric and is hidden by default to keep the main quality view clean.",
    },
    "style_violation_gap": {
        "use": "Shows only the part of style_violation@3 above the 0.05 safety target.",
        "meaning": "Zero means the style-safety target is currently met.",
    },
    "all_fuzzy_ndcg_at_10": {
        "use": "Tracks graded ranking quality on the metadata-assisted fuzzy set.",
        "meaning": "Higher is better, but this set can be optimistic because internal labels may help the query.",
    },
    "natural_fuzzy_ndcg_at_10": {
        "use": "Tracks graded ranking quality on natural-language fuzzy queries.",
        "meaning": "Higher is better and more representative of real user phrasing than metadata-assisted fuzzy queries.",
    },
    "all_fuzzy_style_violation_at_3": {
        "use": "Tracks top-3 style safety on the metadata-assisted fuzzy set.",
        "meaning": "Lower is better; zero means no measured top-3 style violations.",
    },
    "natural_fuzzy_style_violation_at_3": {
        "use": "Tracks top-3 style safety on natural-language fuzzy queries.",
        "meaning": "Lower is better and should stay low even when natural phrasing is harder to parse.",
    },
}

DIAGNOSTIC_DEFAULT_HIDDEN_METRICS = {"llm_seconds_per_judgement"}

QRELS_TRUST_NUMERIC = {"none": 0.0, "low": 0.25, "medium": 0.6, "high": 0.9}


def record_capability_cycle_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    reports, missing_reports = load_capability_reports(list(getattr(args, "reports", None) or default_capability_report_paths()))
    registry = Path(getattr(args, "registry", DEFAULT_CAPABILITY_REGISTRY_PATH))
    previous_cycles = load_capability_cycles(registry)
    as_origin = bool(getattr(args, "as_origin", False))
    previous_cycle = None if as_origin or not previous_cycles else previous_cycles[-1]
    raw_metrics = extract_capability_raw_metrics(reports, missing_reports=missing_reports)
    capabilities = compute_capability_scores(raw_metrics)
    delta = capability_delta(capabilities, previous_cycle.get("capabilities", {}) if previous_cycle else None)
    recommendations = capability_recommendations(raw_metrics)
    cycle_id = getattr(args, "cycle_id", "") or f"origin_{time.strftime('%Y%m%d_%H%M%S')}"
    summary = capability_cycle_summary(
        cycle_id=cycle_id,
        label=getattr(args, "label", ""),
        is_origin=as_origin,
        capabilities=capabilities,
        raw_metrics=raw_metrics,
        delta=delta,
        previous_cycle=previous_cycle,
        recommendations=recommendations,
    )
    cycle = {
        "cycle_id": cycle_id,
        "label": getattr(args, "label", ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "is_origin": as_origin,
        "input_reports": [
            {"path": row["path"], "method": row.get("method", ""), "exists": True, "elapsed_seconds": row.get("elapsed_seconds")}
            for row in reports
        ]
        + [{"path": str(path), "exists": False} for path in missing_reports],
        "raw_metrics": raw_metrics,
        "capabilities": capabilities,
        "delta_vs_previous": delta,
        "summary": summary,
        "recommendations": recommendations,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    append_capability_cycle(registry, cycle)
    output = Path(getattr(args, "output", DEFAULT_CAPABILITY_CYCLE_PATH))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cycle, ensure_ascii=False, indent=2), encoding="utf-8")
    return cycle


def generate_capability_report_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    registry = Path(getattr(args, "registry", DEFAULT_CAPABILITY_REGISTRY_PATH))
    output = Path(getattr(args, "output", DEFAULT_CAPABILITY_REPORT_PATH))
    chart_dir = Path(getattr(args, "chart_dir", DEFAULT_CAPABILITY_CHART_DIR))
    cycles = load_capability_cycles(registry)
    chart_dir.mkdir(parents=True, exist_ok=True)
    charts = generate_capability_charts(cycles, chart_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    trend_html = output.parent / DEFAULT_CORE_METRICS_TREND_HTML.name
    diagnostic_html = output.parent / DEFAULT_DIAGNOSTIC_METRICS_TREND_HTML.name
    trend_html.write_text(core_metrics_trend_html(cycles), encoding="utf-8")
    diagnostic_html.write_text(diagnostic_metrics_trend_html(cycles), encoding="utf-8")
    output.write_text(capability_report_markdown(cycles, charts, output_path=output), encoding="utf-8")
    latest = cycles[-1] if cycles else {}
    return {
        "method": "retrieval_lab_capability_report",
        "registry": str(registry),
        "output": str(output),
        "charts": {key: str(value) for key, value in charts.items()},
        "html": str(trend_html),
        "diagnostic_html": str(diagnostic_html),
        "summary": {
            "cycle_count": len(cycles),
            "latest_cycle_id": latest.get("cycle_id"),
            "latest_overall_score": latest.get("summary", {}).get("overall_score"),
            "output": str(output),
            "html": str(trend_html),
            "diagnostic_html": str(diagnostic_html),
            "chart_count": len(charts),
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        },
    }


def default_capability_report_paths() -> list[Path]:
    return [
        Path(".tmp") / "qrels_audit_next.json",
        Path(".tmp") / "fuzzy_multi_next_fixed.json",
        Path(".tmp") / "rerank_upper_bound_next.json",
        Path(".tmp") / "pooled_qrels_next_report.json",
    ]


def load_capability_reports(paths: list[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    reports = []
    missing = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            missing.append(path)
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        reports.append(
            {
                "path": str(path),
                "method": data.get("method", ""),
                "summary": data.get("summary", {}),
                "graded_metrics": data.get("graded_metrics", {}),
                "scenarios": data.get("scenarios", {}),
                "workflows": data.get("workflows", {}),
                "metrics": data.get("metrics", {}),
                "elapsed_seconds": data.get("elapsed_seconds"),
            }
        )
    return reports, missing


def extract_capability_raw_metrics(reports: list[dict[str, Any]], *, missing_reports: list[Path]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "report_count": len(reports),
        "missing_report_count": len(missing_reports),
        "missing_reports": [str(path) for path in missing_reports],
    }
    sources: dict[str, dict[str, Any]] = {}
    priorities: dict[str, int] = {}
    elapsed_seconds = 0.0
    llm_elapsed_seconds = 0.0
    llm_judgement_count = 0.0
    for report in reports:
        method = str(report.get("method", ""))
        elapsed_seconds += float(report.get("elapsed_seconds") or 0.0)
        summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        if "llm" in method and "judgement_count" in summary:
            llm_elapsed_seconds += float(summary.get("elapsed_seconds") or report.get("elapsed_seconds") or 0.0)
            llm_judgement_count += float(summary.get("judgement_count") or 0.0)
        for container in (report.get("summary", {}), report.get("graded_metrics", {})):
            if not isinstance(container, dict):
                continue
            for raw_key, metric_key in CAPABILITY_METRIC_ALIASES.items():
                if raw_key in container:
                    priority = capability_metric_priority(method, metric_key)
                    if metric_key not in priorities or priority > priorities[metric_key]:
                        values[metric_key] = normalize_metric(container[raw_key])
                        priorities[metric_key] = priority
                        sources[metric_key] = {"method": method, "path": report["path"], "raw_key": raw_key}
        extract_scenario_diagnostic_metrics(report, values, sources=sources, priorities=priorities)
    values["elapsed_seconds_total"] = round(elapsed_seconds, 6)
    if llm_judgement_count > 0:
        values["llm_adjudication_elapsed_seconds"] = round(llm_elapsed_seconds, 6)
        values["llm_judgement_count"] = round(llm_judgement_count, 6)
    if "scene_level_recall_at_10" not in values and "target_recall_at_10" in values:
        values["scene_level_recall_at_10"] = values["target_recall_at_10"]
        sources["scene_level_recall_at_10"] = {**sources.get("target_recall_at_10", {}), "fallback": "target_recall_at_10"}
    add_derived_diagnostic_metrics(values, sources=sources)
    values["metric_sources"] = sources
    return values


def set_metric_if_better(
    values: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    priorities: dict[str, int],
    metric_key: str,
    value: Any,
    *,
    method: str,
    path: str,
    raw_key: str,
    priority: int | None = None,
) -> None:
    normalized = normalize_metric(value)
    if normalized is None or normalized == "":
        return
    selected_priority = capability_metric_priority(method, metric_key) if priority is None else priority
    if metric_key not in priorities or selected_priority > priorities[metric_key]:
        values[metric_key] = normalized
        priorities[metric_key] = selected_priority
        sources[metric_key] = {"method": method, "path": path, "raw_key": raw_key}


def extract_scenario_diagnostic_metrics(
    report: dict[str, Any],
    values: dict[str, Any],
    *,
    sources: dict[str, dict[str, Any]],
    priorities: dict[str, int],
) -> None:
    scenarios = report.get("scenarios", {}) if isinstance(report.get("scenarios"), dict) else {}
    if not scenarios:
        return
    scenario_map = {
        "metadata_assisted_style_safe": ("all_fuzzy_ndcg_at_10", "all_fuzzy_style_violation_at_3"),
        "metadata_assisted_hybrid": ("all_fuzzy_ndcg_at_10", "all_fuzzy_style_violation_at_3"),
        "natural_style_safe": ("natural_fuzzy_ndcg_at_10", "natural_fuzzy_style_violation_at_3"),
        "natural_hybrid": ("natural_fuzzy_ndcg_at_10", "natural_fuzzy_style_violation_at_3"),
    }
    for scenario_name, (ndcg_key, style_key) in scenario_map.items():
        payload = scenarios.get(scenario_name, {})
        summary = payload.get("summary", {}) if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
        if "nDCG@10" in summary:
            set_metric_if_better(
                values,
                sources,
                priorities,
                ndcg_key,
                summary.get("nDCG@10"),
                method="scenario",
                path=str(report.get("path", "")),
                raw_key=f"scenarios.{scenario_name}.summary.nDCG@10",
            )
        if "style_violation_at_3" in summary:
            set_metric_if_better(
                values,
                sources,
                priorities,
                style_key,
                summary.get("style_violation_at_3"),
                method="scenario",
                path=str(report.get("path", "")),
                raw_key=f"scenarios.{scenario_name}.summary.style_violation_at_3",
            )


def add_derived_diagnostic_metrics(values: dict[str, Any], *, sources: dict[str, dict[str, Any]]) -> None:
    derived_sources = sources
    ndcg = optional_float(values.get("ndcg_at_10"))
    if ndcg is not None:
        values["ndcg_headroom_at_10"] = round(max(0.0, 1.0 - ndcg), 6)
        derived_sources["ndcg_headroom_at_10"] = {"method": "derived", "path": "", "raw_key": "1 - ndcg_at_10"}
    qrels_count = optional_float(values.get("qrels_count")) or 0.0
    if qrels_count > 0:
        for numerator_key, output_key in (
            ("bootstrap_only_count", "bootstrap_only_rate"),
            ("needs_adjudication_count", "needs_adjudication_rate"),
            ("llm_count", "llm_coverage_rate"),
        ):
            numerator = optional_float(values.get(numerator_key))
            if numerator is not None:
                values[output_key] = round(numerator / qrels_count, 6)
                derived_sources[output_key] = {"method": "derived", "path": "", "raw_key": f"{numerator_key} / qrels_count"}
    if "rerank_oracle_gap_ndcg_at_10" not in values:
        oracle = optional_float(values.get("oracle_rerank_ndcg_at_10"))
        baseline = optional_float(values.get("baseline_ndcg_at_10"))
        if oracle is not None and baseline is not None:
            values["rerank_oracle_gap_ndcg_at_10"] = round(oracle - baseline, 6)
            derived_sources["rerank_oracle_gap_ndcg_at_10"] = {"method": "derived", "path": "", "raw_key": "oracle_rerank_ndcg_at_10 - baseline_ndcg_at_10"}
    if "rerank_realized_gain_ndcg_at_10" not in values:
        baseline = optional_float(values.get("baseline_ndcg_at_10"))
        best_candidates = [
            optional_float(values.get("calibrated_ndcg_at_10")),
            optional_float(values.get("rule_rerank_ndcg_at_10")),
            optional_float(values.get("llm_rerank_ndcg_at_10")),
        ]
        best = max([value for value in best_candidates if value is not None], default=None)
        if baseline is not None and best is not None:
            values["rerank_realized_gain_ndcg_at_10"] = round(best - baseline, 6)
            derived_sources["rerank_realized_gain_ndcg_at_10"] = {"method": "derived", "path": "", "raw_key": "best_rerank_ndcg_at_10 - baseline_ndcg_at_10"}
    if "rerank_gap_closure_rate" not in values:
        oracle_gap = optional_float(values.get("rerank_oracle_gap_ndcg_at_10"))
        realized = optional_float(values.get("rerank_realized_gain_ndcg_at_10"))
        if oracle_gap is not None and oracle_gap > 0 and realized is not None:
            values["rerank_gap_closure_rate"] = round(realized / oracle_gap, 6)
            derived_sources["rerank_gap_closure_rate"] = {"method": "derived", "path": "", "raw_key": "rerank_realized_gain_ndcg_at_10 / rerank_oracle_gap_ndcg_at_10"}
    style_violation = optional_float(values.get("style_violation_at_3"))
    if style_violation is not None:
        values["style_violation_gap"] = round(max(0.0, style_violation - 0.05), 6)
        derived_sources["style_violation_gap"] = {"method": "derived", "path": "", "raw_key": "max(0, style_violation_at_3 - 0.05)"}
    llm_elapsed = optional_float(values.get("llm_adjudication_elapsed_seconds"))
    llm_judgements = optional_float(values.get("llm_judgement_count"))
    if llm_elapsed is not None and llm_judgements and llm_judgements > 0:
        values["llm_seconds_per_judgement"] = round(llm_elapsed / llm_judgements, 6)
        derived_sources["llm_seconds_per_judgement"] = {"method": "derived", "path": "", "raw_key": "llm_adjudication_elapsed_seconds / llm_judgement_count"}



def capability_metric_priority(method: str, metric_key: str) -> int:
    if metric_key in DIAGNOSTIC_TREND_METRICS:
        if method in {"derived", "scenario"}:
            return 60
        if method.startswith("retrieval_lab_"):
            return 45
    if metric_key.startswith("qrels") or metric_key in {
        "manual_or_llm_count",
        "manual_count",
        "llm_count",
        "bootstrap_only_count",
        "needs_adjudication_count",
        "vote_conflict_rate",
    }:
        return 40 if method == "mock_qrels_audit" else 20
    if metric_key.startswith("rerank") or metric_key in {"oracle_rerank_ndcg_at_10", "baseline_ndcg_at_10"}:
        return 40 if method == "mock_rerank_upper_bound_comparison" else 20
    if metric_key in {
        "ndcg_at_10",
        "mrr_at_10",
        "recall_at_10",
        "scene_level_recall_at_10",
        "target_recall_at_10",
        "stage_level_hit_at_3",
        "purpose_level_hit_at_3",
        "style_violation_at_3",
        "failure_rate",
    }:
        return 40 if method == "mock_fuzzy_multirelevance_evaluation" else 20
    return 10


def normalize_metric(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def compute_capability_scores(raw_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "retrieval_quality": capability_row(
            weighted_score_optional(
                [
                    (scale_metric_optional(raw_metrics.get("ndcg_at_10"), target=0.75), 0.45),
                    (scale_metric_optional(raw_metrics.get("mrr_at_10"), target=0.9), 0.25),
                    (scale_metric_optional(raw_metrics.get("scene_level_recall_at_10"), target=0.7), 0.30),
                ]
            ),
            "capability",
            {
                "ndcg_at_10": raw_metrics.get("ndcg_at_10"),
                "mrr_at_10": raw_metrics.get("mrr_at_10"),
                "scene_level_recall_at_10": raw_metrics.get("scene_level_recall_at_10"),
            },
        ),
        "fuzzy_understanding": capability_row(
            weighted_score_optional(
                [
                    (scale_metric_optional(raw_metrics.get("stage_level_hit_at_3"), target=0.95), 0.35),
                    (scale_metric_optional(raw_metrics.get("purpose_level_hit_at_3"), target=0.9), 0.35),
                    (scale_metric_optional(raw_metrics.get("scene_level_recall_at_10"), target=0.7), 0.30),
                ]
            ),
            "capability",
            {
                "stage_level_hit_at_3": raw_metrics.get("stage_level_hit_at_3"),
                "purpose_level_hit_at_3": raw_metrics.get("purpose_level_hit_at_3"),
                "scene_level_recall_at_10": raw_metrics.get("scene_level_recall_at_10"),
            },
        ),
        "style_safety": capability_row(
            style_safety_score_optional(raw_metrics.get("style_violation_at_3")),
            "capability",
            {"style_violation_at_3": raw_metrics.get("style_violation_at_3")},
        ),
        "qrels_trust": capability_row(
            qrels_trust_score_optional(raw_metrics),
            "capability",
            {
                "qrels_trust_level": raw_metrics.get("qrels_trust_level"),
                "manual_or_llm_count": raw_metrics.get("manual_or_llm_count"),
                "bootstrap_only_count": raw_metrics.get("bootstrap_only_count"),
                "needs_adjudication_count": raw_metrics.get("needs_adjudication_count"),
                "vote_conflict_rate": raw_metrics.get("vote_conflict_rate"),
            },
        ),
        "cycle_operability": capability_row(
            cycle_operability_score(raw_metrics),
            "capability",
            {
                "report_count": raw_metrics.get("report_count"),
                "missing_report_count": raw_metrics.get("missing_report_count"),
                "active_sample_count": raw_metrics.get("active_sample_count"),
                "llm_call_count": raw_metrics.get("llm_call_count"),
                "elapsed_seconds_total": raw_metrics.get("elapsed_seconds_total"),
            },
        ),
        "rerank_potential": capability_row(
            scale_metric_optional(raw_metrics.get("rerank_opportunity_ndcg_at_10"), target=0.30),
            "opportunity",
            {"rerank_opportunity_ndcg_at_10": raw_metrics.get("rerank_opportunity_ndcg_at_10")},
        ),
    }


def capability_row(score: float, kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if score is None:
        return {"score": None, "kind": kind, "inputs": inputs, "available": False}
    return {"score": round(clamp(score, 0.0, 100.0), 3), "kind": kind, "inputs": inputs, "available": True}


def capability_delta(
    capabilities: dict[str, dict[str, Any]],
    previous_capabilities: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, row in capabilities.items():
        current = optional_float(row.get("score"))
        previous = None
        if previous_capabilities and name in previous_capabilities:
            previous = optional_float(previous_capabilities[name].get("score"))
        result[name] = {
            "current_score": round(current, 3) if current is not None else None,
            "previous_score": round(previous, 3) if previous is not None else None,
            "score_delta": round(current - previous, 3) if current is not None and previous is not None else None,
        }
    return result


def capability_cycle_summary(
    *,
    cycle_id: str,
    label: str,
    is_origin: bool,
    capabilities: dict[str, dict[str, Any]],
    raw_metrics: dict[str, Any],
    delta: dict[str, dict[str, Any]],
    previous_cycle: dict[str, Any] | None,
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = [float(row["score"]) for row in capabilities.values() if row.get("kind") == "capability" and row.get("score") is not None]
    improved = [name for name, row in delta.items() if optional_float(row.get("score_delta")) is not None and float(row.get("score_delta")) > 0.01]
    regressed = [name for name, row in delta.items() if optional_float(row.get("score_delta")) is not None and float(row.get("score_delta")) < -0.01]
    return {
        "cycle_id": cycle_id,
        "label": label,
        "is_origin": is_origin,
        "previous_cycle_id": previous_cycle.get("cycle_id") if previous_cycle else None,
        "overall_score": round(mean(scores), 3),
        "qrels_trust_level": raw_metrics.get("qrels_trust_level", "low"),
        "rerank_opportunity_nDCG@10": raw_metrics.get("rerank_opportunity_ndcg_at_10"),
        "report_count": raw_metrics.get("report_count", 0),
        "missing_report_count": raw_metrics.get("missing_report_count", 0),
        "improved_capabilities": improved,
        "regressed_capabilities": regressed,
        "top_recommendation": recommendations[0]["title"] if recommendations else "",
    }


def capability_recommendations(raw_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations = []
    if str(raw_metrics.get("qrels_trust_level", "low")) == "low":
        recommendations.append(
            {
                "priority": 1,
                "title": "Improve qrels trust",
                "reason": "qrels_trust_level is low, so capability deltas are still bootstrap-guided.",
                "command": "python -m retrieval_lab qrels sample-active --split test --limit 60 --sample-size 80 --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\active_qrels_next.jsonl",
            }
        )
    if (optional_float(raw_metrics.get("rerank_opportunity_ndcg_at_10")) or 0.0) >= 0.15:
        recommendations.append(
            {
                "priority": 2,
                "title": "Run real reranker sample",
                "reason": "oracle rerank has a large nDCG@10 opportunity.",
                "command": "python -m retrieval_lab eval rerank-upper-bound --split test --limit 60 --qrels .tmp\\pooled_qrels_next.jsonl --llm-rerank-sample-size 10",
            }
        )
    if (optional_float(raw_metrics.get("style_violation_at_3")) or 0.0) > 0.05:
        recommendations.append(
            {
                "priority": 3,
                "title": "Tighten style negative handling",
                "reason": "style_violation_at_3 is above the first-stage target.",
                "command": "python -m retrieval_lab eval style-risk --split test --limit 60",
            }
        )
    return sorted(recommendations, key=lambda row: int(row["priority"])) or [
        {
            "priority": 9,
            "title": "Continue the measured flywheel",
            "reason": "No critical bottleneck was detected.",
            "command": "python -m retrieval_lab flywheel guide",
        }
    ]


def load_capability_cycles(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def append_capability_cycle(path: Path, cycle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(cycle, ensure_ascii=False, sort_keys=True) + "\n")


def generate_capability_charts(cycles: list[dict[str, Any]], chart_dir: Path) -> dict[str, Path]:
    if not cycles:
        return {}
    latest = cycles[-1]
    charts = {
        "capability_bar": chart_dir / "capability_bar_latest.svg",
        "delta_bar": chart_dir / "capability_delta_latest.svg",
        "trend_line": chart_dir / "capability_trend.svg",
        "qrels_trust": chart_dir / "qrels_trust_stack.svg",
    }
    charts["capability_bar"].write_text(capability_bar_svg(latest), encoding="utf-8")
    charts["delta_bar"].write_text(capability_delta_svg(latest), encoding="utf-8")
    charts["trend_line"].write_text(capability_trend_svg(cycles), encoding="utf-8")
    charts["qrels_trust"].write_text(qrels_trust_stack_svg(latest), encoding="utf-8")
    return charts


def capability_report_markdown(cycles: list[dict[str, Any]], charts: dict[str, Path], *, output_path: Path) -> str:
    if not cycles:
        return "# Retrieval Capability Report\n\nNo capability cycles recorded yet.\n"
    latest = cycles[-1]
    previous = cycles[-2] if len(cycles) >= 2 else None
    origin = next((cycle for cycle in cycles if cycle.get("is_origin")), cycles[0])
    lines = [
        "# Retrieval Capability Report",
        "",
        "## Cycle Summary",
        "",
        f"- latest_cycle: `{latest.get('cycle_id')}`",
        f"- previous_cycle: `{previous.get('cycle_id') if previous else ''}`",
        f"- origin_cycle: `{origin.get('cycle_id')}`",
        f"- git_sha: `{latest.get('git_sha', '')}`",
        f"- input_reports: `{latest.get('summary', {}).get('report_count', 0)}`",
        f"- overall_score: `{latest.get('summary', {}).get('overall_score', 0.0)}`",
        "",
        "## Charts",
        "",
    ]
    for key, path in charts.items():
        lines.extend([f"![{key}]({markdown_chart_path(path, output_path)})", ""])
    lines.extend(["## Capability Scoreboard", "", "| capability | score | delta | kind |", "|---|---:|---:|---|"])
    for name, row in latest.get("capabilities", {}).items():
        delta = latest.get("delta_vs_previous", {}).get(name, {}).get("score_delta", 0.0)
        score = row.get("score")
        lines.append(f"| {name} | {format_trend_value(score)} | {signed_delta(delta)} | {row.get('kind', '')} |")
    trend_path = output_path.parent / DEFAULT_CORE_METRICS_TREND_HTML.name
    diagnostic_path = output_path.parent / DEFAULT_DIAGNOSTIC_METRICS_TREND_HTML.name
    lines.extend([
        "",
        "## Diagnostic Metric Trends",
        "",
        f"- HTML dashboard: [{markdown_chart_path(diagnostic_path, output_path)}]({markdown_chart_path(diagnostic_path, output_path)})",
        "",
        "| metric | latest | previous | delta | direction | source |",
        "|---|---:|---:|---:|---|---|",
    ])
    for metric in DIAGNOSTIC_TREND_METRICS:
        latest_value = diagnostic_metric_value(latest, metric)
        previous_value = diagnostic_metric_value(previous or {}, metric)
        delta_value = trend_delta(latest_value, previous_value)
        lines.append(
            f"| {DIAGNOSTIC_TREND_LABELS[metric]} | {format_trend_value(latest_value)} | {format_trend_value(previous_value)} | {format_trend_delta(delta_value)} | {DIAGNOSTIC_DIRECTIONS[metric]} | {metric_source(latest, metric)} |"
        )
    lines.extend([
        "",
        "## Core Metric Trends",
        "",
        f"- HTML dashboard: [{markdown_chart_path(trend_path, output_path)}]({markdown_chart_path(trend_path, output_path)})",
        "",
        "| metric | latest | previous | delta |",
        "|---|---:|---:|---:|",
    ])
    for metric in CORE_TREND_METRICS:
        latest_value = core_metric_value(latest, metric)
        previous_value = core_metric_value(previous or {}, metric)
        delta_value = trend_delta(latest_value, previous_value)
        lines.append(
            f"| {CORE_TREND_LABELS[metric]} | {format_trend_value(latest_value)} | {format_trend_value(previous_value)} | {format_trend_delta(delta_value)} |"
        )
    lines.extend(["", "## Bottleneck Analysis", ""])
    for recommendation in latest.get("recommendations", []):
        lines.append(f"- **{recommendation['title']}**: {recommendation['reason']}")
    lines.extend(["", "## Next Cycle Recommendation", ""])
    for recommendation in latest.get("recommendations", []):
        lines.append(f"- `{recommendation['command']}`")
    lines.append("")
    return "\n".join(lines)


def capability_bar_svg(cycle: dict[str, Any]) -> str:
    rows = [
        (name, optional_float(row.get("score")) or 0.0, row.get("kind", "capability"))
        for name, row in cycle.get("capabilities", {}).items()
        if optional_float(row.get("score")) is not None
    ]
    return bar_svg(rows, title="Capability Scores")


def capability_delta_svg(cycle: dict[str, Any]) -> str:
    rows = [
        (name, optional_float(row.get("score_delta")) or 0.0, "delta")
        for name, row in cycle.get("delta_vs_previous", {}).items()
        if optional_float(row.get("score_delta")) is not None
    ]
    return bar_svg(rows, title="Delta vs Previous Cycle", diverging=True)


def capability_trend_svg(cycles: list[dict[str, Any]]) -> str:
    series = {
        "retrieval_quality": [optional_float(cycle.get("capabilities", {}).get("retrieval_quality", {}).get("score")) for cycle in cycles],
        "fuzzy_understanding": [optional_float(cycle.get("capabilities", {}).get("fuzzy_understanding", {}).get("score")) for cycle in cycles],
        "qrels_trust": [optional_float(cycle.get("capabilities", {}).get("qrels_trust", {}).get("score")) for cycle in cycles],
    }
    return line_svg(series, title="Capability Trend")


def qrels_trust_stack_svg(cycle: dict[str, Any]) -> str:
    raw = cycle.get("raw_metrics", {})
    qrels_count = optional_float(raw.get("qrels_count")) or 0.0
    manual = optional_float(raw.get("manual_count")) or 0.0
    llm = optional_float(raw.get("llm_count")) or 0.0
    bootstrap = optional_float(raw.get("bootstrap_only_count")) or max(0.0, qrels_count - manual - llm)
    needs = optional_float(raw.get("needs_adjudication_count")) or 0.0
    return stacked_bar_svg(
        [
            ("bootstrap", bootstrap, "#9ca3af"),
            ("manual", manual, "#16a34a"),
            ("llm", llm, "#2563eb"),
            ("needs_adjudication", needs, "#f59e0b"),
        ],
        total=max(qrels_count, bootstrap + manual + llm + needs, 1.0),
        title="Qrels Trust Composition",
    )


def core_metrics_trend_html(cycles: list[dict[str, Any]]) -> str:
    labels = [str(cycle.get("cycle_id", f"cycle_{index + 1}")) for index, cycle in enumerate(cycles)]
    series = {metric: [core_metric_numeric(cycle, metric) for cycle in cycles] for metric in CORE_TREND_METRICS}
    latest = cycles[-1] if cycles else {}
    rows = []
    for metric in CORE_TREND_METRICS:
        current = core_metric_value(latest, metric)
        rows.append(
            "<tr>"
            f"<td>{html_escape(CORE_TREND_LABELS[metric])}</td>"
            f"<td>{format_trend_value(current)}</td>"
            f"<td>{html_escape(metric_source(latest, metric))}</td>"
            "</tr>"
        )
    svg = core_metrics_line_svg(labels, series)
    data_json = json.dumps({"labels": labels, "series": series}, ensure_ascii=False)
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "<title>Retrieval Core Metrics Trend</title>",
            "<style>",
            "body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}",
            ".wrap{max-width:1180px;margin:0 auto}.panel{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:18px;margin-bottom:18px}",
            "h1{font-size:22px;margin:0 0 8px}p{color:#4b5563;margin:0 0 12px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left}th{background:#f3f4f6}.small{font-size:12px;color:#6b7280}svg{width:100%;height:auto}",
            "</style>",
            "</head>",
            "<body><div class=\"wrap\">",
            "<div class=\"panel\"><h1>Retrieval Core Metrics Trend</h1><p>Longitudinal raw metrics from capability cycles. qrels_trust_level is mapped as none=0, low=0.25, medium=0.60, high=0.90 for plotting.</p></div>",
            f"<div class=\"panel\">{svg}</div>",
            "<div class=\"panel\"><h2>Latest Values</h2><table><thead><tr><th>Metric</th><th>Latest</th><th>Source</th></tr></thead><tbody>",
            *rows,
            "</tbody></table></div>",
            f"<script type=\"application/json\" id=\"core-metrics-data\">{script_json_escape(data_json)}</script>",
            "<div class=\"small\">Generated by retrieval_lab report capability.</div>",
            "</div></body></html>",
        ]
    )


def diagnostic_metrics_trend_html(cycles: list[dict[str, Any]]) -> str:
    labels = [str(cycle.get("cycle_id", f"cycle_{index + 1}")) for index, cycle in enumerate(cycles)]
    series = {metric: [diagnostic_metric_numeric(cycle, metric) for cycle in cycles] for metric in DIAGNOSTIC_TREND_METRICS}
    latest = cycles[-1] if cycles else {}
    data_json = json.dumps(
        {
            "labels": labels,
            "series": series,
            "directions": DIAGNOSTIC_DIRECTIONS,
            "labelsByMetric": DIAGNOSTIC_TREND_LABELS,
            "groups": diagnostic_metric_groups(),
        },
        ensure_ascii=False,
    )
    rows = []
    for metric in DIAGNOSTIC_TREND_METRICS:
        observation = latest_observed_diagnostic(cycles, metric)
        rows.append(
            f"<tr data-metric-row=\"{html_escape(metric)}\" data-direction=\"{html_escape(DIAGNOSTIC_DIRECTIONS.get(metric, ''))}\">"
            f"<td>{html_escape(DIAGNOSTIC_TREND_LABELS[metric])}</td>"
            f"<td>{format_trend_value(observation.get('value'))}</td>"
            f"<td>{sparkline_svg(series.get(metric, []), metric=metric)}</td>"
            f"<td class=\"source-cell\">{html_escape(diagnostic_source_explanation(cycles, metric, observation))}</td>"
            "</tr>"
        )
    controls = diagnostic_metric_controls()
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "<title>Retrieval Diagnostic Metrics Trend</title>",
            "<style>",
            "body{font-family:Inter,Segoe UI,Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}",
            ".wrap{max-width:1240px;margin:0 auto}.panel{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:18px;margin-bottom:18px}",
            "h1{font-size:22px;margin:0 0 8px}h2{font-size:16px;margin:0 0 12px}h3{font-size:14px;margin:0 0 10px}p{color:#4b5563;margin:0 0 12px;line-height:1.45}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}th{background:#f3f4f6}.small{font-size:12px;color:#6b7280}svg{width:100%;height:auto}.metric-controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.metric-group{border:1px solid #e5e7eb;border-radius:8px;padding:12px;background:#f9fafb}.metric-toggle{display:flex;align-items:flex-start;gap:8px;margin:8px 0;font-size:13px;color:#111827}.metric-toggle input{margin-top:2px}.metric-direction{display:block;font-size:11px;color:#6b7280}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.toolbar button{border:1px solid #d1d5db;background:#fff;border-radius:6px;padding:6px 10px;font-size:12px;cursor:pointer}.toolbar button:hover{background:#f3f4f6}.chart-grid{display:grid;grid-template-columns:1fr;gap:16px}.chart-panel{border:1px solid #e5e7eb;border-radius:8px;padding:12px;background:#fff}.chart-panel[data-direction=higher_is_better]{border-left:4px solid #16a34a}.chart-panel[data-direction=lower_is_better]{border-left:4px solid #dc2626}.chart-panel[data-direction=opportunity]{border-left:4px solid #f59e0b}.chart-note{font-size:12px;color:#6b7280;margin:0 0 8px}.sparkline{width:126px;height:34px}.sparkline text{font-size:8px}.source-cell{max-width:560px}tr[data-metric-row].is-hidden{display:none}.metric-hit{cursor:default;opacity:0}.metric-hit:hover{opacity:.08}.tooltip{position:fixed;z-index:10;display:none;max-width:360px;padding:8px 10px;border:1px solid #d1d5db;border-radius:6px;background:#111827;color:#fff;font-size:12px;line-height:1.35;box-shadow:0 8px 20px rgba(15,23,42,.18);pointer-events:none}",
            ".chart-switcher{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.chart-switcher button{border:1px solid #d1d5db;background:#fff;border-radius:6px;padding:7px 12px;font-size:13px;cursor:pointer}.chart-switcher button:hover{background:#f3f4f6}.chart-switcher button.is-active{background:#111827;border-color:#111827;color:#fff}.chart-panel{display:none}.chart-panel.is-active{display:block}",
            ".panel-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.panel-heading h2{margin:0}.control-toggle{border:1px solid #d1d5db;background:#fff;border-radius:6px;padding:7px 12px;font-size:12px;cursor:pointer;white-space:nowrap}.control-toggle:hover{background:#f3f4f6}.metric-controls-body.is-hidden{display:none}",
            "</style>",
            "</head>",
            "<body><div class=\"wrap\">",
            "<div class=\"panel\"><h1>Diagnostic Metrics Trend</h1><p>These are bottleneck and headroom metrics. Metrics are split by direction so high-is-good and low-is-good signals are not visually mixed. Use the checkboxes to toggle individual lines. Missing cycle values are left blank rather than plotted as zero.</p></div>",
            f"<div class=\"panel metric-visibility-panel\"><div class=\"panel-heading\"><h2>Metric Visibility</h2><button type=\"button\" class=\"control-toggle\" data-toggle-metric-controls aria-expanded=\"true\">Hide metric controls</button></div><div class=\"metric-controls-body\" id=\"metric-controls-body\"><div class=\"toolbar\"><button type=\"button\" data-action=\"all\">Show all</button><button type=\"button\" data-action=\"none\">Hide all</button><button type=\"button\" data-action=\"higher_is_better\">Show higher-is-better</button><button type=\"button\" data-action=\"lower_is_better\">Show lower-is-better</button><button type=\"button\" data-action=\"opportunity\">Show opportunity</button></div><div class=\"metric-controls\">{controls}</div></div></div>",
            "<div class=\"panel chart-grid\"><div class=\"chart-switcher\"><button type=\"button\" class=\"is-active\" data-chart-tab=\"lower_is_better\">Lower Is Better</button><button type=\"button\" data-chart-tab=\"higher_is_better\">Higher Is Better</button><button type=\"button\" data-chart-tab=\"opportunity\">Opportunity</button></div><div class=\"chart-panel\" data-direction=\"higher_is_better\"><h2>Higher Is Better</h2><p class=\"chart-note\">Quality, coverage, and realized gain metrics.</p><div id=\"chart-higher-is-better\"></div></div><div class=\"chart-panel is-active\" data-direction=\"lower_is_better\"><h2>Lower Is Better</h2><p class=\"chart-note\">Headroom, conflict, latency, and risk metrics.</p><div id=\"chart-lower-is-better\"></div></div><div class=\"chart-panel\" data-direction=\"opportunity\"><h2>Opportunity / Headroom</h2><p class=\"chart-note\">Large values mean useful room to investigate, not necessarily a regression.</p><div id=\"chart-opportunity\"></div></div></div>",
            "<div class=\"panel\"><h2>Latest Diagnostic Values</h2><table><thead><tr><th>Metric</th><th>Latest Observed</th><th>Trend</th><th>Source / Use / Meaning</th></tr></thead><tbody>",
            *rows,
            "</tbody></table></div>",
            "<div id=\"metric-tooltip\" class=\"tooltip\"></div>",
            f"<script type=\"application/json\" id=\"diagnostic-metrics-data\">{script_json_escape(data_json)}</script>",
            f"<script>{diagnostic_dashboard_script()}</script>",
            "<div class=\"small\">Generated by retrieval_lab report capability.</div>",
            "</div></body></html>",
        ]
    )


def diagnostic_metric_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"higher_is_better": [], "lower_is_better": [], "opportunity": []}
    for metric in DIAGNOSTIC_TREND_METRICS:
        groups.setdefault(DIAGNOSTIC_DIRECTIONS.get(metric, "opportunity"), []).append(metric)
    return groups


def diagnostic_metric_controls() -> str:
    group_titles = {
        "higher_is_better": "Higher Is Better",
        "lower_is_better": "Lower Is Better",
        "opportunity": "Opportunity / Headroom",
    }
    parts = []
    for direction, metrics in diagnostic_metric_groups().items():
        parts.append(f'<div class="metric-group" data-direction="{html_escape(direction)}"><h3>{html_escape(group_titles.get(direction, direction))}</h3>')
        for metric in metrics:
            checked = " checked" if metric not in DIAGNOSTIC_DEFAULT_HIDDEN_METRICS else ""
            parts.append(
                f'<label class="metric-toggle"><input type="checkbox" data-metric="{html_escape(metric)}" data-direction="{html_escape(direction)}"{checked}> '
                f'<span>{html_escape(DIAGNOSTIC_TREND_LABELS[metric])}<span class="metric-direction">{html_escape(metric)}</span></span></label>'
            )
        parts.append("</div>")
    return "".join(parts)


def diagnostic_dashboard_script() -> str:
    return r'''(function() {
  const raw = document.getElementById('diagnostic-metrics-data').textContent;
  const data = JSON.parse(raw);
  const colors = ['#2563eb','#16a34a','#dc2626','#7c3aed','#f59e0b','#0891b2','#be123c','#65a30d','#475569','#9333ea','#0f766e','#b45309','#4f46e5','#9f1239'];
  const targets = {
    higher_is_better: document.getElementById('chart-higher-is-better'),
    lower_is_better: document.getElementById('chart-lower-is-better'),
    opportunity: document.getElementById('chart-opportunity')
  };
  function esc(value) {
    return String(value).replace(/[&<>\"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[ch]));
  }
  function selectedMetrics(direction) {
    return Array.from(document.querySelectorAll('input[data-metric][data-direction="' + direction + '"]')).filter(input => input.checked).map(input => input.dataset.metric);
  }
  function formatValue(value) {
    if (value === null || value === undefined) return '';
    const numeric = Number(value);
    return Number.isFinite(numeric) ? String(Math.round(numeric * 1000000) / 1000000) : String(value);
  }
  function normalize(values) {
    const nums = values.filter(value => value !== null && value !== undefined).map(Number);
    if (!nums.length) return values.map(_ => null);
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    if (max === min) {
      const scale = Math.max(Math.abs(max), 1);
      return values.map(value => value === null || value === undefined ? null : Math.max(0, Math.min(1, Number(value) / scale)));
    }
    return values.map(value => value === null || value === undefined ? null : (Number(value) - min) / (max - min));
  }
  function renderChart(direction) {
    const metrics = selectedMetrics(direction);
    const width = 1120, height = 390, left = 72, top = 36, chartWidth = 780, chartHeight = 250;
    const maxPoints = Math.max(1, data.labels.length);
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(direction)} diagnostic chart"><rect width="100%" height="100%" fill="#fff"/><rect x="${left}" y="${top}" width="${chartWidth}" height="${chartHeight}" fill="#fff" stroke="#d1d5db"/>`; 
    for (let tick = 0; tick <= 5; tick += 1) {
      const value = tick / 5;
      const y = top + chartHeight - chartHeight * value;
      svg += `<line x1="${left}" y1="${y.toFixed(2)}" x2="${left + chartWidth}" y2="${y.toFixed(2)}" stroke="#e5e7eb"/><text x="24" y="${(y + 4).toFixed(2)}" font-size="11" fill="#6b7280">${value.toFixed(1)}</text>`;
    }
    data.labels.forEach((label, index) => {
      if (maxPoints === 1 || index % Math.max(1, Math.floor(maxPoints / 8)) === 0 || index === maxPoints - 1) {
        const x = left + chartWidth * index / Math.max(1, maxPoints - 1);
        svg += `<text x="${x.toFixed(2)}" y="${top + chartHeight + 24}" font-size="10" fill="#6b7280" transform="rotate(30 ${x.toFixed(2)} ${top + chartHeight + 24})">${esc(label)}</text>`;
      }
    });
    if (!metrics.length) {
      svg += `<text x="${left + 18}" y="${top + 48}" font-size="13" fill="#6b7280">No metrics selected.</text></svg>`;
      targets[direction].innerHTML = svg;
      return;
    }
    metrics.forEach((metric, seriesIndex) => {
      const values = normalize(data.series[metric] || []);
      const rawValues = data.series[metric] || [];
      const color = colors[seriesIndex % colors.length];
      const points = [];
      values.forEach((value, pointIndex) => {
        if (value === null || value === undefined) return;
        const x = left + chartWidth * pointIndex / Math.max(1, maxPoints - 1);
        const y = top + chartHeight - chartHeight * Math.max(0, Math.min(1, Number(value)));
        points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
      });
      if (points.length) svg += `<polyline points="${points.join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`; 
      values.forEach((value, pointIndex) => {
        if (value === null || value === undefined) return;
        const x = left + chartWidth * pointIndex / Math.max(1, maxPoints - 1);
        const y = top + chartHeight - chartHeight * Math.max(0, Math.min(1, Number(value)));
        const label = data.labels[pointIndex] || `cycle_${pointIndex + 1}`;
        const rawValue = formatValue(rawValues[pointIndex]);
        const tooltip = `${data.labelsByMetric[metric] || metric}\ncycle: ${label}\nvalue: ${rawValue}`;
        svg += `<circle class="metric-hit" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="8" fill="${color}" stroke="none" data-tooltip="${esc(tooltip)}"><title>${esc(tooltip)}</title></circle>`;
      });
      const legendX = 880;
      const legendY = 48 + seriesIndex * 32;
      svg += `<rect x="${legendX}" y="${legendY - 10}" width="12" height="12" fill="${color}"/><text x="${legendX + 18}" y="${legendY}" font-size="11" fill="#111827">${esc(data.labelsByMetric[metric] || metric)}</text><text x="${legendX + 18}" y="${legendY + 12}" font-size="9" fill="#6b7280">${esc(metric)}</text>`;
    });
    svg += '</svg>';
    targets[direction].innerHTML = svg;
  }
  function renderAll() {
    Object.keys(targets).forEach(renderChart);
    bindMetricTooltips();
  }
  function bindMetricTooltips() {
    const tooltip = document.getElementById('metric-tooltip');
    if (!tooltip) return;
    document.querySelectorAll('[data-tooltip]').forEach(node => {
      node.addEventListener('mouseenter', event => {
        tooltip.textContent = node.getAttribute('data-tooltip') || '';
        tooltip.style.display = 'block';
        tooltip.style.left = `${event.clientX + 12}px`;
        tooltip.style.top = `${event.clientY + 12}px`;
      });
      node.addEventListener('mousemove', event => {
        tooltip.style.left = `${event.clientX + 12}px`;
        tooltip.style.top = `${event.clientY + 12}px`;
      });
      node.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
      });
    });
  }
  function setActiveChart(direction) {
    document.querySelectorAll('.chart-panel[data-direction]').forEach(panel => {
      panel.classList.toggle('is-active', panel.dataset.direction === direction);
    });
    document.querySelectorAll('button[data-chart-tab]').forEach(button => {
      button.classList.toggle('is-active', button.dataset.chartTab === direction);
    });
    document.querySelectorAll('tr[data-metric-row]').forEach(row => {
      row.classList.toggle('is-hidden', row.dataset.direction !== direction);
    });
  }
  document.querySelectorAll('input[data-metric]').forEach(input => input.addEventListener('change', renderAll));
  document.querySelectorAll('button[data-chart-tab]').forEach(button => button.addEventListener('click', () => setActiveChart(button.dataset.chartTab)));
  document.querySelectorAll('button[data-toggle-metric-controls]').forEach(button => button.addEventListener('click', event => {
    event.stopPropagation();
    const body = document.getElementById('metric-controls-body');
    if (!body) return;
    const hidden = !body.classList.contains('is-hidden');
    body.classList.toggle('is-hidden', hidden);
    button.setAttribute('aria-expanded', hidden ? 'false' : 'true');
    button.textContent = hidden ? 'Show metric controls' : 'Hide metric controls';
  }));
  document.querySelectorAll('.metric-visibility-panel').forEach(panel => panel.addEventListener('click', event => {
    const body = document.getElementById('metric-controls-body');
    const button = panel.querySelector('button[data-toggle-metric-controls]');
    if (!body || !button || !body.classList.contains('is-hidden')) return;
    event.stopPropagation();
    body.classList.remove('is-hidden');
    button.setAttribute('aria-expanded', 'true');
    button.textContent = 'Hide metric controls';
  }));
  document.querySelectorAll('button[data-action]').forEach(button => button.addEventListener('click', () => {
    const action = button.dataset.action;
    document.querySelectorAll('input[data-metric]').forEach(input => {
      input.checked = action === 'all' ? true : action === 'none' ? false : input.dataset.direction === action;
    });
    if (targets[action]) setActiveChart(action);
    renderAll();
  }));
  setActiveChart('lower_is_better');
  renderAll();
})();''' 


def core_metrics_line_svg(labels: list[str], series: dict[str, list[float | None]]) -> str:
    width = 1100
    height = 650
    left = 74
    top = 52
    chart_width = 840
    chart_height = 430
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#65a30d", "#be123c", "#9333ea", "#475569"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Core metrics trend">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-size="20" font-weight="700" fill="#111827">Core Metrics Trend</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#fff" stroke="#d1d5db"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + chart_height - chart_height * value
        lines.append(f'<line x1="{left}" y1="{round(y,2)}" x2="{left + chart_width}" y2="{round(y,2)}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="24" y="{round(y + 4,2)}" font-size="11" fill="#6b7280">{round(value, 2)}</text>')
    max_points = max(1, len(labels))
    for index, label in enumerate(labels):
        if max_points == 1 or index % max(1, max_points // 8) == 0 or index == max_points - 1:
            x = left + chart_width * index / max(1, max_points - 1)
            lines.append(f'<text x="{round(x,2)}" y="{top + chart_height + 24}" font-size="10" fill="#6b7280" transform="rotate(30 {round(x,2)} {top + chart_height + 24})">{svg_escape(label)}</text>')
    for series_index, (metric, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = []
        for point_index, value in enumerate(values):
            if value is None:
                continue
            x = left + chart_width * point_index / max(1, max_points - 1)
            y = top + chart_height - chart_height * clamp(float(value), 0.0, 1.0)
            points.append(f"{round(x,2)},{round(y,2)}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = 940
        legend_y = 58 + series_index * 27
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{legend_y}" font-size="12" fill="#111827">{svg_escape(CORE_TREND_LABELS[metric])}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def diagnostic_metrics_line_svg(labels: list[str], series: dict[str, list[float | None]]) -> str:
    width = 1180
    height = 760
    left = 86
    top = 58
    chart_width = 820
    chart_height = 500
    colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b", "#0891b2", "#be123c", "#65a30d", "#475569", "#9333ea", "#0f766e", "#b45309", "#4f46e5", "#9f1239"]
    normalized = normalize_series_for_plot(series)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Diagnostic metrics trend">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-size="20" font-weight="700" fill="#111827">Diagnostic Metrics Trend</text>',
        '<text x="24" y="52" font-size="12" fill="#6b7280">Each metric is normalized by its own observed range. Blank values are not plotted.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#fff" stroke="#d1d5db"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + chart_height - chart_height * value
        lines.append(f'<line x1="{left}" y1="{round(y,2)}" x2="{left + chart_width}" y2="{round(y,2)}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="28" y="{round(y + 4,2)}" font-size="11" fill="#6b7280">{round(value, 2)}</text>')
    max_points = max(1, len(labels))
    for index, label in enumerate(labels):
        if max_points == 1 or index % max(1, max_points // 9) == 0 or index == max_points - 1:
            x = left + chart_width * index / max(1, max_points - 1)
            lines.append(f'<text x="{round(x,2)}" y="{top + chart_height + 24}" font-size="10" fill="#6b7280" transform="rotate(30 {round(x,2)} {top + chart_height + 24})">{svg_escape(label)}</text>')
    for series_index, metric in enumerate(DIAGNOSTIC_TREND_METRICS):
        values = normalized.get(metric, [])
        color = colors[series_index % len(colors)]
        points = []
        for point_index, value in enumerate(values):
            if value is None:
                continue
            x = left + chart_width * point_index / max(1, max_points - 1)
            y = top + chart_height - chart_height * clamp(float(value), 0.0, 1.0)
            points.append(f"{round(x,2)},{round(y,2)}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            for point in points[-1:]:
                x, y = point.split(",")
                lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        legend_x = 930
        legend_y = 64 + series_index * 28
        direction = DIAGNOSTIC_DIRECTIONS.get(metric, "")
        lines.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 18}" y="{legend_y}" font-size="11" fill="#111827">{svg_escape(DIAGNOSTIC_TREND_LABELS[metric])}</text>')
        lines.append(f'<text x="{legend_x + 18}" y="{legend_y + 12}" font-size="9" fill="#6b7280">{svg_escape(direction)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def normalize_series_for_plot(series: dict[str, list[float | None]]) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    for metric, values in series.items():
        numeric_values = [float(value) for value in values if value is not None]
        if not numeric_values:
            result[metric] = [None for _value in values]
            continue
        minimum = min(numeric_values)
        maximum = max(numeric_values)
        if maximum == minimum:
            scale = max(abs(maximum), 1.0)
            result[metric] = [None if value is None else clamp(float(value) / scale, 0.0, 1.0) for value in values]
        else:
            result[metric] = [None if value is None else (float(value) - minimum) / (maximum - minimum) for value in values]
    return result


def bar_svg(rows: list[tuple[str, float, str]], *, title: str, diverging: bool = False) -> str:
    width = 760
    row_height = 34
    height = 70 + row_height * max(1, len(rows))
    chart_width = 440
    label_x = 24
    bar_x = 250
    axis_x = 470
    lines = svg_header(width, height, title)
    if diverging:
        lines.append(f'<line x1="{axis_x}" y1="45" x2="{axis_x}" y2="{height - 20}" stroke="#6b7280" stroke-width="1"/>')
    for index, (name, value, kind) in enumerate(rows):
        y = 52 + index * row_height
        lines.append(f'<text x="{label_x}" y="{y + 18}" font-size="13" fill="#111827">{svg_escape(name)}</text>')
        if diverging:
            bar_width = min(chart_width / 2, abs(value) * chart_width / 200)
            x = axis_x if value >= 0 else axis_x - bar_width
            color = "#16a34a" if value >= 0 else "#dc2626"
            lines.append(f'<rect x="{round(x, 2)}" y="{y}" width="{round(bar_width, 2)}" height="20" fill="{color}" rx="3"/>')
            lines.append(f'<text x="{axis_x + chart_width / 2 + 16}" y="{y + 15}" font-size="12" fill="#111827">{signed_delta(value)}</text>')
        else:
            pct = clamp(value / 100, 0.0, 1.0)
            color = "#2563eb" if kind == "capability" else "#f59e0b"
            lines.append(f'<rect x="{bar_x}" y="{y}" width="{chart_width}" height="20" fill="#e5e7eb" rx="3"/>')
            lines.append(f'<rect x="{bar_x}" y="{y}" width="{round(chart_width * pct, 2)}" height="20" fill="{color}" rx="3"/>')
            lines.append(f'<text x="{bar_x + chart_width + 12}" y="{y + 15}" font-size="12" fill="#111827">{round(value, 2)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def line_svg(series: dict[str, list[float]], *, title: str) -> str:
    width = 760
    height = 320
    left = 58
    top = 50
    chart_width = 620
    chart_height = 210
    colors = ["#2563eb", "#16a34a", "#f59e0b"]
    max_points = max([len(values) for values in series.values()] or [1])
    lines = svg_header(width, height, title)
    lines.append(f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d1d5db"/>')
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = []
        for point_index, value in enumerate(values):
            if value is None:
                continue
            x = left + (chart_width * point_index / max(1, max_points - 1))
            y = top + chart_height - chart_height * clamp(value, 0.0, 100.0) / 100
            points.append(f"{round(x, 2)},{round(y, 2)}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = left + index * 190
        lines.append(f'<rect x="{legend_x}" y="270" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 14}" y="280" font-size="12" fill="#111827">{svg_escape(name)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def stacked_bar_svg(rows: list[tuple[str, float, str]], *, total: float, title: str) -> str:
    width = 760
    height = 170
    bar_x = 40
    bar_y = 62
    bar_width = 650
    lines = svg_header(width, height, title)
    current_x = bar_x
    for name, value, color in rows:
        width_value = bar_width * value / max(1.0, total)
        lines.append(f'<rect x="{round(current_x, 2)}" y="{bar_y}" width="{round(width_value, 2)}" height="28" fill="{color}"/>')
        current_x += width_value
    lines.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="28" fill="none" stroke="#111827"/>')
    legend_x = bar_x
    for name, value, color in rows:
        lines.append(f'<rect x="{legend_x}" y="112" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 14}" y="122" font-size="12" fill="#111827">{svg_escape(name)}: {int(value)}</text>')
        legend_x += 165
    lines.append("</svg>")
    return "\n".join(lines)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9fafb"/>',
        f'<text x="24" y="30" font-size="18" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
    ]


def markdown_chart_path(path: Path, output_path: Path) -> str:
    try:
        return path.relative_to(output_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def core_metric_value(cycle: dict[str, Any], metric: str) -> Any:
    if not cycle:
        return None
    raw = cycle.get("raw_metrics", {}) if isinstance(cycle.get("raw_metrics"), dict) else {}
    if metric in raw:
        return raw.get(metric)
    backfilled = backfilled_core_metrics(cycle)
    return backfilled.get(metric)


def backfilled_core_metrics(cycle: dict[str, Any]) -> dict[str, Any]:
    cached = cycle.get("_core_metric_backfill")
    if isinstance(cached, dict):
        return cached
    paths = []
    for row in cycle.get("input_reports", []):
        if not isinstance(row, dict) or row.get("exists") is False:
            continue
        path = Path(str(row.get("path", "")))
        if path.exists():
            paths.append(path)
    if not paths:
        cycle["_core_metric_backfill"] = {}
        return {}
    try:
        reports, _missing = load_capability_reports(paths)
        raw = extract_capability_raw_metrics(reports, missing_reports=[])
    except Exception:
        raw = {}
    cycle["_core_metric_backfill"] = raw
    return raw


def core_metric_numeric(cycle: dict[str, Any], metric: str) -> float | None:
    value = core_metric_value(cycle, metric)
    if metric == "qrels_trust_level":
        return QRELS_TRUST_NUMERIC.get(str(value), None)
    return optional_float(value)


def diagnostic_metric_value(cycle: dict[str, Any], metric: str) -> Any:
    if not cycle:
        return None
    raw = cycle.get("raw_metrics", {}) if isinstance(cycle.get("raw_metrics"), dict) else {}
    if metric in raw:
        return raw.get(metric)
    backfilled = backfilled_core_metrics(cycle)
    return backfilled.get(metric)


def diagnostic_metric_numeric(cycle: dict[str, Any], metric: str) -> float | None:
    return optional_float(diagnostic_metric_value(cycle, metric))


def latest_observed_diagnostic(cycles: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    for index in range(len(cycles) - 1, -1, -1):
        cycle = cycles[index]
        value = diagnostic_metric_value(cycle, metric)
        if optional_float(value) is None and value not in QRELS_TRUST_NUMERIC:
            continue
        return {"cycle": cycle, "cycle_index": index, "cycle_id": cycle.get("cycle_id", f"cycle_{index + 1}"), "value": value}
    return {"cycle": {}, "cycle_index": None, "cycle_id": "", "value": None}


def diagnostic_metric_source_detail(cycle: dict[str, Any], metric: str) -> dict[str, Any]:
    raw = cycle.get("raw_metrics", {}) if isinstance(cycle.get("raw_metrics"), dict) else {}
    sources = raw.get("metric_sources", {}) if isinstance(raw.get("metric_sources"), dict) else {}
    if metric not in sources:
        backfilled = backfilled_core_metrics(cycle)
        sources = backfilled.get("metric_sources", {}) if isinstance(backfilled.get("metric_sources"), dict) else sources
    source = sources.get(metric, {})
    return source if isinstance(source, dict) else {}


def diagnostic_source_explanation(cycles: list[dict[str, Any]], metric: str, observation: dict[str, Any]) -> str:
    detail = DIAGNOSTIC_METRIC_DETAILS.get(metric, {})
    direction = DIAGNOSTIC_DIRECTIONS.get(metric, "")
    direction_text = DIAGNOSTIC_DIRECTION_DETAILS.get(direction, direction)
    cycle = observation.get("cycle", {}) if isinstance(observation.get("cycle"), dict) else {}
    cycle_id = observation.get("cycle_id") or "no observed cycle"
    latest_value = format_trend_value(observation.get("value")) or "missing"
    source = diagnostic_metric_source_detail(cycle, metric) if cycle else {}
    source_parts = []
    if source.get("method"):
        source_parts.append(f"method={source.get('method')}")
    if source.get("raw_key"):
        source_parts.append(f"raw_key={source.get('raw_key')}")
    if source.get("path"):
        source_parts.append(f"path={source.get('path')}")
    if source.get("fallback"):
        source_parts.append(f"fallback={source.get('fallback')}")
    source_text = "; ".join(source_parts) if source_parts else "source unavailable in registry/backfill"
    use = detail.get("use", "Diagnostic metric used to guide retrieval-lab flywheel decisions.")
    meaning = detail.get("meaning", direction_text)
    return f"Latest observed={latest_value} at cycle={cycle_id}. Use: {use} Meaning: {meaning} Direction: {direction_text} Source: {source_text}."


def sparkline_svg(values: list[float | None], *, metric: str) -> str:
    width = 126
    height = 34
    left = 4
    top = 4
    chart_width = 112
    chart_height = 22
    normalized = normalize_series_for_plot({metric: values}).get(metric, [])
    points = []
    max_points = max(1, len(values))
    for index, value in enumerate(normalized):
        if value is None:
            continue
        x = left + chart_width * index / max(1, max_points - 1)
        y = top + chart_height - chart_height * clamp(float(value), 0.0, 1.0)
        points.append(f"{round(x, 2)},{round(y, 2)}")
    direction = DIAGNOSTIC_DIRECTIONS.get(metric, "")
    color = {"higher_is_better": "#16a34a", "lower_is_better": "#dc2626", "opportunity": "#f59e0b"}.get(direction, "#2563eb")
    label = DIAGNOSTIC_TREND_LABELS.get(metric, metric)
    lines = [
        f'<svg class="sparkline" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{svg_escape(label)} trend sparkline">',
        '<rect width="100%" height="100%" rx="5" fill="#f9fafb"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#e5e7eb"/>',
    ]
    if points:
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.8"/>')
    else:
        lines.append('<text x="8" y="21" fill="#9ca3af" font-size="9">no data</text>')
    lines.append(f'<title>{svg_escape(label)} | {svg_escape(direction)} | latest observed: {svg_escape(format_trend_value(next((value for value in reversed(values) if value is not None), None)))}</title>')
    lines.append("</svg>")
    return "".join(lines)


def metric_source(cycle: dict[str, Any], metric: str) -> str:
    raw = cycle.get("raw_metrics", {}) if isinstance(cycle.get("raw_metrics"), dict) else {}
    sources = raw.get("metric_sources", {}) if isinstance(raw.get("metric_sources"), dict) else {}
    if metric not in sources:
        backfilled = backfilled_core_metrics(cycle)
        sources = backfilled.get("metric_sources", {}) if isinstance(backfilled.get("metric_sources"), dict) else sources
    source = sources.get(metric, {}) if isinstance(sources.get(metric), dict) else {}
    method = source.get("method", "")
    raw_key = source.get("raw_key", metric)
    fallback = source.get("fallback")
    suffix = f" via {fallback}" if fallback else ""
    return f"{method}:{raw_key}{suffix}" if method else ""


def trend_delta(current: Any, previous: Any) -> float | None:
    current_num = optional_float(current) if current not in QRELS_TRUST_NUMERIC else QRELS_TRUST_NUMERIC.get(str(current))
    previous_num = optional_float(previous) if previous not in QRELS_TRUST_NUMERIC else QRELS_TRUST_NUMERIC.get(str(previous))
    if current_num is None or previous_num is None:
        return None
    return round(current_num - previous_num, 6)


def format_trend_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    numeric = optional_float(value)
    if numeric is None:
        return str(value)
    return str(round(numeric, 6))


def format_trend_delta(value: float | None) -> str:
    if value is None:
        return ""
    sign = "+" if value > 0 else ""
    return f"{sign}{round(value, 6)}"


def scale_metric(value: Any, *, target: float) -> float:
    numeric = optional_float(value)
    if numeric is None or target <= 0:
        return 0.0
    return clamp(numeric / target * 100, 0.0, 100.0)


def scale_metric_optional(value: Any, *, target: float) -> float | None:
    numeric = optional_float(value)
    if numeric is None or target <= 0:
        return None
    return clamp(numeric / target * 100, 0.0, 100.0)


def style_safety_score(value: Any) -> float:
    violation = optional_float(value)
    if violation is None:
        return 50.0
    if violation <= 0.05:
        return 100.0
    if violation >= 0.20:
        return 0.0
    return clamp((0.20 - violation) / 0.15 * 100, 0.0, 100.0)


def style_safety_score_optional(value: Any) -> float | None:
    if optional_float(value) is None:
        return None
    return style_safety_score(value)


def qrels_trust_score(raw_metrics: dict[str, Any]) -> float:
    level = str(raw_metrics.get("qrels_trust_level", "low"))
    base = {"low": 25.0, "medium": 60.0, "high": 90.0}.get(level, 25.0)
    qrels_count = optional_float(raw_metrics.get("qrels_count")) or 0.0
    reviewed = optional_float(raw_metrics.get("manual_or_llm_count")) or 0.0
    needs = optional_float(raw_metrics.get("needs_adjudication_count")) or 0.0
    conflict = optional_float(raw_metrics.get("vote_conflict_rate")) or 0.0
    return clamp(base + min(10.0, reviewed / max(1.0, qrels_count) * 25.0) - min(15.0, needs / max(1.0, qrels_count) * 20.0) - min(10.0, conflict * 30.0), 0.0, 100.0)


def qrels_trust_score_optional(raw_metrics: dict[str, Any]) -> float | None:
    if "qrels_trust_level" not in raw_metrics and "qrels_count" not in raw_metrics:
        return None
    return qrels_trust_score(raw_metrics)


def cycle_operability_score(raw_metrics: dict[str, Any]) -> float:
    report_count = optional_float(raw_metrics.get("report_count")) or 0.0
    missing_count = optional_float(raw_metrics.get("missing_report_count")) or 0.0
    active_sample_count = optional_float(raw_metrics.get("active_sample_count"))
    llm_call_count = optional_float(raw_metrics.get("llm_call_count")) or 0.0
    elapsed = optional_float(raw_metrics.get("elapsed_seconds_total")) or 0.0
    report_score = clamp(report_count / max(1.0, report_count + missing_count) * 100, 0.0, 100.0)
    active_score = 100.0 if active_sample_count and active_sample_count > 0 else 60.0
    llm_score = 100.0 if llm_call_count == 0 else 80.0
    elapsed_score = 100.0 if elapsed > 0 else 60.0
    return weighted_score([(report_score, 0.45), (active_score, 0.20), (llm_score, 0.20), (elapsed_score, 0.15)])


def weighted_score(parts: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _score, weight in parts)
    if total_weight <= 0:
        return 0.0
    return sum(score * weight for score, weight in parts) / total_weight


def weighted_score_optional(parts: list[tuple[float | None, float]]) -> float | None:
    available = [(score, weight) for score, weight in parts if score is not None]
    if not available:
        return None
    total_weight = sum(weight for _score, weight in available)
    if total_weight <= 0:
        return None
    return sum(float(score) * weight for score, weight in available) / total_weight


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def signed_delta(value: Any) -> str:
    numeric = optional_float(value) or 0.0
    sign = "+" if numeric > 0 else ""
    return f"{sign}{round(numeric, 3)}"


def svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def html_escape(value: Any) -> str:
    return svg_escape(value).replace("'", "&#39;")


def script_json_escape(value: str) -> str:
    return value.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def git_sha() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True)
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


__all__ = [
    "capability_delta",
    "capability_report_markdown",
    "compute_capability_scores",
    "DIAGNOSTIC_TREND_METRICS",
    "diagnostic_metrics_trend_html",
    "extract_capability_raw_metrics",
    "generate_capability_report_command",
    "record_capability_cycle_command",
]
