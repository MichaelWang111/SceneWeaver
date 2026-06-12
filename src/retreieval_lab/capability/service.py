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
    "scene_level_recall_at_10": "scene_level_recall_at_10",
    "stage_level_hit_at_3": "stage_level_hit_at_3",
    "purpose_level_hit_at_3": "purpose_level_hit_at_3",
    "style_violation_at_3": "style_violation_at_3",
    "rerank_opportunity_nDCG@10": "rerank_opportunity_ndcg_at_10",
    "oracle_rerank_nDCG@10": "oracle_rerank_ndcg_at_10",
    "baseline_nDCG@10": "baseline_ndcg_at_10",
    "qrels_trust_level": "qrels_trust_level",
    "manual_or_llm_count": "manual_or_llm_count",
    "manual_count": "manual_count",
    "llm_count": "llm_count",
    "bootstrap_only_count": "bootstrap_only_count",
    "needs_adjudication_count": "needs_adjudication_count",
    "vote_conflict_rate": "vote_conflict_rate",
    "qrels_count": "qrels_count",
    "sample_count": "active_sample_count",
    "llm_call_count": "llm_call_count",
}


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
    output.write_text(capability_report_markdown(cycles, charts, output_path=output), encoding="utf-8")
    latest = cycles[-1] if cycles else {}
    return {
        "method": "retrieval_lab_capability_report",
        "registry": str(registry),
        "output": str(output),
        "charts": {key: str(value) for key, value in charts.items()},
        "summary": {
            "cycle_count": len(cycles),
            "latest_cycle_id": latest.get("cycle_id"),
            "latest_overall_score": latest.get("summary", {}).get("overall_score"),
            "output": str(output),
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
    for report in reports:
        method = str(report.get("method", ""))
        elapsed_seconds += float(report.get("elapsed_seconds") or 0.0)
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
    values["elapsed_seconds_total"] = round(elapsed_seconds, 6)
    values["metric_sources"] = sources
    return values


def capability_metric_priority(method: str, metric_key: str) -> int:
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
        "scene_level_recall_at_10",
        "stage_level_hit_at_3",
        "purpose_level_hit_at_3",
        "style_violation_at_3",
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
            weighted_score(
                [
                    (scale_metric(raw_metrics.get("ndcg_at_10"), target=0.75), 0.45),
                    (scale_metric(raw_metrics.get("mrr_at_10"), target=0.9), 0.25),
                    (scale_metric(raw_metrics.get("scene_level_recall_at_10"), target=0.7), 0.30),
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
            weighted_score(
                [
                    (scale_metric(raw_metrics.get("stage_level_hit_at_3"), target=0.95), 0.35),
                    (scale_metric(raw_metrics.get("purpose_level_hit_at_3"), target=0.9), 0.35),
                    (scale_metric(raw_metrics.get("scene_level_recall_at_10"), target=0.7), 0.30),
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
            style_safety_score(raw_metrics.get("style_violation_at_3")),
            "capability",
            {"style_violation_at_3": raw_metrics.get("style_violation_at_3")},
        ),
        "qrels_trust": capability_row(
            qrels_trust_score(raw_metrics),
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
            scale_metric(raw_metrics.get("rerank_opportunity_ndcg_at_10"), target=0.30),
            "opportunity",
            {"rerank_opportunity_ndcg_at_10": raw_metrics.get("rerank_opportunity_ndcg_at_10")},
        ),
    }


def capability_row(score: float, kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"score": round(clamp(score, 0.0, 100.0), 3), "kind": kind, "inputs": inputs}


def capability_delta(
    capabilities: dict[str, dict[str, Any]],
    previous_capabilities: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, row in capabilities.items():
        current = float(row.get("score", 0.0))
        previous = None
        if previous_capabilities and name in previous_capabilities:
            previous = float(previous_capabilities[name].get("score", 0.0))
        result[name] = {
            "current_score": round(current, 3),
            "previous_score": round(previous, 3) if previous is not None else None,
            "score_delta": round(current - previous, 3) if previous is not None else 0.0,
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
    scores = [float(row["score"]) for row in capabilities.values() if row.get("kind") == "capability"]
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
        "improved_capabilities": [name for name, row in delta.items() if float(row.get("score_delta", 0.0)) > 0.01],
        "regressed_capabilities": [name for name, row in delta.items() if float(row.get("score_delta", 0.0)) < -0.01],
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
                "command": "python -m retreieval_lab qrels sample-active --split test --limit 60 --sample-size 80 --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\active_qrels_next.jsonl",
            }
        )
    if (optional_float(raw_metrics.get("rerank_opportunity_ndcg_at_10")) or 0.0) >= 0.15:
        recommendations.append(
            {
                "priority": 2,
                "title": "Run real reranker sample",
                "reason": "oracle rerank has a large nDCG@10 opportunity.",
                "command": "python -m retreieval_lab eval rerank-upper-bound --split test --limit 60 --qrels .tmp\\pooled_qrels_next.jsonl --llm-rerank-sample-size 10",
            }
        )
    if (optional_float(raw_metrics.get("style_violation_at_3")) or 0.0) > 0.05:
        recommendations.append(
            {
                "priority": 3,
                "title": "Tighten style negative handling",
                "reason": "style_violation_at_3 is above the first-stage target.",
                "command": "python -m retreieval_lab eval style-risk --split test --limit 60",
            }
        )
    return sorted(recommendations, key=lambda row: int(row["priority"])) or [
        {
            "priority": 9,
            "title": "Continue the measured flywheel",
            "reason": "No critical bottleneck was detected.",
            "command": "python -m retreieval_lab flywheel guide",
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
        lines.append(f"| {name} | {row.get('score', '')} | {signed_delta(delta)} | {row.get('kind', '')} |")
    lines.extend(["", "## Bottleneck Analysis", ""])
    for recommendation in latest.get("recommendations", []):
        lines.append(f"- **{recommendation['title']}**: {recommendation['reason']}")
    lines.extend(["", "## Next Cycle Recommendation", ""])
    for recommendation in latest.get("recommendations", []):
        lines.append(f"- `{recommendation['command']}`")
    lines.append("")
    return "\n".join(lines)


def capability_bar_svg(cycle: dict[str, Any]) -> str:
    rows = [(name, float(row.get("score", 0.0)), row.get("kind", "capability")) for name, row in cycle.get("capabilities", {}).items()]
    return bar_svg(rows, title="Capability Scores")


def capability_delta_svg(cycle: dict[str, Any]) -> str:
    rows = [(name, float(row.get("score_delta", 0.0)), "delta") for name, row in cycle.get("delta_vs_previous", {}).items()]
    return bar_svg(rows, title="Delta vs Previous Cycle", diverging=True)


def capability_trend_svg(cycles: list[dict[str, Any]]) -> str:
    series = {
        "retrieval_quality": [float(cycle.get("capabilities", {}).get("retrieval_quality", {}).get("score", 0.0)) for cycle in cycles],
        "fuzzy_understanding": [float(cycle.get("capabilities", {}).get("fuzzy_understanding", {}).get("score", 0.0)) for cycle in cycles],
        "qrels_trust": [float(cycle.get("capabilities", {}).get("qrels_trust", {}).get("score", 0.0)) for cycle in cycles],
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


def scale_metric(value: Any, *, target: float) -> float:
    numeric = optional_float(value)
    if numeric is None or target <= 0:
        return 0.0
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


def qrels_trust_score(raw_metrics: dict[str, Any]) -> float:
    level = str(raw_metrics.get("qrels_trust_level", "low"))
    base = {"low": 25.0, "medium": 60.0, "high": 90.0}.get(level, 25.0)
    qrels_count = optional_float(raw_metrics.get("qrels_count")) or 0.0
    reviewed = optional_float(raw_metrics.get("manual_or_llm_count")) or 0.0
    needs = optional_float(raw_metrics.get("needs_adjudication_count")) or 0.0
    conflict = optional_float(raw_metrics.get("vote_conflict_rate")) or 0.0
    return clamp(base + min(10.0, reviewed / max(1.0, qrels_count) * 25.0) - min(15.0, needs / max(1.0, qrels_count) * 20.0) - min(10.0, conflict * 30.0), 0.0, 100.0)


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
    "extract_capability_raw_metrics",
    "generate_capability_report_command",
    "record_capability_cycle_command",
]
