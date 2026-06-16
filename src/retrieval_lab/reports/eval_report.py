from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import read_json
from retrieval_lab.experiments.compare import extract_report_metrics
from retrieval_lab.reports.markdown import markdown_report, simple_metric_rows


DEFAULT_EVAL_REPORT_INPUT = Path(".tmp") / "retrieval_lab" / "experiment_comparison_latest.json"
DEFAULT_EVAL_REPORT_OUTPUT = Path(".tmp") / "retrieval_lab" / "eval_report_latest.md"


def generate_eval_report_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    paths = eval_report_paths(args)
    reports, missing = load_eval_report_sources(paths)
    output = Path(getattr(args, "output", DEFAULT_EVAL_REPORT_OUTPUT))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(eval_report_markdown(reports, missing_reports=missing, output_path=output), encoding="utf-8")
    summary = {
        "source_report_count": len(reports),
        "missing_report_count": len(missing),
        "methods": [row.get("method", "") for row in reports],
        "output": str(output),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    return {
        "method": "retrieval_lab_generate_eval_report",
        "output": str(output),
        "summary": summary,
    }


def eval_report_paths(args: Any) -> list[Path]:
    paths: list[Path] = []
    input_path = getattr(args, "input", None)
    if input_path:
        paths.append(Path(input_path))
    for path in list(getattr(args, "inputs", None) or []):
        paths.append(Path(path))
    if not paths:
        paths.append(DEFAULT_EVAL_REPORT_INPUT)
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def load_eval_report_sources(paths: list[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    reports = []
    missing = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            missing.append(path)
            continue
        data = read_json(path)
        if not isinstance(data, dict):
            missing.append(path)
            continue
        reports.append(
            {
                "path": str(path),
                "method": data.get("method", ""),
                "elapsed_seconds": data.get("elapsed_seconds", summary_elapsed_seconds(data)),
                "metrics": metrics_with_selection_score(extract_report_metrics(data)),
                "data": data,
            }
        )
    return reports, missing


def eval_report_markdown(
    reports: list[dict[str, Any]],
    *,
    missing_reports: list[Path],
    output_path: Path | None = None,
) -> str:
    lines = [
        "# Retrieval Lab Evaluation Report",
        "",
        f"- generated_at: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`",
        f"- source_report_count: `{len(reports)}`",
        f"- missing_report_count: `{len(missing_reports)}`",
        "",
    ]
    lines.extend(source_report_section(reports, missing_reports))
    lines.extend(executive_summary_section(reports))
    lines.extend(key_metrics_section(reports))
    lines.extend(comparison_section(reports))
    lines.extend(failure_section(reports))
    lines.extend(qrels_section(reports))
    lines.extend(workflow_sections(reports))
    lines.extend(next_actions_section(reports, missing_reports))
    if len(reports) == 1:
        lines.extend(["## Source Detail", ""])
        detail = markdown_report(reports[0]["data"])
        lines.extend(detail.splitlines()[1:])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def source_report_section(reports: list[dict[str, Any]], missing_reports: list[Path]) -> list[str]:
    lines = ["## Source Reports", "", "| status | method | elapsed | path |", "|---|---|---:|---|"]
    for report in reports:
        lines.append(
            "| "
            + " | ".join(
                [
                    "loaded",
                    str(report.get("method", "")),
                    str(report.get("elapsed_seconds", "")),
                    str(report.get("path", "")),
                ]
            )
            + " |"
        )
    for path in missing_reports:
        lines.append(f"| missing |  |  | {path} |")
    lines.append("")
    return lines


def executive_summary_section(reports: list[dict[str, Any]]) -> list[str]:
    if not reports:
        return ["## Executive Summary", "", "- No source reports were loaded.", ""]
    best = best_report(reports)
    facts = summary_facts(reports)
    lines = [
        "## Executive Summary",
        "",
        f"- best_report: `{best.get('method', '')}` from `{best.get('path', '')}`",
        f"- best_selection_score: `{best.get('metrics', {}).get('_selection_score', '')}`",
    ]
    for fact in facts:
        lines.append(f"- {fact}")
    lines.append("")
    return lines


def summary_facts(reports: list[dict[str, Any]]) -> list[str]:
    facts = []
    for report in reports:
        data = report["data"]
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        if "top_failure_type" in summary:
            facts.append(f"top_failure_type: `{summary.get('top_failure_type')}`")
        if "qrels_trust_level" in summary:
            facts.append(f"qrels_trust_level: `{summary.get('qrels_trust_level')}`")
        if "best_method" in summary:
            facts.append(f"experiment_best_method: `{summary.get('best_method')}`")
        if "best_workflow" in summary:
            facts.append(f"best_workflow: `{summary.get('best_workflow')}`")
        if "best_planner" in summary:
            facts.append(f"best_planner: `{summary.get('best_planner')}`")
    return facts[:8]


def key_metrics_section(reports: list[dict[str, Any]]) -> list[str]:
    metric_names = preferred_metric_names(reports)
    if not metric_names:
        return []
    lines = ["## Key Metrics", "", "| report | " + " | ".join(metric_names) + " |", "|---" + "|---:" * len(metric_names) + "|"]
    for report in reports:
        metrics = report["metrics"]
        lines.append(
            "| "
            + " | ".join([report_label(report), *[str(metrics.get(name, "")) for name in metric_names]])
            + " |"
        )
    lines.append("")
    return lines


def preferred_metric_names(reports: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "nDCG@10",
        "MRR@10",
        "Recall@10",
        "scene_level_recall_at_10",
        "target_recall_at_10",
        "stage_level_hit_at_3",
        "purpose_level_hit_at_3",
        "style_violation_at_3",
        "failure_rate",
        "Judged@10",
    ]
    available = {key for report in reports for key in report["metrics"]}
    selected = [key for key in preferred if key in available]
    if selected:
        return selected[:8]
    return sorted(available)[:8]


def comparison_section(reports: list[dict[str, Any]]) -> list[str]:
    comparison_reports = [report for report in reports if isinstance(report["data"].get("reports"), list)]
    if not comparison_reports:
        return []
    lines = ["## Experiment Comparison", ""]
    for report in comparison_reports:
        lines.extend(["| method | score | delta | path |", "|---|---:|---|---|"])
        for row in report["data"].get("reports", [])[:30]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("method", "")),
                        str(row.get("selection_score", "")),
                        compact_delta(row.get("delta_vs_baseline", {})),
                        str(row.get("path", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def failure_section(reports: list[dict[str, Any]]) -> list[str]:
    failure_reports = [report for report in reports if isinstance(report["data"].get("failures"), list)]
    if not failure_reports:
        return []
    lines = ["## Failure Analysis", ""]
    for report in failure_reports:
        summary = report["data"].get("summary", {})
        lines.extend(["| metric | value |", "|---|---:|"])
        for key in ("failure_count", "failure_rate", "top_failure_type"):
            if key in summary:
                lines.append(f"| {key} | {summary.get(key)} |")
        lines.append("")
        lines.extend(["| case_id | type | target_rank | top1 | next_action |", "|---|---|---:|---|---|"])
        for row in report["data"].get("failures", [])[:20]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("case_id", "")),
                        str(row.get("failure_type", "")),
                        str(row.get("target_rank", "")),
                        str(row.get("top1_item_id", "")),
                        str(row.get("suggested_next_action", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return lines


def qrels_section(reports: list[dict[str, Any]]) -> list[str]:
    qrels_reports = [report for report in reports if "qrels" in str(report.get("method", "")).lower()]
    if not qrels_reports:
        return []
    lines = ["## Qrels Trust", ""]
    for report in qrels_reports:
        summary = report["data"].get("summary", {})
        rows = simple_metric_rows(
            summary,
            skip={"grade_counts", "source_counts"},
        )
        if not rows:
            continue
        lines.extend(["| metric | value |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in rows)
        lines.append("")
    return lines


def workflow_sections(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for report in reports:
        data = report["data"]
        if isinstance(data.get("workflows"), dict):
            lines.extend(["## Workflow Metrics", "", "| workflow | scene@10 | stage@3 | purpose@3 | style@3 | recall@10 |", "|---|---:|---:|---:|---:|---:|"])
            for name, row in data["workflows"].items():
                summary = row.get("summary", {}) if isinstance(row, dict) else {}
                metrics = row.get("metrics", {}) if isinstance(row, dict) else {}
                overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(name),
                            str(summary.get("scene_level_recall_at_10", summary.get("target_recall_at_10", ""))),
                            str(summary.get("stage_level_hit_at_3", summary.get("stage_hit_at_3", ""))),
                            str(summary.get("purpose_level_hit_at_3", summary.get("purpose_hit_at_3", ""))),
                            str(summary.get("style_violation_at_3", "")),
                            str(overall.get("recall_at_10", "")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
        if isinstance(data.get("planners"), dict):
            lines.extend(["## Query Understanding Metrics", "", "| planner | scene@10 | stage@3 | purpose@3 | neg_leak |", "|---|---:|---:|---:|---:|"])
            for name, row in data["planners"].items():
                summary = row.get("planner_summary", {}) if isinstance(row, dict) else {}
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(name),
                            str(summary.get("target_recall_at_10", "")),
                            str(summary.get("stage_hit_at_3", "")),
                            str(summary.get("purpose_hit_at_3", "")),
                            str(summary.get("negative_leak_rate", "")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return lines


def next_actions_section(reports: list[dict[str, Any]], missing_reports: list[Path]) -> list[str]:
    actions = []
    if missing_reports:
        actions.append("Resolve missing source reports before treating this report as a complete cycle record.")
    for report in reports:
        summary = report["data"].get("summary", {}) if isinstance(report["data"].get("summary"), dict) else {}
        if summary.get("qrels_trust_level") == "low":
            actions.append("Prioritize active qrels sampling and adjudication; low qrels trust weakens all downstream deltas.")
        if float(summary.get("failure_rate", 0.0) or 0.0) > 0:
            actions.append(f"Investigate `{summary.get('top_failure_type', 'unknown')}` failures before adding more retrieval complexity.")
        if float(summary.get("style_violation_at_3", 0.0) or 0.0) > 0.05:
            actions.append("Run style-risk validation and mine hard negatives for soft style constraints.")
    if not actions:
        actions.append("Continue the measured flywheel: run qrels audit, fuzzy multirelevance, rerank upper bound, then record a capability cycle.")
    lines = ["## Next Actions", ""]
    for action in dedupe(actions)[:8]:
        lines.append(f"- {action}")
    lines.append("")
    return lines


def best_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return max(reports, key=lambda report: report["metrics"].get("_selection_score", 0.0))


def metrics_with_selection_score(metrics: dict[str, float]) -> dict[str, float]:
    return {**metrics, "_selection_score": selection_score(metrics)}


def summary_elapsed_seconds(data: dict[str, Any]) -> Any:
    summary = data.get("summary", {})
    if isinstance(summary, dict):
        return summary.get("elapsed_seconds")
    return None


def report_label(report: dict[str, Any]) -> str:
    method = str(report.get("method", ""))
    path = Path(str(report.get("path", ""))).stem
    return method or path


def selection_score(metrics: dict[str, float]) -> float:
    positive = [
        metrics.get("nDCG@10"),
        metrics.get("MRR@10"),
        metrics.get("Recall@10"),
        metrics.get("scene_level_recall_at_10"),
        metrics.get("target_recall_at_10"),
    ]
    negative = [
        metrics.get("style_violation_at_3"),
        metrics.get("failure_rate"),
        metrics.get("Unjudged@10"),
    ]
    values = [value for value in positive if value is not None]
    values.extend(1 - value for value in negative if value is not None)
    return round(sum(values) / max(1, len(values)), 6)


def compact_delta(delta: Any) -> str:
    if not isinstance(delta, dict) or not delta:
        return ""
    parts = []
    for key, value in list(delta.items())[:4]:
        try:
            numeric = float(value)
            sign = "+" if numeric > 0 else ""
            parts.append(f"{key}={sign}{round(numeric, 4)}")
        except (TypeError, ValueError):
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def dedupe(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


__all__ = [
    "DEFAULT_EVAL_REPORT_INPUT",
    "DEFAULT_EVAL_REPORT_OUTPUT",
    "eval_report_markdown",
    "eval_report_paths",
    "generate_eval_report_command",
    "load_eval_report_sources",
]
