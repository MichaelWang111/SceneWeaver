from __future__ import annotations

from typing import Any

from retrieval_lab.capability.service import capability_report_markdown


def markdown_report(report: dict[str, Any]) -> str:
    title = report_title(report)
    lines = [
        f"# {title}",
        "",
        f"- method: `{report.get('method', '')}`",
        f"- elapsed_seconds: `{report.get('elapsed_seconds', '')}`",
        "",
    ]
    summary = report.get("summary", {})
    if isinstance(summary, dict) and summary:
        lines.extend(["## Summary", "", "| metric | value |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in simple_metric_rows(summary))
        lines.append("")
    experiment = report.get("experiment", {})
    if isinstance(experiment, dict) and experiment:
        lines.extend(["## Experiment", ""])
        for key in ("command", "git_sha", "elapsed_seconds"):
            if key in experiment:
                lines.append(f"- {key}: `{experiment.get(key)}`")
        lines.append("")
    low_confidence = report.get("low_confidence_examples", [])
    if isinstance(low_confidence, list) and low_confidence:
        lines.extend(["## Low Confidence Examples", "", "| query_id | item_id | grade | confidence |", "|---|---|---:|---:|"])
        for row in low_confidence[:20]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("query_id", "")),
                        str(row.get("item_id", "")),
                        str(row.get("grade", "")),
                        str(row.get("confidence", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    changed = report.get("changed_examples", [])
    if isinstance(changed, list) and changed:
        lines.extend(["## Changed Examples", "", "| query_id | item_id | grade | source |", "|---|---|---:|---|"])
        for row in changed[:20]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("query_id", "")),
                        str(row.get("item_id", "")),
                        str(row.get("grade", "")),
                        str(row.get("source", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    failures = report.get("failures", [])
    if isinstance(failures, list) and failures:
        lines.extend(
            [
                "## Failures",
                "",
                "| case_id | run | type | target_rank | top1 | next_action |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for row in failures[:30]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("case_id", "")),
                        str(row.get("run_name", "")),
                        str(row.get("failure_type", "")),
                        str(row.get("target_rank", "")),
                        str(row.get("top1_item_id", "")),
                        str(row.get("suggested_next_action", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    reports = report.get("reports", [])
    if isinstance(reports, list) and reports:
        lines.extend(
            [
                "## Compared Reports",
                "",
                "| label | method | score | path |",
                "|---|---|---:|---|",
            ]
        )
        for row in reports[:50]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("label", "")),
                        str(row.get("method", "")),
                        str(row.get("selection_score", "")),
                        str(row.get("path", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    run_metrics = report.get("run_metrics", {})
    if isinstance(run_metrics, dict) and run_metrics:
        lines.extend(
            [
                "## Run Metrics",
                "",
                "| run | nDCG@3 | nDCG@10 | ERR@10 | MRR@10 | Judged@10 | Recall@10 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run_name, metrics in run_metrics.items():
            if not isinstance(metrics, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(run_name),
                        str(metrics.get("nDCG@3", "")),
                        str(metrics.get("nDCG@10", "")),
                        str(metrics.get("ERR@10", "")),
                        str(metrics.get("MRR@10", "")),
                        str(metrics.get("Judged@10", "")),
                        str(metrics.get("Recall@10", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    command_coverage = report.get("command_coverage", [])
    if isinstance(command_coverage, list) and command_coverage:
        lines.extend(["## Command Coverage", "", "| command | status | note |", "|---|---|---|"])
        for row in command_coverage:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("command", "")),
                        str(row.get("status", "")),
                        str(row.get("note", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    layer_coverage = report.get("layer_coverage", [])
    if isinstance(layer_coverage, list) and layer_coverage:
        lines.extend(["## Layer Coverage", "", "| layer | status | files | expectation |", "|---|---|---:|---|"])
        for row in layer_coverage:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("layer", "")),
                        str(row.get("status", "")),
                        str(row.get("python_file_count", "")),
                        str(row.get("expectation", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    fixtures = report.get("fixtures", [])
    if isinstance(fixtures, list) and fixtures:
        lines.extend(["## Fixtures", "", "| fixture | cases | scenes | industries | styles |", "|---|---:|---:|---|---|"])
        for row in fixtures[:40]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("fixture_id", "")),
                        str(row.get("case_count", "")),
                        str(row.get("target_scene_count", "")),
                        ", ".join(str(item) for item in row.get("industries", [])),
                        ", ".join(str(item) for item in row.get("styles", [])),
                    ]
                )
                + " |"
            )
        lines.append("")
    sample_cases = report.get("sample_cases", [])
    if isinstance(sample_cases, list) and sample_cases:
        lines.extend(["## Sample Cases", "", "| case_id | split | type | fixture | stage |", "|---|---|---|---|---|"])
        for row in sample_cases:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("case_id", "")),
                        str(row.get("split", "")),
                        str(row.get("case_type", "")),
                        str(row.get("fixture_id", "")),
                        str(row.get("target_stage", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def report_title(report: dict[str, Any]) -> str:
    method = str(report.get("method", ""))
    if "qrels_audit" in method:
        return "Qrels Audit Report"
    if "merge_adjudicated_qrels" in method:
        return "Qrels Adjudication Merge Report"
    if "failure_analysis" in method:
        return "Failure Analysis Report"
    if "experiment_comparison" in method:
        return "Experiment Comparison Report"
    if "run_evaluation" in method:
        return "Run Evaluation Report"
    if "infra_coverage" in method:
        return "Infra Coverage Audit"
    if "dataset_manifest" in method:
        return "Dataset Manifest"
    return "Retrieval Lab Report"


def simple_metric_rows(data: dict[str, Any], *, skip: set[str] | None = None) -> list[tuple[str, str]]:
    skip = skip or set()
    rows = []
    for key, value in data.items():
        if key in skip or isinstance(value, (dict, list)):
            continue
        rows.append((str(key), str(value)))
    return rows


__all__ = ["capability_report_markdown", "markdown_report", "simple_metric_rows"]
