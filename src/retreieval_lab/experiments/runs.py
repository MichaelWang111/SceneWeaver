from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, read_json, write_json


DEFAULT_RUN_ARTIFACT_PATH = Path(".tmp") / "retrieval_lab" / "run_rows_latest.json"


def export_run_artifact_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    reports = [Path(path) for path in list(getattr(args, "reports", []) or [])]
    if not reports:
        raise ValueError("at least one --reports path is required")
    run_rows: dict[str, list[dict[str, Any]]] = {}
    sources = []
    for report_path in reports:
        source = extract_run_rows_from_path(report_path, run_name=getattr(args, "run_name", ""))
        sources.append(source["source"])
        for run_name, rows in source["run_rows"].items():
            unique_name = unique_run_name(run_rows, run_name)
            run_rows[unique_name] = rows
    cases = cases_from_run_rows(run_rows)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = run_artifact_summary(run_rows, cases)
    summary.update(
        {
            "source_report_count": len(reports),
            "skipped_report_count": sum(1 for source in sources if source.get("skipped")),
            "elapsed_seconds": elapsed_seconds,
        }
    )
    artifact = {
        "method": "retrieval_lab_run_artifact",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_rows": run_rows,
        "cases": cases,
        "source_reports": sources,
        "summary": summary,
        "fingerprint": data_sha256(run_rows),
    }
    output = Path(getattr(args, "output", DEFAULT_RUN_ARTIFACT_PATH))
    write_json(output, artifact)
    return {
        "method": "retrieval_lab_export_run_artifact",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": artifact["fingerprint"]},
    }


def extract_run_rows_from_path(path: Path, *, run_name: str = "") -> dict[str, Any]:
    path = Path(path)
    try:
        report = read_json(path)
    except Exception as exc:
        return {
            "source": {
                "path": str(path),
                "method": "",
                "skipped": True,
                "skip_reason": f"json_parse_error: {exc}",
                "extracted_run_count": 0,
                "row_count": 0,
            },
            "run_rows": {},
        }
    return extract_run_rows_from_report(report, source_path=path, run_name=run_name)


def extract_run_rows_from_report(
    report: dict[str, Any],
    *,
    source_path: Path | None = None,
    run_name: str = "",
) -> dict[str, Any]:
    extracted: dict[str, list[dict[str, Any]]] = {}
    if isinstance(report.get("run_rows"), dict):
        for name, rows in report["run_rows"].items():
            normalized = normalize_run_rows(rows)
            if normalized:
                extracted[str(name)] = normalized
    if isinstance(report.get("cases"), list):
        normalized = normalize_run_rows(report["cases"])
        if normalized:
            extracted[run_name or default_run_name(report, source_path)] = normalized
    workflows = report.get("workflows", {})
    if isinstance(workflows, dict):
        for workflow_name, workflow_report in workflows.items():
            if isinstance(workflow_report, dict) and isinstance(workflow_report.get("cases"), list):
                normalized = normalize_run_rows(workflow_report["cases"])
                if normalized:
                    extracted[str(workflow_name)] = normalized
    row_count = sum(len(rows) for rows in extracted.values())
    skipped = row_count == 0
    return {
        "source": {
            "path": str(source_path) if source_path is not None else "",
            "method": report.get("method", ""),
            "skipped": skipped,
            "skip_reason": "no ranked rows found" if skipped else "",
            "extracted_run_count": len(extracted),
            "row_count": row_count,
        },
        "run_rows": extracted,
    }


def normalize_run_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("case_id") or not isinstance(row.get("top_results"), list):
            continue
        copied = dict(row)
        copied["top_results"] = [dict(result) for result in row.get("top_results", []) if isinstance(result, dict)]
        normalized.append(copied)
    return normalized


def default_run_name(report: dict[str, Any], source_path: Path | None = None) -> str:
    planner = str(report.get("query_planner", "") or "")
    ranking_key = str(report.get("ranking_key", "") or "")
    if planner and ranking_key:
        return f"{planner}::{ranking_key}"
    if ranking_key:
        return ranking_key
    method = str(report.get("method", "") or "")
    if method:
        return method
    if source_path is not None:
        return source_path.stem
    return "run"


def unique_run_name(existing: dict[str, Any], run_name: str) -> str:
    if run_name not in existing:
        return run_name
    index = 2
    while f"{run_name}#{index}" in existing:
        index += 1
    return f"{run_name}#{index}"


def cases_from_run_rows(run_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rows in run_rows.values():
        for row in rows:
            case_id = str(row.get("case_id", ""))
            if case_id and case_id not in by_id:
                by_id[case_id] = {
                    "case_id": case_id,
                    "user_input": row.get("user_input", ""),
                    "target_item_id": row.get("target_item_id"),
                    "target_stage": row.get("target_stage"),
                    "target_purposes": row.get("target_purposes", []),
                }
    return list(by_id.values())


def run_artifact_summary(run_rows: dict[str, list[dict[str, Any]]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    row_counts = {run_name: len(rows) for run_name, rows in run_rows.items()}
    top_result_counts = [
        len(row.get("top_results", []))
        for rows in run_rows.values()
        for row in rows
    ]
    ranking_key_counts = Counter(
        str(row.get("ranking_key") or row.get("top_results", [{}])[0].get("ranking_key", ""))
        for rows in run_rows.values()
        for row in rows
    )
    return {
        "run_count": len(run_rows),
        "case_count": len(cases),
        "row_count": sum(row_counts.values()),
        "row_counts": row_counts,
        "mean_top_results": round(sum(top_result_counts) / max(1, len(top_result_counts)), 6),
        "ranking_key_counts": dict(sorted(ranking_key_counts.items())),
    }


__all__ = [
    "DEFAULT_RUN_ARTIFACT_PATH",
    "cases_from_run_rows",
    "default_run_name",
    "export_run_artifact_command",
    "extract_run_rows_from_path",
    "extract_run_rows_from_report",
    "normalize_run_rows",
    "run_artifact_summary",
]
