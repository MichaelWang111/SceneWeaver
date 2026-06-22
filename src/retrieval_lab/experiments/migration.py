from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, write_json
from retrieval_lab.experiments.constants import CORE_EXPERIMENT_COMMANDS
from retrieval_lab.experiments.coverage import command_coverage_rows, core_experiment_replacement_coverage, layer_coverage_rows


DEFAULT_MIGRATION_AUDIT_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "migration_audit_latest.json"
DEFAULT_MIGRATION_CERTIFICATION_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "migration_certification_latest.json"


def migration_audit_command(args: Any) -> dict[str, Any]:
    output = Path(getattr(args, "output", DEFAULT_MIGRATION_AUDIT_REPORT_PATH))
    previous_path = Path(getattr(args, "previous", "")) if getattr(args, "previous", "") else None
    markdown_output = Path(getattr(args, "markdown_output", "")) if getattr(args, "markdown_output", "") else None
    round_id = str(getattr(args, "round_id", "") or "round")
    label = str(getattr(args, "label", "") or "")

    command_rows = command_coverage_rows()
    layer_rows = layer_coverage_rows(Path(__file__).resolve().parents[1])
    coverage = coverage_summary(command_rows, layer_rows)
    mocktesting = mocktesting_diff_status()
    previous_summary = load_previous_summary(previous_path)
    delta = coverage_delta(coverage, previous_summary)
    report = {
        "method": "retrieval_lab_migration_audit",
        "round_id": round_id,
        "label": label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": {
            **coverage,
            "mocktesting_clean": mocktesting["clean"],
            "mocktesting_diff_line_count": mocktesting["diff_line_count"],
            "previous_report": str(previous_path) if previous_path else "",
            "output": str(output),
        },
        "delta_vs_previous": delta,
        "mocktesting": mocktesting,
        "command_coverage": command_rows,
        "layer_coverage": layer_rows,
        "self_check_contract": {
            "mocktesting_must_remain_clean": True,
            "infra_audit_must_run": True,
            "focused_tests_required": "tests/test_retreieval_lab_cli.py",
            "full_tests_required": "pytest -q --basetemp .pytest-tmp/<round> -p no:cacheprovider",
        },
        "recommendations": migration_recommendations(coverage, mocktesting),
    }
    report["fingerprint"] = data_sha256(
        {
            "round_id": round_id,
            "coverage": coverage,
            "mocktesting": mocktesting,
            "delta": delta,
        }
    )
    write_json(output, report)
    if markdown_output:
        write_migration_markdown(markdown_output, report)
    return {
        "method": "retrieval_lab_migration_audit",
        "output": str(output),
        "summary": {**report["summary"], "fingerprint": report["fingerprint"]},
    }


def migration_certify_command(args: Any) -> dict[str, Any]:
    output = Path(getattr(args, "output", DEFAULT_MIGRATION_CERTIFICATION_REPORT_PATH))
    markdown_output = Path(getattr(args, "markdown_output", "")) if getattr(args, "markdown_output", "") else None
    round_id = str(getattr(args, "round_id", "") or "certification")
    command_rows = command_coverage_rows()
    layer_rows = layer_coverage_rows(Path(__file__).resolve().parents[1])
    mocktesting = mocktesting_diff_status()
    core_rows = [row for row in command_rows if row["command"] in CORE_EXPERIMENT_COMMANDS]
    critical_compat = [row["command"] for row in core_rows if row["status"] == "compat_only"]
    parity_reports = [str(path) for path in list(getattr(args, "parity_reports", []) or [])]
    test_reports = [str(path) for path in list(getattr(args, "test_reports", []) or [])]
    summary = {
        **coverage_summary(command_rows, layer_rows),
        "legacy_command_coverage": coverage_summary(command_rows, layer_rows)["coverage_rate"],
        "core_experiment_replacement_coverage": core_experiment_replacement_coverage(command_rows),
        "critical_command_count": len(core_rows),
        "critical_command_compat_count": len(critical_compat),
        "critical_commands_native_or_partial": len(core_rows) - len(critical_compat),
        "mocktesting_clean": mocktesting["clean"],
        "mocktesting_diff_line_count": mocktesting["diff_line_count"],
        "parity_report_count": len(parity_reports),
        "test_report_count": len(test_reports),
        "certified": mocktesting["clean"] and not critical_compat,
        "output": str(output),
    }
    report = {
        "method": "retrieval_lab_migration_certification",
        "round_id": round_id,
        "label": str(getattr(args, "label", "") or ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "mocktesting": mocktesting,
        "critical_command_coverage": core_rows,
        "critical_compat_commands": critical_compat,
        "parity_reports": parity_reports,
        "test_reports": test_reports,
        "acceptance_criteria": {
            "mocktesting_clean": True,
            "critical_command_compat_count": 0,
            "llm_default": "off",
            "human_annotation_required": False,
        },
        "recommendations": certification_recommendations(summary, critical_compat),
    }
    report["fingerprint"] = data_sha256({"summary": summary, "critical": core_rows, "mocktesting": mocktesting})
    write_json(output, report)
    if markdown_output:
        write_certification_markdown(markdown_output, report)
    return {
        "method": "retrieval_lab_migration_certification",
        "output": str(output),
        "summary": {**summary, "fingerprint": report["fingerprint"]},
    }


def coverage_summary(command_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> dict[str, Any]:
    legacy_rows = [row for row in command_rows if row["status"] != "native_only"]
    return {
        "legacy_command_count": len(legacy_rows),
        "native_command_count": sum(1 for row in command_rows if row["status"] == "native"),
        "partial_command_count": sum(1 for row in command_rows if row["status"] == "partial"),
        "compat_only_command_count": sum(1 for row in command_rows if row["status"] == "compat_only"),
        "native_only_command_count": sum(1 for row in command_rows if row["status"] == "native_only"),
        "layer_count": len(layer_rows),
        "empty_layer_count": sum(1 for row in layer_rows if row["status"] == "empty"),
        "implemented_layer_count": sum(1 for row in layer_rows if row["status"] == "implemented"),
        "coverage_rate": round(
            sum(1 for row in legacy_rows if row["status"] in {"native", "partial"}) / max(1, len(legacy_rows)),
            6,
        ),
        "core_experiment_replacement_coverage": core_experiment_replacement_coverage(command_rows),
    }


def mocktesting_diff_status() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "diff", "--", "src/mocktesting"],
        check=False,
        capture_output=True,
        text=True,
    )
    diff = result.stdout or ""
    return {
        "clean": diff.strip() == "",
        "diff_line_count": len([line for line in diff.splitlines() if line.strip()]),
        "git_returncode": result.returncode,
        "diff_preview": diff.splitlines()[:40],
    }


def load_previous_summary(previous_path: Path | None) -> dict[str, Any]:
    if not previous_path or not previous_path.exists():
        return {}
    data = read_json(previous_path)
    return data.get("summary", {}) if isinstance(data, dict) else {}


def coverage_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "native_command_count",
        "partial_command_count",
        "compat_only_command_count",
        "native_only_command_count",
        "empty_layer_count",
        "coverage_rate",
    )
    delta: dict[str, Any] = {}
    for key in keys:
        if key in previous:
            current_value = current.get(key, 0)
            previous_value = previous.get(key, 0)
            if isinstance(current_value, int | float) and isinstance(previous_value, int | float):
                delta[key] = round(current_value - previous_value, 6)
    return delta


def migration_recommendations(coverage: dict[str, Any], mocktesting: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    if not mocktesting["clean"]:
        rows.append(
            {
                "priority": "P0",
                "title": "Restore mocktesting baseline cleanliness",
                "reason": "The baseline package must remain unchanged for trustworthy parity checks.",
            }
        )
    if coverage["compat_only_command_count"] > 0:
        rows.append(
            {
                "priority": "P1",
                "title": "Continue migrating compat-only commands",
                "reason": "Native Retrieval Lab coverage still depends on the compatibility backend.",
            }
        )
    if coverage["empty_layer_count"] > 0:
        rows.append(
            {
                "priority": "P1",
                "title": "Fill empty architecture layers",
                "reason": "Empty layers hide migration risk and should at least expose contracts.",
            }
        )
    return rows


def write_migration_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    delta = report.get("delta_vs_previous", {})
    lines = [
        "# Retrieval Lab Migration Audit",
        "",
        f"- round_id: `{report.get('round_id', '')}`",
        f"- label: `{report.get('label', '')}`",
        f"- mocktesting_clean: `{summary.get('mocktesting_clean')}`",
        f"- coverage_rate: `{summary.get('coverage_rate')}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    if delta:
        lines.extend(["", "## Delta Vs Previous", "", "| metric | delta |", "|---|---:|"])
        for key, value in delta.items():
            lines.append(f"| {key} | {value} |")
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for row in recommendations:
            lines.append(f"- `{row.get('priority')}` {row.get('title')}: {row.get('reason')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def certification_recommendations(summary: dict[str, Any], critical_compat: list[str]) -> list[dict[str, str]]:
    rows = []
    if not summary.get("mocktesting_clean"):
        rows.append({"priority": "P0", "title": "Restore mocktesting cleanliness", "reason": "Baseline drift invalidates parity."})
    if critical_compat:
        rows.append(
            {
                "priority": "P0",
                "title": "Migrate remaining critical commands",
                "reason": ", ".join(critical_compat[:8]),
            }
        )
    if not rows:
        rows.append(
            {
                "priority": "P1",
                "title": "Run parity and full test.md reports",
                "reason": "Core commands are native; final confidence depends on parity artifacts and full pytest output.",
            }
        )
    return rows


def write_certification_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    lines = [
        "# Retrieval Lab Migration Certification",
        "",
        f"- round_id: `{report.get('round_id', '')}`",
        f"- certified: `{summary.get('certified')}`",
        f"- mocktesting_clean: `{summary.get('mocktesting_clean')}`",
        f"- core_experiment_replacement_coverage: `{summary.get('core_experiment_replacement_coverage')}`",
        f"- critical_command_compat_count: `{summary.get('critical_command_compat_count')}`",
        "",
        "## Critical Commands",
        "",
        "| command | status | note |",
        "|---|---|---|",
    ]
    for row in report.get("critical_command_coverage", []):
        lines.append(f"| {row.get('command')} | {row.get('status')} | {row.get('note')} |")
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for row in recommendations:
            lines.append(f"- `{row.get('priority')}` {row.get('title')}: {row.get('reason')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_MIGRATION_AUDIT_REPORT_PATH",
    "DEFAULT_MIGRATION_CERTIFICATION_REPORT_PATH",
    "coverage_summary",
    "migration_audit_command",
    "migration_certify_command",
    "mocktesting_diff_status",
]
