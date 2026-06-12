from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, write_json
from retreieval_lab.compat import LEGACY_COMMANDS


DEFAULT_INFRA_AUDIT_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "infra_audit_latest.json"

NATIVE_COMMANDS = {
    "retrieval-flywheel-guide",
    "audit-qrels",
    "pool-qrels-from-runs",
    "sample-active-qrels-from-runs",
    "merge-adjudicated-qrels",
    "record-capability-cycle",
    "generate-capability-report",
    "write-artifact-manifest",
    "export-run-artifact",
    "run-legacy-with-artifacts",
    "rerank-run-artifact",
    "workflow-run-artifact",
    "analyze-failures-from-runs",
    "evaluate-runs",
    "compare-experiments",
    "generate-eval-report",
    "index-inspect",
    "index-manifest",
    "migration-audit",
    "planner-audit-cache",
    "planner-compare",
    "planner-plan",
    "retrieval-compare-legacy",
    "retrieval-run",
    "schema-catalog",
    "schema-show",
    "schema-validate",
    "workflow-compare-runs",
}

PARTIALLY_NATIVE_LEGACY_COMMANDS = {
    "audit-qrels": "native route exists for qrels audit",
    "merge-adjudicated-qrels": "native route exists for qrels merge",
    "compare-experiments": "native report comparator exists",
    "compare-ranking-workflows": "native workflow compare-runs exists for saved artifacts; fresh ranking comparison remains compatibility-backed",
    "generate-capability-report": "native capability report exists",
    "record-capability-cycle": "native capability cycle exists",
    "generate-eval-report": "native eval report exists",
    "analyze-failures": "native artifact-first failure analysis exists; legacy still offers fresh-signal analysis",
    "analyze-recall-bound": "metric helpers are native; dataset-running command still compatibility-backed",
    "build-pooled-qrels": "native pool-from-runs exists; ranking-first pooled build is compatibility-backed",
    "sample-active-qrels": "native sample-active-from-runs exists; ranking-first sampler is compatibility-backed",
}

LAYER_EXPECTATIONS = {
    "config": "path conventions and package metadata",
    "artifacts": "JSON/JSONL IO, fingerprints, manifests",
    "qrels": "qrels validation, audit, pooling, active sampling, adjudication",
    "evaluators": "graded metrics, run evaluation, failure attribution",
    "ranking": "rerank and workflow rerank over run artifacts",
    "experiments": "legacy bridge, run export, experiment comparison, flywheel",
    "reports": "Markdown reports and capability reports",
    "capability": "capability cycle registry and charts",
    "schemas": "versioned contracts for query plans, run rows, qrels, indexes, and LLM judgements",
    "datasets": "case registries, split manifests, fixture inventory",
    "planners": "native query understanding planners and cache",
    "indexes": "native prepared index/cache metadata",
    "retrieval": "native recall/search workflows",
    "fixtures": "test fixture registry",
    "llm": "optional LLM client adapters and fake clients",
}


def audit_infra_coverage_command(args: Any) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    rows = command_coverage_rows()
    layer_rows = layer_coverage_rows(package_root)
    summary = {
        "legacy_command_count": len(LEGACY_COMMANDS),
        "native_command_count": len(NATIVE_COMMANDS),
        "fully_native_legacy_command_count": sum(1 for row in rows if row["status"] == "native"),
        "partially_native_legacy_command_count": sum(1 for row in rows if row["status"] == "partial"),
        "compat_only_legacy_command_count": sum(1 for row in rows if row["status"] == "compat_only"),
        "native_only_command_count": sum(1 for row in rows if row["status"] == "native_only"),
        "layer_count": len(layer_rows),
        "empty_layer_count": sum(1 for row in layer_rows if row["status"] == "empty"),
        "coverage_rate": round(
            sum(1 for row in rows if row["status"] in {"native", "partial"}) / max(1, len(LEGACY_COMMANDS)),
            6,
        ),
        "top_gaps": top_infra_gaps(rows, layer_rows),
    }
    report = {
        "method": "retrieval_lab_infra_coverage_audit",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "command_coverage": rows,
        "layer_coverage": layer_rows,
        "recommendations": infra_recommendations(rows, layer_rows),
    }
    report["fingerprint"] = data_sha256({"commands": rows, "layers": layer_rows})
    output = Path(getattr(args, "output", DEFAULT_INFRA_AUDIT_REPORT_PATH))
    write_json(output, report)
    return {
        "method": "retrieval_lab_audit_infra_coverage",
        "output": str(output),
        "summary": {**summary, "output": str(output), "fingerprint": report["fingerprint"]},
    }


def command_coverage_rows() -> list[dict[str, Any]]:
    rows = []
    for command in sorted(LEGACY_COMMANDS):
        if command in NATIVE_COMMANDS:
            status = "native"
            note = "native command route exists"
        elif command in PARTIALLY_NATIVE_LEGACY_COMMANDS:
            status = "partial"
            note = PARTIALLY_NATIVE_LEGACY_COMMANDS[command]
        else:
            status = "compat_only"
            note = "available through mocktesting compatibility backend"
        rows.append({"command": command, "status": status, "note": note})
    native_only = sorted(NATIVE_COMMANDS - set(LEGACY_COMMANDS))
    for command in native_only:
        rows.append({"command": command, "status": "native_only", "note": "new Retrieval Lab native infra command"})
    return rows


def layer_coverage_rows(package_root: Path) -> list[dict[str, Any]]:
    rows = []
    for layer, expectation in sorted(LAYER_EXPECTATIONS.items()):
        path = package_root / layer
        py_files = [file for file in path.glob("*.py") if file.name != "__init__.py"] if path.exists() else []
        status = "missing"
        if path.exists() and py_files:
            status = "implemented"
        elif path.exists():
            status = "empty"
        rows.append(
            {
                "layer": layer,
                "status": status,
                "python_file_count": len(py_files),
                "expectation": expectation,
                "path": str(path),
            }
        )
    return rows


def top_infra_gaps(command_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> list[str]:
    gaps = []
    for layer in layer_rows:
        if layer["status"] in {"empty", "missing"}:
            gaps.append(f"{layer['layer']}: {layer['expectation']}")
    compat_only = [row["command"] for row in command_rows if row["status"] == "compat_only"]
    if compat_only:
        gaps.append("compat_only_commands: " + ", ".join(compat_only[:10]))
    return gaps[:12]


def infra_recommendations(command_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    empty_layers = {row["layer"] for row in layer_rows if row["status"] in {"empty", "missing"}}
    recommendations = []
    if "datasets" in empty_layers:
        recommendations.append(
            {
                "priority": "P0",
                "title": "Add dataset/split registry",
                "reason": "Hidden split, fixture lineage, and generated variants need first-class manifests.",
            }
        )
    if "planners" in empty_layers:
        recommendations.append(
            {
                "priority": "P0",
                "title": "Move query understanding registry into Retrieval Lab",
                "reason": "Planner comparison is still compatibility-backed, limiting fuzzy-query experimentation.",
            }
        )
    if "indexes" in empty_layers or "retrieval" in empty_layers:
        recommendations.append(
            {
                "priority": "P1",
                "title": "Add native index and retrieval run metadata",
                "reason": "Prepared index fingerprints and search workflows are still owned by mocktesting.",
            }
        )
    if any(row["command"] == "validate-rerank-gate" and row["status"] == "compat_only" for row in command_rows):
        recommendations.append(
            {
                "priority": "P1",
                "title": "Migrate rerank gate analysis",
                "reason": "Gated rerank is a core enterprise retrieval pattern and should operate on run artifacts.",
            }
        )
    return recommendations


__all__ = [
    "DEFAULT_INFRA_AUDIT_REPORT_PATH",
    "NATIVE_COMMANDS",
    "audit_infra_coverage_command",
    "command_coverage_rows",
    "layer_coverage_rows",
]
