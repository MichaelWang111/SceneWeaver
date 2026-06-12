from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from retreieval_lab.artifacts import ArtifactRef, artifact_manifest, write_json
from retreieval_lab.experiments.runs import DEFAULT_RUN_ARTIFACT_PATH, export_run_artifact_command


DEFAULT_LEGACY_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "legacy_report_latest.json"
DEFAULT_LEGACY_MANIFEST_PATH = Path(".tmp") / "retrieval_lab" / "legacy_run_manifest_latest.json"


def run_legacy_with_artifacts_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    command = str(getattr(args, "legacy_command", "") or getattr(args, "command_name", ""))
    if not command:
        raise ValueError("--legacy-command is required")
    report_output = Path(getattr(args, "report_output", DEFAULT_LEGACY_REPORT_PATH))
    run_output = Path(getattr(args, "run_output", DEFAULT_RUN_ARTIFACT_PATH))
    manifest_output = Path(getattr(args, "manifest_output", DEFAULT_LEGACY_MANIFEST_PATH))
    legacy_args = normalize_remainder(list(getattr(args, "legacy_args", []) or []))
    legacy_args = with_output_option(legacy_args, report_output)
    invocation = [sys.executable, "-m", "mocktesting.mock_retriever", command, *legacy_args]
    result = subprocess.run(invocation, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "legacy command failed with exit code "
            f"{result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    export_result = export_run_artifact_command(
        SimpleArgs(
            reports=[report_output],
            output=run_output,
            run_name=str(getattr(args, "run_name", "")),
        )
    )
    manifest = artifact_manifest(
        [
            ArtifactRef(report_output, role="legacy_report"),
            ArtifactRef(run_output, role="run_artifact"),
        ],
        manifest_id=str(getattr(args, "manifest_id", "") or f"legacy_run_{time.strftime('%Y%m%d_%H%M%S')}"),
        label=str(getattr(args, "label", "") or command),
        metadata={
            "legacy_command": command,
            "legacy_args": legacy_args,
            "returncode": result.returncode,
            "stdout_preview": result.stdout.strip()[:1000],
        },
    )
    write_json(manifest_output, manifest)
    elapsed_seconds = round(time.perf_counter() - started_at, 3)
    summary = {
        "legacy_command": command,
        "report_output": str(report_output),
        "run_output": str(run_output),
        "manifest_output": str(manifest_output),
        "run_count": export_result["summary"].get("run_count", 0),
        "row_count": export_result["summary"].get("row_count", 0),
        "skipped_report_count": export_result["summary"].get("skipped_report_count", 0),
        "legacy_returncode": result.returncode,
        "elapsed_seconds": elapsed_seconds,
    }
    return {
        "method": "retrieval_lab_legacy_run_bridge",
        "legacy_invocation": invocation,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "run_artifact": export_result,
        "manifest": manifest,
        "summary": summary,
    }


def normalize_remainder(values: list[str]) -> list[str]:
    if values and values[0] == "--":
        return values[1:]
    return values


def with_output_option(args: list[str], output: Path) -> list[str]:
    result = []
    index = 0
    replaced = False
    while index < len(args):
        value = args[index]
        if value == "--output":
            result.extend(["--output", str(output)])
            index += 2
            replaced = True
            continue
        if value.startswith("--output="):
            result.append(f"--output={output}")
            index += 1
            replaced = True
            continue
        result.append(value)
        index += 1
    if not replaced:
        result.extend(["--output", str(output)])
    return result


class SimpleArgs:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


__all__ = [
    "DEFAULT_LEGACY_MANIFEST_PATH",
    "DEFAULT_LEGACY_REPORT_PATH",
    "run_legacy_with_artifacts_command",
    "with_output_option",
]
