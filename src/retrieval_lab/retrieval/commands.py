from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_lab.artifacts import write_json
from retrieval_lab.retrieval.service import (
    DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT,
    DEFAULT_RETRIEVAL_RUN_OUTPUT,
    compare_run_artifacts,
    retrieval_run,
    write_retrieval_run,
)


def retrieval_run_command(args: Any) -> dict[str, Any]:
    artifact = retrieval_run(
        dataset_path=Path(getattr(args, "dataset")),
        split=str(getattr(args, "split", "test.md")),
        limit=int(getattr(args, "limit", 0)),
        planner=str(getattr(args, "planner", "multi_query")),
        planner_cache=getattr(args, "planner_cache", None),
        top_k=int(getattr(args, "top_k", 10)),
        candidate_depth=int(getattr(args, "candidate_depth", 100)),
        run_name=str(getattr(args, "run_name", "") or ""),
        ranking_key=str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature") or "hybrid_rrf_constraints_signature"),
        card_sources=[Path(path) for path in getattr(args, "cards", []) or []] or None,
        queries=list(getattr(args, "query", []) or []),
        query_file=getattr(args, "query_file", None),
        channel_policy=str(getattr(args, "channel_policy", "combined") or "combined"),
    )
    output = Path(getattr(args, "output", DEFAULT_RETRIEVAL_RUN_OUTPUT))
    write_retrieval_run(output, artifact)
    return {
        "method": "retrieval_lab_retrieval_run",
        "output": str(output),
        "summary": {**artifact["summary"], "output": str(output), "fingerprint": artifact["fingerprint"]},
    }


def retrieval_compare_legacy_command(args: Any) -> dict[str, Any]:
    report = compare_run_artifacts(Path(getattr(args, "native")), Path(getattr(args, "legacy")))
    output = Path(getattr(args, "output", DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT))
    write_json(output, report)
    return {
        "method": "retrieval_lab_compare_legacy_retrieval",
        "output": str(output),
        "summary": {**report["summary"], "output": str(output)},
    }


__all__ = ["retrieval_compare_legacy_command", "retrieval_run_command"]
