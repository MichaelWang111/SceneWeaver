from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_lab.indexes.service import (
    DEFAULT_INDEX_MANIFEST_PATH,
    build_index_manifest,
    index_manifest_summary,
    write_index_manifest,
)


def index_manifest_command(args: Any) -> dict[str, Any]:
    manifest = build_index_manifest(
        dataset_path=Path(getattr(args, "dataset")),
        split=str(getattr(args, "split", "test")),
        limit=int(getattr(args, "limit", 0)),
        index_id=str(getattr(args, "index_id", "") or ""),
    )
    output = Path(getattr(args, "output", DEFAULT_INDEX_MANIFEST_PATH))
    write_index_manifest(output, manifest)
    return {
        "method": "retrieval_lab_index_manifest",
        "output": str(output),
        "summary": {**index_manifest_summary(manifest), "output": str(output)},
    }


def index_inspect_command(args: Any) -> dict[str, Any]:
    return index_manifest_command(args)


__all__ = ["index_inspect_command", "index_manifest_command"]
