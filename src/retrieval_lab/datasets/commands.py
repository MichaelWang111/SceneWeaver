from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, write_json
from retrieval_lab.datasets.service import DEFAULT_DATASET_PATH, dataset_manifest


DEFAULT_DATASET_MANIFEST_PATH = Path(".tmp") / "retrieval_lab" / "dataset_manifest_latest.json"


def inspect_dataset_command(args: Any) -> dict[str, Any]:
    started_at = time.perf_counter()
    manifest = dataset_manifest(
        Path(getattr(args, "input", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "all")),
        limit=int(getattr(args, "limit", 0)),
    )
    manifest["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
    manifest["fingerprint"] = data_sha256(
        {
            "source_sha256": manifest.get("source_sha256"),
            "split": manifest.get("split"),
            "limit": manifest.get("limit"),
            "summary": manifest.get("summary"),
        }
    )
    output = Path(getattr(args, "output", DEFAULT_DATASET_MANIFEST_PATH))
    write_json(output, manifest)
    return {
        "method": "retrieval_lab_inspect_dataset",
        "output": str(output),
        "summary": {**manifest["summary"], "output": str(output), "fingerprint": manifest["fingerprint"]},
    }


__all__ = ["DEFAULT_DATASET_MANIFEST_PATH", "inspect_dataset_command"]
