from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from retreieval_lab.artifacts.io import ArtifactRef, artifact_manifest, write_json


DEFAULT_ARTIFACT_MANIFEST_PATH = Path(".tmp") / "retrieval_lab" / "artifact_manifest_latest.json"


def write_artifact_manifest_command(args: Any) -> dict[str, Any]:
    inputs = [
        ArtifactRef(Path(path), role="input")
        for path in list(getattr(args, "inputs", None) or [])
    ]
    outputs = [
        ArtifactRef(Path(path), role="output")
        for path in list(getattr(args, "outputs", None) or [])
    ]
    manifest = artifact_manifest(
        [*inputs, *outputs],
        manifest_id=getattr(args, "manifest_id", "") or f"manifest_{time.strftime('%Y%m%d_%H%M%S')}",
        label=getattr(args, "label", ""),
        metadata={
            "command": getattr(args, "command", "write-artifact-manifest"),
            "input_count": len(inputs),
            "output_count": len(outputs),
        },
    )
    output = Path(getattr(args, "output", DEFAULT_ARTIFACT_MANIFEST_PATH))
    write_json(output, manifest)
    return {
        "method": "retrieval_lab_write_artifact_manifest",
        "output": str(output),
        "summary": {
            "manifest_id": manifest["manifest_id"],
            "artifact_count": manifest["artifact_count"],
            "missing_count": manifest["missing_count"],
            "fingerprint": manifest["fingerprint"],
            "output": str(output),
        },
    }


__all__ = ["DEFAULT_ARTIFACT_MANIFEST_PATH", "write_artifact_manifest_command"]
