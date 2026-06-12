from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    role: str = ""
    kind: str = ""


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def data_sha256(data: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(data)).hexdigest()


def file_sha256(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(ref: ArtifactRef | Path | str, *, role: str = "", kind: str = "") -> dict[str, Any]:
    if isinstance(ref, ArtifactRef):
        path = ref.path
        role = ref.role or role
        kind = ref.kind or kind
    else:
        path = Path(ref)
    exists = path.exists()
    record: dict[str, Any] = {
        "path": str(path),
        "role": role,
        "kind": kind or infer_artifact_kind(path),
        "exists": exists,
    }
    if exists:
        stat = path.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "sha256": file_sha256(path),
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
            }
        )
    return record


def infer_artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".svg":
        return "svg"
    return suffix.lstrip(".") or "unknown"


def artifact_manifest(
    artifacts: Iterable[ArtifactRef | Path | str],
    *,
    manifest_id: str,
    label: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_rows = [artifact_record(ref) for ref in artifacts]
    return {
        "method": "retrieval_lab_artifact_manifest",
        "manifest_id": manifest_id,
        "label": label,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_count": len(artifact_rows),
        "missing_count": sum(1 for row in artifact_rows if not row["exists"]),
        "artifacts": artifact_rows,
        "metadata": metadata or {},
        "fingerprint": data_sha256(
            [
                {
                    "path": row["path"],
                    "role": row["role"],
                    "kind": row["kind"],
                    "exists": row["exists"],
                    "sha256": row.get("sha256"),
                }
                for row in artifact_rows
            ]
        ),
    }


__all__ = [
    "ArtifactRef",
    "artifact_manifest",
    "artifact_record",
    "canonical_json_bytes",
    "data_sha256",
    "file_sha256",
    "infer_artifact_kind",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]
