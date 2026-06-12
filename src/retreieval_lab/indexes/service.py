from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, file_sha256, write_json
from retreieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retreieval_lab.schemas import IndexManifestModel


DEFAULT_INDEX_MANIFEST_PATH = Path(".tmp") / "retrieval_lab" / "index_manifest_latest.json"
INDEX_CHANNELS = ["lexical", "metadata", "constraints", "scene_signature"]


def build_index_manifest(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    split: str = "test",
    limit: int = 0,
    index_id: str = "",
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    cases = read_cases(dataset_path, split=split, limit=limit)
    items = index_items_from_cases(cases)
    manifest = {
        "index_id": index_id or f"dataset::{dataset_path.stem}::{split}::{limit or 'all'}",
        "source_dataset_id": dataset_path.stem,
        "item_count": len(items),
        "channels": INDEX_CHANNELS,
        "embedding_model": "",
        "lexical_tokenizer": "retrieval_lab_cjk_bigram_ascii_v1",
        "fingerprint": data_sha256({"dataset": str(dataset_path), "source_sha256": file_sha256(dataset_path), "items": items}),
        "cache_paths": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_path": str(dataset_path),
        "split": split,
        "limit": limit,
        "stage_counts": dict(sorted(Counter(str(item["metadata"].get("script_stage", "")) for item in items).items())),
        "fixture_counts": dict(sorted(Counter(str(item["metadata"].get("fixture_id", "")) for item in items).items())),
    }
    return IndexManifestModel.model_validate(manifest).model_dump(mode="json", exclude_none=True)


def index_items_from_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        for key in ("target", "expected_prefer"):
            target = case.get(key)
            if not isinstance(target, dict):
                continue
            item = item_from_target(target, case)
            if item["item_id"] not in by_id:
                by_id[item["item_id"]] = item
            else:
                by_id[item["item_id"]]["source_case_ids"].append(str(case.get("case_id", "")))
                by_id[item["item_id"]]["text"] = merge_texts(by_id[item["item_id"]]["text"], item["text"])
    return list(sorted(by_id.values(), key=lambda row: row["item_id"]))


def item_from_target(target: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    item_id = target_item_id(target)
    metadata = {
        "fixture_id": target.get("fixture_id", ""),
        "video_id": target.get("video_id", ""),
        "scene_id": target.get("scene_id", ""),
        "retrieval_id": target.get("retrieval_id", ""),
        "script_stage": target.get("script_stage", ""),
        "creative_purpose": list(target.get("creative_purpose", []) or []),
        "title": target.get("title", ""),
        "industry": target.get("industry", ""),
        "style": target.get("style", ""),
    }
    text = item_text(target, case)
    return {
        "item_id": item_id,
        "metadata": metadata,
        "text": text,
        "tokens": lexical_tokens(text),
        "source_case_ids": [str(case.get("case_id", ""))],
    }


def item_text(target: dict[str, Any], case: dict[str, Any]) -> str:
    embedding_texts = case.get("target_embedding_texts", {}) if isinstance(case.get("target_embedding_texts"), dict) else {}
    parts = [
        target.get("title", ""),
        target.get("script_stage", ""),
        " ".join(str(value) for value in target.get("creative_purpose", []) or []),
        target.get("industry", ""),
        target.get("style", ""),
        case.get("target_summary", ""),
        case.get("target_tags_text", ""),
        *[str(value) for value in embedding_texts.values()],
    ]
    return " ".join(part for part in parts if part)


def target_item_id(target: dict[str, Any]) -> str:
    fixture_id = str(target.get("fixture_id", ""))
    scene_id = str(target.get("scene_id", ""))
    retrieval_id = str(target.get("retrieval_id", ""))
    return "::".join(part for part in (fixture_id, scene_id, retrieval_id) if part)


def lexical_tokens(text: str) -> list[str]:
    lower = str(text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", lower)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lower)
    cjk_bigrams = ["".join(pair) for pair in zip(cjk_chars, cjk_chars[1:], strict=False)]
    return sorted(set([token for token in ascii_tokens if len(token) > 1] + cjk_bigrams + cjk_chars))


def merge_texts(left: str, right: str) -> str:
    if right in left:
        return left
    return f"{left} {right}".strip()


def write_index_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, {"method": "retrieval_lab_index_manifest", "summary": index_manifest_summary(manifest), "manifest": manifest})


def index_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "index_id": manifest.get("index_id", ""),
        "item_count": manifest.get("item_count", 0),
        "channels": manifest.get("channels", []),
        "fingerprint": manifest.get("fingerprint", ""),
        "split": manifest.get("split", ""),
        "limit": manifest.get("limit", 0),
    }


__all__ = [
    "DEFAULT_INDEX_MANIFEST_PATH",
    "INDEX_CHANNELS",
    "build_index_manifest",
    "index_items_from_cases",
    "index_manifest_summary",
    "lexical_tokens",
    "target_item_id",
    "write_index_manifest",
]
