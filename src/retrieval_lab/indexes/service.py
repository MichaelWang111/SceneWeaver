from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, file_sha256, write_json
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retrieval_lab.schemas import IndexManifestModel


DEFAULT_INDEX_MANIFEST_PATH = Path(".tmp") / "retrieval_lab" / "index_manifest_latest.json"
INDEX_CHANNELS = ["lexical", "metadata", "constraints", "scene_signature"]
STYLE_RISK_ALIASES = {
    "big_company_office": ("大厂味", "大厂", "互联网大厂", "generic office", "office"),
    "ad_like": ("广告感", "广告", "宣传片腔", "宣传片", "口号", "ad-like", "slogan", "campaign", "advertising"),
    "tech_showoff": ("炫技", "技术炫耀", "技术堆砌", "功能展示", "technology showcase"),
    "product_pitch": ("产品卖点", "卖点", "卖点堆叠", "sales pitch", "product pitch"),
    "corporate_report_tone": ("汇报片", "企业汇报", "corporate report"),
    "slogan_driven": ("口号感", "口号驱动", "slogan driven", "slogan-heavy", "tagline"),
    "generic_brand_film": ("品牌片", "品牌质感", "generic brand film", "brand film"),
    "fortune_500_polish": ("世界500强", "五百强", "fortune 500", "fortune-500", "polished corporate"),
    "tech_coldness": ("科技冷感", "冰冷科技", "tech coldness", "cold technology"),
}
STYLE_TRAIT_ALIASES = {
    "human_warmth": ("有人味", "人的温度", "human", "warmth"),
    "documentary": ("纪录片", "纪实", "documentary", "observational"),
    "real_location": ("真实现场", "现场感", "real location", "on location"),
}


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
        "style_traits": list(target.get("style_traits", []) or []),
        "style_risks": list(target.get("style_risks", []) or []),
    }
    text = item_text(target, case)
    metadata["style_traits"] = merge_unique([*metadata["style_traits"], *infer_style_hits(text, STYLE_TRAIT_ALIASES)])
    metadata["style_risks"] = merge_unique([*metadata["style_risks"], *infer_style_hits(text, STYLE_RISK_ALIASES)])
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


def infer_style_hits(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    lower = str(text or "").lower()
    return [name for name, terms in aliases.items() if any(term.lower() in lower for term in terms)]


def merge_unique(values: list[Any]) -> list[Any]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


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
    "infer_style_hits",
    "target_item_id",
    "write_index_manifest",
]
