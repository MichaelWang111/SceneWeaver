from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mocktesting.eval_input_generator import INDUSTRY_WORDS, PURPOSE_WORDS, STAGE_WORDS

DEFAULT_REVIEW_ROOT = Path(__file__).resolve().parent / "retrieval_review"

DEFAULT_CHANNEL_WEIGHTS = {
    "script_use": 0.50,
    "experience": 0.25,
    "visual_tags": 0.15,
    "combined": 0.10,
}

CHANNEL_ALIASES = {
    "query_script_use": "script_use",
    "query_experience": "experience",
    "query_visual": "visual_tags",
    "query_combined": "combined",
}

NEGATIVE_RE = re.compile(r"(不要|避免|不想|不是|别|拒绝)([^。；，,]*)")
VISUAL_HINT_RE = re.compile(r"画面可以(?:有|借用)(.+?)这类")


def load_review_items(review_root: Path = DEFAULT_REVIEW_ROOT) -> list[dict[str, Any]]:
    manifest_path = review_root / "collection_manifest.json"
    if manifest_path.exists():
        fixture_ids = json.loads(manifest_path.read_text(encoding="utf-8"))["fixtures"]
    else:
        fixture_ids = sorted(path.name for path in review_root.iterdir() if path.is_dir())

    rows: list[dict[str, Any]] = []
    for fixture_id in fixture_ids:
        fixture_dir = review_root / fixture_id
        manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
        retrieval = json.loads((fixture_dir / "retrieval.json").read_text(encoding="utf-8"))
        knowledge = _read_optional_json(fixture_dir / "knowledge.json")
        knowledge_by_scene = {
            row.get("scene_id"): row
            for row in knowledge.get("scene_knowledge", [])
            if isinstance(row, dict)
        }
        for item in retrieval.get("items", []):
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "video_id": retrieval["video_id"],
                    "title": manifest["title"],
                    "industry": manifest["industry"],
                    "style": manifest["film_style"],
                    "company_profile": manifest.get("company_profile", ""),
                    "scene_id": item["scene_id"],
                    "retrieval_id": item["retrieval_id"],
                    "script_stage": item.get("script_stage", "general"),
                    "creative_purpose": item.get("creative_purpose", []),
                    "script_use_sentence": item.get("script_use_sentence", ""),
                    "llm_tags": item.get("llm_tags", {}),
                    "embedding_texts": item.get("embedding_texts", {}),
                    "knowledge": knowledge_by_scene.get(item.get("scene_id"), {}),
                }
            )
    return rows


def build_item_channels(
    item: dict[str, Any],
    *,
    extra_channels: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    stage = _stage_word(item.get("script_stage", ""))
    purposes = _purpose_text(item.get("creative_purpose", []))
    industry = _industry_word(item.get("industry", ""))
    tags = _flatten_tags(item.get("llm_tags", {}))
    visual = _clean_join(tags)
    embedding_texts = item.get("embedding_texts", {})
    knowledge_text = _knowledge_text(item)
    script_use = _clean_join(
        [
            f"脚本阶段：{stage}",
            f"脚本用途：{item.get('script_use_sentence', '')}",
            f"创作目的：{purposes}",
            f"行业语境：{industry}",
            embedding_texts.get("script_usage", ""),
        ]
    )
    experience = _clean_join(
        [
            f"导演经验：{embedding_texts.get('knowledge_semantic', '')}",
            knowledge_text,
            f"适合用途：{item.get('script_use_sentence', '')}",
        ]
    )
    visual_tags = _clean_join(
        [
            f"画面元素：{visual}",
            embedding_texts.get("visual_semantic", ""),
        ]
    )
    channels = [
        _channel("script_use", script_use),
        _channel("experience", experience),
        _channel("visual_tags", visual_tags),
        _channel("combined", _clean_join([script_use, experience, visual_tags])),
    ]
    if extra_channels:
        for extra in extra_channels:
            channels.append(
                {
                    "channel": str(extra["channel"]),
                    "text": str(extra.get("text", "")),
                    "weight": float(extra.get("weight", DEFAULT_CHANNEL_WEIGHTS.get(str(extra["channel"]), 0.0))),
                    "enabled": bool(extra.get("enabled", True)),
                }
            )
    return channels


def build_query_channels(
    user_input: str,
    *,
    extra_channels: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    clean = _clean_text(user_input)
    visual_hint = _extract_visual_hint(clean)
    negative_hint = _extract_negative_hint(clean)
    stage_hint = _extract_stage_hint(clean)
    script_use = _clean_join(
        [
            f"用户脚本需求：{clean}",
            f"可能脚本阶段：{stage_hint}",
            f"负面约束：{negative_hint}",
        ]
    )
    visual = _clean_join(
        [
            f"用户画面偏好：{visual_hint or clean}",
            f"负面画面约束：{negative_hint}",
        ]
    )
    experience = _clean_join(
        [
            f"用户导演经验需求：{clean}",
            f"希望避免：{negative_hint}",
        ]
    )
    channels = [
        _channel("query_script_use", script_use, target_channel="script_use"),
        _channel("query_experience", experience, target_channel="experience"),
        _channel("query_visual", visual, target_channel="visual_tags"),
        _channel("query_combined", clean, target_channel="combined"),
    ]
    if extra_channels:
        channels.extend(extra_channels)
    return channels


def build_item_channel_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "item_id": item_id(item),
                "metadata": item_metadata(item),
                "channels": build_item_channels(item),
            }
        )
    return rows


def item_id(item: dict[str, Any]) -> str:
    return f"{item['fixture_id']}::{item['scene_id']}::{item['retrieval_id']}"


def item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": item["fixture_id"],
        "video_id": item["video_id"],
        "scene_id": item["scene_id"],
        "retrieval_id": item["retrieval_id"],
        "title": item["title"],
        "industry": item["industry"],
        "style": item["style"],
        "script_stage": item.get("script_stage", "general"),
        "creative_purpose": item.get("creative_purpose", []),
        "script_use_sentence": item.get("script_use_sentence", ""),
    }


def target_item_id(target: dict[str, Any]) -> str:
    return f"{target['fixture_id']}::{target['scene_id']}::{target['retrieval_id']}"


def target_channel_for_query(channel: str) -> str:
    return CHANNEL_ALIASES.get(channel, channel)


def _channel(
    name: str,
    text: str,
    *,
    target_channel: str | None = None,
) -> dict[str, Any]:
    row = {
        "channel": name,
        "text": _clean_text(text),
        "weight": DEFAULT_CHANNEL_WEIGHTS.get(target_channel or name, 0.0),
        "enabled": True,
    }
    if target_channel is not None:
        row["target_channel"] = target_channel
    return row


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_tags(tags: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("entities", "relations", "actions_and_expressions", "background_and_setting", "social_relations"):
        for value in tags.get(key, []):
            if isinstance(value, str) and value not in values:
                values.append(value)
    return values


def _knowledge_text(item: dict[str, Any]) -> str:
    knowledge = item.get("knowledge", {})
    parts: list[str] = []
    for key in ("director_intent", "emotional_function", "reusable_knowledge", "best_usage", "risk"):
        value = knowledge.get(key)
        if isinstance(value, str):
            parts.append(value)
    techniques = knowledge.get("shooting_techniques", [])
    if isinstance(techniques, list):
        parts.extend(str(value) for value in techniques if isinstance(value, str))
    return _clean_join(parts)


def _stage_word(stage: str) -> str:
    return STAGE_WORDS.get(stage, stage.replace("_", " ") or "通用")


def _industry_word(industry: str) -> str:
    return INDUSTRY_WORDS.get(industry, industry.replace("_", " ") or "企业")


def _purpose_text(purposes: list[str]) -> str:
    words = [PURPOSE_WORDS.get(purpose, purpose.replace("_", " ")) for purpose in purposes]
    return "、".join(words)


def _extract_visual_hint(text: str) -> str:
    match = VISUAL_HINT_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_negative_hint(text: str) -> str:
    matches = ["".join(match).strip() for match in NEGATIVE_RE.findall(text)]
    return "；".join(matches)


def _extract_stage_hint(text: str) -> str:
    hits = [word for word in STAGE_WORDS.values() if word and word in text]
    return "、".join(hits)


def _clean_join(parts: list[str]) -> str:
    return _clean_text("。".join(part for part in parts if str(part).strip()))


def _clean_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    clean = clean.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    clean = clean.replace('"', "").replace("'", "")
    return clean
