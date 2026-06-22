from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from retrieval_lab.indexes.service import lexical_tokens, merge_unique
from sceneweaver.retrieval.style import infer_card_style_risks, infer_card_style_traits
from sceneweaver.retrieval.usecase import build_script_usecase
from sceneweaver.schemas import ExperienceCard, ScriptUseCase, TagProfile

SCENEWEAVER_CHANNEL_WEIGHTS: dict[str, float] = {
    "summary": 0.35,
    "script_use": 0.45,
    "experience": 0.30,
    "visual_tags": 0.20,
    "tags": 0.15,
    "combined": 1.00,
}

SCENEWEAVER_CHANNEL_POLICIES: dict[str, tuple[str, ...]] = {
    "combined": ("combined",),
    "summary": ("summary",),
    "tags": ("tags",),
    "script_use": ("script_use",),
    "experience": ("experience",),
    "visual_tags": ("visual_tags",),
    "summary_tags": ("summary", "tags"),
    "script_experience": ("script_use", "experience"),
    "all": ("summary", "script_use", "experience", "visual_tags", "tags"),
}


def sceneweaver_items_from_sources(
    sources: Iterable[Path],
    *,
    channel_policy: str = "combined",
) -> list[dict[str, Any]]:
    """Read SceneWeaver outputs and expose them as Retrieval Lab index items."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        for cards_path in resolve_card_paths(Path(source)):
            for raw in read_jsonl(cards_path):
                item = item_from_card_payload(raw, cards_path=cards_path, channel_policy=channel_policy)
                if item["item_id"] in seen:
                    continue
                seen.add(item["item_id"])
                items.append(item)
    return sorted(items, key=lambda row: row["item_id"])


def resolve_card_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    direct = source / "analysis" / "experience_cards.jsonl"
    if direct.exists():
        return [direct]
    return sorted(source.glob("**/analysis/experience_cards.jsonl"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def item_from_card_payload(
    payload: dict[str, Any],
    *,
    cards_path: Path,
    channel_policy: str = "combined",
) -> dict[str, Any]:
    card = normalize_card(payload)
    channels = build_card_channels(card)
    text = channel_text(channels, channel_policy)
    metadata = card_metadata(card, cards_path=cards_path, payload=payload)
    return {
        "item_id": item_id_for_card(card),
        "metadata": metadata,
        "text": text,
        "tokens": lexical_tokens(text),
        "channels": channels,
        "channel_policy": channel_policy,
        "payload": card_payload(card),
        "payload_ref": f"{cards_path}#{card.card_id}",
        "source_case_ids": [],
    }


def normalize_card(payload: dict[str, Any]) -> ExperienceCard:
    data = dict(payload)
    if "tags" not in data and "fingerprint" in data:
        data["tags"] = data.pop("fingerprint")
    if "script_usecase" not in data:
        tags = TagProfile.model_validate(data["tags"])
        temp = float(data.get("confidence", tags.confidence) or tags.confidence)
        usecase = build_script_usecase(tags, text=card_source_text(data), base_confidence=temp)
        data["script_usecase"] = usecase.model_dump(mode="json")
    return ExperienceCard.model_validate(data)


def build_card_channels(card: ExperienceCard) -> dict[str, str]:
    tags_text = tag_text(card.tags)
    summary = clean_join([card.script_usecase.best_usage, card.reuse_condition])
    script_use = clean_join(
        [
            card.script_usecase.script_stage,
            *card.script_usecase.creative_purpose,
            card.script_usecase.best_usage,
            card.script_usecase.risk,
            card.reuse_condition,
        ]
    )
    experience = clean_join(
        [
            card.director_strategy,
            card.narrative_logic,
            card.underlying_emotion,
            *card.shooting_techniques,
            card.copywriting_tone,
        ]
    )
    visual_tags = clean_join([*card.keywords, *card.visual_symbols, tags_text])
    combined = clean_join([summary, script_use, experience, visual_tags, tags_text])
    return {
        "summary": summary,
        "script_use": script_use,
        "experience": experience,
        "visual_tags": visual_tags,
        "tags": tags_text,
        "combined": combined,
    }


def channel_text(channels: dict[str, str], channel_policy: str) -> str:
    try:
        names = SCENEWEAVER_CHANNEL_POLICIES[channel_policy]
    except KeyError as exc:
        allowed = ", ".join(sorted(SCENEWEAVER_CHANNEL_POLICIES))
        raise ValueError(f"unknown SceneWeaver channel policy: {channel_policy}; expected one of {allowed}") from exc
    return clean_join([channels.get(name, "") for name in names])


def card_metadata(card: ExperienceCard, *, cards_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    usecase = card.script_usecase
    scene_id = card.source_scene_ids[0]
    style_traits = infer_card_style_traits(card)
    style_risks = infer_card_style_risks(card)
    return {
        "corpus_source": "sceneweaver_experience_cards",
        "source_path": str(cards_path),
        "video_id": card.source_video_id,
        "source_video_id": card.source_video_id,
        "scene_id": scene_id,
        "source_scene_ids": list(card.source_scene_ids),
        "card_id": card.card_id,
        "retrieval_id": card.card_id,
        "script_stage": usecase.script_stage,
        "creative_purpose": list(usecase.creative_purpose),
        "script_usecase_best_usage": usecase.best_usage,
        "script_usecase_risk": usecase.risk,
        "script_usecase_confidence": usecase.confidence,
        "style_traits": style_traits,
        "style_risks": style_risks,
        "confidence": card.confidence,
        "tag_confidence": card.tags.confidence,
        "emotion_temperature_range": list(card.emotion_temperature_range),
        "title": payload.get("title", card.card_id),
        "industry": payload.get("industry", ""),
        "style": payload.get("style", ""),
    }


def card_payload(card: ExperienceCard) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "source_video_id": card.source_video_id,
        "source_scene_ids": list(card.source_scene_ids),
        "keywords": list(card.keywords),
        "underlying_emotion": card.underlying_emotion,
        "narrative_logic": card.narrative_logic,
        "director_strategy": card.director_strategy,
        "shooting_techniques": list(card.shooting_techniques),
        "visual_symbols": list(card.visual_symbols),
        "copywriting_tone": card.copywriting_tone,
        "avoid": list(card.avoid),
        "reuse_condition": card.reuse_condition,
        "script_usecase": card.script_usecase.model_dump(mode="json"),
        "tags": card.tags.model_dump(mode="json"),
    }


def item_id_for_card(card: ExperienceCard) -> str:
    return f"{card.source_video_id}::{card.source_scene_ids[0]}::{card.card_id}"


def tag_text(tags: TagProfile) -> str:
    values: list[str] = []
    for key, value in tags.model_dump(mode="json").items():
        if key == "evidence":
            continue
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
    return clean_join(merge_unique(values))


def card_source_text(payload: dict[str, Any]) -> str:
    parts = [
        *string_list(payload.get("keywords")),
        str(payload.get("underlying_emotion", "")),
        str(payload.get("narrative_logic", "")),
        str(payload.get("director_strategy", "")),
        *string_list(payload.get("shooting_techniques")),
        *string_list(payload.get("visual_symbols")),
        str(payload.get("copywriting_tone", "")),
        str(payload.get("reuse_condition", "")),
    ]
    return clean_join(parts)


def string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def clean_join(parts: Iterable[str]) -> str:
    seen: list[str] = []
    for part in parts:
        text = " ".join(str(part or "").split())
        if text and text not in seen:
            seen.append(text)
    return " ".join(seen)


__all__ = [
    "SCENEWEAVER_CHANNEL_POLICIES",
    "SCENEWEAVER_CHANNEL_WEIGHTS",
    "build_card_channels",
    "item_from_card_payload",
    "sceneweaver_items_from_sources",
]
