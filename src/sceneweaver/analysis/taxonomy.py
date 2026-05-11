from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable

from sceneweaver.schemas.tags import TagEvidence, TagProfile

TAG_TAXONOMY_VERSION = "director_tags_v1"
TAG_DIMENSIONS = (
    "emotion_core",
    "audience_projection",
    "narrative_function",
    "interaction_mode",
    "visual_motifs",
    "symbolic_logic",
    "rhythm_pattern",
)
FINGERPRINT_TAXONOMY_VERSION = TAG_TAXONOMY_VERSION
FINGERPRINT_DIMENSIONS = TAG_DIMENSIONS
GENERAL_EXPRESSION_TAG = "general_expression"
TAXONOMY_PATH = Path(__file__).resolve().parents[3] / "taxonomy" / "director_tags_v1.json"


@dataclass(frozen=True)
class TaxonomyTag:
    tag: str
    aliases: tuple[str, ...]


FALLBACK_TAXONOMY: dict[str, tuple[TaxonomyTag, ...]] = {
    dimension: (TaxonomyTag(GENERAL_EXPRESSION_TAG, (GENERAL_EXPRESSION_TAG,)),)
    for dimension in TAG_DIMENSIONS
}


class TagNormalizer:
    def __init__(self, taxonomy: dict[str, tuple[TaxonomyTag, ...]] | None = None) -> None:
        self.taxonomy = taxonomy or load_taxonomy()

    def normalize_text(
        self,
        text: str,
        *,
        evidence: list[TagEvidence],
    ) -> TagProfile:
        tags = self.tags_from_text(text)
        return TagProfile(
            **tags,
            evidence=evidence,
            confidence=confidence_from_tags(tags),
        )

    def tags_from_text(self, text: str) -> dict[str, list[str]]:
        lowered = text.lower()
        tags: dict[str, list[str]] = {}
        for dimension in TAG_DIMENSIONS:
            dimension_tags: list[str] = []
            for entry in self.taxonomy[dimension]:
                if entry.tag == GENERAL_EXPRESSION_TAG:
                    continue
                if any(alias.lower() in lowered for alias in entry.aliases):
                    dimension_tags.append(entry.tag)
            tags[dimension] = _dedupe(dimension_tags)
        if not any(tags.values()):
            tags["symbolic_logic"] = [GENERAL_EXPRESSION_TAG]
        return tags

    def candidate_phrases_from_text(self, text: str, *, max_candidates: int = 12) -> list[str]:
        candidates: list[str] = []
        for phrase in split_candidate_phrases(text):
            phrase_lowered = phrase.lower()
            if any(alias.lower() in phrase_lowered for alias in self.all_aliases):
                continue
            if phrase not in candidates:
                candidates.append(phrase)
            if len(candidates) >= max_candidates:
                break
        return candidates

    @property
    def all_aliases(self) -> tuple[str, ...]:
        aliases: list[str] = []
        for entries in self.taxonomy.values():
            for entry in entries:
                aliases.append(entry.tag)
                aliases.extend(entry.aliases)
        return tuple(alias for alias in aliases if alias)


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict[str, tuple[TaxonomyTag, ...]]:
    if not path.exists():
        return FALLBACK_TAXONOMY
    data = json.loads(path.read_text(encoding="utf-8"))
    taxonomy: dict[str, list[TaxonomyTag]] = {dimension: [] for dimension in TAG_DIMENSIONS}
    for item in data.get("canonical_tags", []):
        dimension = item.get("dimension")
        tag = str(item.get("tag", "")).strip()
        if dimension not in taxonomy or not tag:
            continue
        aliases = [tag, str(item.get("label", "")).strip()]
        aliases.extend(str(alias).strip() for alias in item.get("aliases", []))
        taxonomy[dimension].append(
            TaxonomyTag(
                tag=normalize_candidate_tag(tag),
                aliases=tuple(alias for alias in aliases if alias),
            )
        )
    return {
        dimension: tuple(entries) if entries else FALLBACK_TAXONOMY[dimension]
        for dimension, entries in taxonomy.items()
    }


def confidence_from_tags(tags: dict[str, list[str]]) -> float:
    tag_count = sum(len(values) for values in tags.values() if GENERAL_EXPRESSION_TAG not in values)
    return min(0.95, max(0.35, 0.35 + tag_count * 0.05))


def canonical_tags(profile: TagProfile, dimension: str) -> set[str]:
    return {tag for tag in getattr(profile, dimension) if tag != GENERAL_EXPRESSION_TAG}


FingerprintNormalizer = TagNormalizer


def normalize_candidate_tag(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value.strip().lower())
    return re.sub(r"\W+", "_", normalized, flags=re.UNICODE).strip("_")


def split_candidate_phrases(text: str) -> list[str]:
    separators = r"[\s,.;:!?|/\\()\[\]{}<>\"'`~\u3001\u3002\uff0c\uff1b\uff1a\uff01\uff1f\uff08\uff09\u3010\u3011\u300a\u300b]+"
    result: list[str] = []
    for part in re.split(separators, text):
        clean = part.strip()
        if 2 <= len(clean) <= 32 and not clean.isdigit():
            result.append(clean)
    return result


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = normalize_candidate_tag(value)
        if normalized and normalized not in result:
            result.append(normalized)
    return result
