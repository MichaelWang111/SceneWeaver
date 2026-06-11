from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable

from pydantic import Field

from sceneweaver.analysis.taxonomy import TAG_DIMENSIONS, TagNormalizer, canonical_tags, normalize_candidate_tag
from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.schemas.experience_card import ExperienceCard
from sceneweaver.schemas.scene_analysis import SceneAnalysis, ScenesAnalysis
from sceneweaver.schemas.tags import TagCandidate, TagEvidence, TagProfile
from sceneweaver.storage.json_store import read_jsonl, write_json

LogFn = Callable[[str], None]
_CANDIDATE_LOG_LOCK = Lock()


@dataclass(frozen=True)
class TagMatch:
    score: float
    matched_dimensions: dict[str, list[str]]
    card: ExperienceCard


class ExperienceCardMatch(StrictBaseModel):
    card_id: str
    score: float = Field(ge=0)
    tag_score: float = Field(default=0, ge=0)
    usecase_score: float = Field(default=0, ge=0)
    intent_score: float = Field(default=0)
    constraint_score: float = Field(default=0)
    constraint_hits: dict[str, list[str]] = Field(default_factory=dict)
    quality_score: float = Field(default=0, ge=0)
    semantic_score: float | None = Field(default=None, ge=-1, le=1)
    lexical_score: float | None = Field(default=None, ge=0)
    rrf_score: float = Field(default=0, ge=0)
    ranking_workflow: str = "semantic_constraints"
    matched_dimensions: dict[str, list[str]]
    matched_usecase: dict[str, list[str]] = Field(default_factory=dict)
    script_stage: str = "general"
    creative_purpose: list[str] = Field(default_factory=list)
    best_usage: str = ""
    risk: str = ""
    evidence: list[TagEvidence]
    card: ExperienceCard


class RetrievalResult(StrictBaseModel):
    query_tags: TagProfile
    results: list[ExperienceCardMatch]

    @property
    def query_fingerprint(self) -> TagProfile:
        return self.query_tags


WEIGHTS = {
    "audience_projection": 2.0,
    "interaction_mode": 2.0,
    "emotion_core": 1.5,
    "narrative_function": 1.25,
    "symbolic_logic": 1.25,
    "visual_motifs": 1.0,
    "rhythm_pattern": 1.0,
}


def generate_analysis_tags(
    output_dir: Path,
    *,
    force: bool = False,
    log: LogFn | None = None,
) -> ScenesAnalysis:
    output_dir = output_dir.resolve()
    scenes_path = output_dir / "analysis" / "scenes.json"
    data = json.loads(scenes_path.read_text(encoding="utf-8"))
    scenes_data = data.get("scenes", [])
    candidate_log_path = output_dir / "analysis" / "tag_candidates.jsonl"
    for scene_data in scenes_data:
        if force or "tags" not in scene_data:
            tag_profile, candidates = build_scene_tags_and_candidates_from_payload(scene_data)
            scene_data["tags"] = tag_profile.model_dump(mode="json")
            append_tag_candidates(candidate_log_path, candidates)
            if log:
                log(f"Tags ready: {scene_data.get('scene_id', 'unknown_scene')}")
        scene = SceneAnalysis.model_validate(scene_data)
        write_json(output_dir / "analysis" / f"{scene.scene_id}.json", scene)
    data["scenes"] = scenes_data
    scenes = ScenesAnalysis.model_validate(data)
    write_json(scenes_path, scenes)
    return scenes


def read_scene_analysis_with_tags(path: Path, *, write_back: bool = False) -> SceneAnalysis:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tags" not in data:
        tag_profile, candidates = build_scene_tags_and_candidates_from_payload(data)
        data["tags"] = tag_profile.model_dump(mode="json")
        if write_back:
            append_tag_candidates(path.parent / "tag_candidates.jsonl", candidates)
    scene = SceneAnalysis.model_validate(data)
    if write_back:
        write_json(path, scene)
    return scene


def read_scenes_analysis_with_tags(path: Path, *, write_back: bool = False) -> ScenesAnalysis:
    data = json.loads(path.read_text(encoding="utf-8"))
    for scene_data in data.get("scenes", []):
        if "tags" not in scene_data:
            tag_profile, candidates = build_scene_tags_and_candidates_from_payload(scene_data)
            scene_data["tags"] = tag_profile.model_dump(mode="json")
            if write_back:
                append_tag_candidates(path.parent / "tag_candidates.jsonl", candidates)
    scenes = ScenesAnalysis.model_validate(data)
    if write_back:
        write_json(path, scenes)
    return scenes


def add_tags_to_scene_raw(raw: dict, *, candidate_log_path: Path | None = None) -> dict:
    data = dict(raw)
    tag_profile, candidates = build_scene_tags_and_candidates_from_payload(data)
    data["tags"] = tag_profile.model_dump(mode="json")
    if candidate_log_path is not None:
        append_tag_candidates(candidate_log_path, candidates)
    return data


def build_scene_tags(scene: SceneAnalysis) -> TagProfile:
    return TagNormalizer().normalize_text(_scene_text(scene), evidence=_scene_evidence(scene))


def build_scene_tags_from_payload(data: dict) -> TagProfile:
    tags, _ = build_scene_tags_and_candidates_from_payload(data)
    return tags


def build_scene_tags_and_candidates_from_payload(data: dict) -> tuple[TagProfile, list[TagCandidate]]:
    scene_id = str(data.get("scene_id", "scene_000"))
    interpretation = data.get("director_interpretation") or {}
    observation = data.get("visual_observation") or {}
    candidates = data.get("experience_candidates") or []
    text_parts = [
        observation.get("setting", ""),
        observation.get("characters", ""),
        observation.get("action_change", ""),
        observation.get("composition", ""),
        observation.get("lighting", ""),
        observation.get("color", ""),
        observation.get("camera_motion", ""),
        interpretation.get("narrative_function", ""),
        interpretation.get("emotional_function", ""),
        interpretation.get("brand_personality_signal", ""),
        interpretation.get("underlying_emotion", ""),
        interpretation.get("audience_projection", ""),
        " ".join(interpretation.get("shooting_techniques") or []),
        interpretation.get("why_it_works", ""),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            text_parts.extend(
                [
                    " ".join(candidate.get("keywords") or []),
                    candidate.get("emotion", ""),
                    candidate.get("narrative_logic", ""),
                    " ".join(candidate.get("techniques") or []),
                    candidate.get("reuse_condition", ""),
                ]
            )
    evidence = [
        TagEvidence(
            source_id=scene_id,
            source_type="scene",
            field="director_interpretation.underlying_emotion",
            quote=_clip(str(interpretation.get("underlying_emotion", "") or "No underlying emotion provided.")),
        ),
        TagEvidence(
            source_id=scene_id,
            source_type="scene",
            field="director_interpretation.audience_projection",
            quote=_clip(str(interpretation.get("audience_projection", "") or "No audience projection provided.")),
        ),
    ]
    text = " ".join(text_parts)
    normalizer = TagNormalizer()
    tag_profile = normalizer.normalize_text(text, evidence=evidence)
    return tag_profile, build_tag_candidates(
        normalizer.candidate_phrases_from_text(text),
        source_id=scene_id,
        source_type="scene",
        field="analysis_text",
        quote=_clip(text, max_chars=220),
    )


def build_film_tags(scenes: Iterable[SceneAnalysis]) -> TagProfile:
    scene_list = list(scenes)
    merged = _merge_tags(scene.tags for scene in scene_list)
    evidence: list[TagEvidence] = []
    for scene in scene_list:
        evidence.extend(scene.tags.evidence[:2])
    if not evidence:
        evidence = [
            TagEvidence(
                source_id="film",
                source_type="film",
                field="scenes",
                quote="No scene tags were available.",
            )
        ]
    return TagProfile(
        **merged,
        evidence=evidence[:20],
        confidence=_average_confidence(scene.tags for scene in scene_list),
    )


def build_query_tags(
    input_text: str,
    *,
    extra_text: str = "",
    candidate_log_path: Path | None = None,
) -> TagProfile:
    text = f"{input_text}\n{extra_text}".strip()
    evidence = [
        TagEvidence(
            source_id="query",
            source_type="query",
            field="input_text",
            quote=_clip(input_text),
            note="query tags extracted from user brief and association output",
        )
    ]
    normalizer = TagNormalizer()
    tag_profile = normalizer.normalize_text(text, evidence=evidence)
    if candidate_log_path is not None:
        append_tag_candidates(
            candidate_log_path,
            build_tag_candidates(
                normalizer.candidate_phrases_from_text(text),
                source_id="query",
                source_type="query",
                field="input_text",
                quote=_clip(input_text, max_chars=220),
            ),
        )
    return tag_profile


def build_card_tags(card: ExperienceCard) -> TagProfile:
    text = " ".join(
        [
            " ".join(card.keywords),
            card.underlying_emotion,
            card.narrative_logic,
            card.director_strategy,
            " ".join(card.shooting_techniques),
            " ".join(card.visual_symbols),
            card.copywriting_tone,
            card.reuse_condition,
        ]
    )
    evidence = [
        TagEvidence(
            source_id=card.source_scene_ids[0],
            source_type="scene",
            field="experience_card",
            quote=_clip(card.director_strategy),
            note=f"derived from card {card.card_id}",
        )
    ]
    return TagNormalizer().normalize_text(text, evidence=evidence)


def retrieve_experience_cards(
    query_tags: TagProfile,
    cards: list[ExperienceCard],
    *,
    top_k: int = 5,
) -> list[ExperienceCard]:
    return [match.card for match in match_experience_cards(query_tags, cards, top_k=top_k)]


def match_experience_cards(
    query_tags: TagProfile,
    cards: list[ExperienceCard],
    *,
    top_k: int = 5,
) -> list[ExperienceCardMatch]:
    matches: list[tuple[TagMatch, int]] = []
    for index, card in enumerate(cards):
        match = _tag_match(query_tags, card)
        if match.score > 0:
            matches.append((match, index))
    matches.sort(key=lambda row: (-row[0].score, row[1]))
    return [
        ExperienceCardMatch(
            card_id=match.card.card_id,
            score=round(match.score, 3),
            tag_score=round(match.score, 3),
            script_stage=match.card.script_usecase.script_stage,
            creative_purpose=match.card.script_usecase.creative_purpose,
            best_usage=match.card.script_usecase.best_usage,
            risk=match.card.script_usecase.risk,
            matched_dimensions=match.matched_dimensions,
            evidence=match.card.tags.evidence,
            card=match.card,
        )
        for match, _ in matches[:top_k]
    ]


def score_experience_card(query_tags: TagProfile, card: ExperienceCard) -> TagMatch:
    return _tag_match(query_tags, card)


def retrieve_experience_card_matches_from_jsonl(
    query_tags: TagProfile,
    cards_path: Path,
    *,
    top_k: int = 5,
) -> RetrievalResult:
    cards = read_jsonl(cards_path, ExperienceCard)
    return RetrievalResult(
        query_tags=query_tags,
        results=match_experience_cards(query_tags, cards, top_k=top_k),
    )


def retrieve_experience_cards_from_jsonl(
    query_tags: TagProfile,
    cards_path: Path,
    *,
    top_k: int = 5,
) -> list[ExperienceCard]:
    return retrieve_experience_cards(
        query_tags,
        read_jsonl(cards_path, ExperienceCard),
        top_k=top_k,
    )


def _scene_text(scene: SceneAnalysis) -> str:
    interpretation = scene.director_interpretation
    observation = scene.visual_observation
    candidates = " ".join(
        " ".join(
            [
                " ".join(candidate.keywords),
                candidate.emotion,
                candidate.narrative_logic,
                " ".join(candidate.techniques),
                candidate.reuse_condition,
            ]
        )
        for candidate in scene.experience_candidates
    )
    return " ".join(
        [
            observation.setting,
            observation.characters,
            observation.action_change,
            observation.composition,
            observation.lighting,
            observation.color,
            observation.camera_motion,
            interpretation.narrative_function,
            interpretation.emotional_function,
            interpretation.brand_personality_signal,
            interpretation.underlying_emotion,
            interpretation.audience_projection,
            " ".join(interpretation.shooting_techniques),
            interpretation.why_it_works,
            candidates,
        ]
    )


def _scene_evidence(scene: SceneAnalysis) -> list[TagEvidence]:
    interpretation = scene.director_interpretation
    return [
        TagEvidence(
            source_id=scene.scene_id,
            source_type="scene",
            field="director_interpretation.underlying_emotion",
            quote=_clip(interpretation.underlying_emotion),
        ),
        TagEvidence(
            source_id=scene.scene_id,
            source_type="scene",
            field="director_interpretation.audience_projection",
            quote=_clip(interpretation.audience_projection),
        ),
        TagEvidence(
            source_id=scene.scene_id,
            source_type="scene",
            field="director_interpretation.why_it_works",
            quote=_clip(interpretation.why_it_works),
        ),
    ]


def _merge_tags(tag_profiles: Iterable[TagProfile]) -> dict[str, list[str]]:
    merged = {dimension: [] for dimension in TAG_DIMENSIONS}
    for profile in tag_profiles:
        for dimension in TAG_DIMENSIONS:
            for tag in getattr(profile, dimension):
                if tag not in merged[dimension]:
                    merged[dimension].append(tag)
    if not any(merged.values()):
        merged["symbolic_logic"] = ["general_expression"]
    return {dimension: tags[:12] for dimension, tags in merged.items()}


def _tag_match(query: TagProfile, card: ExperienceCard) -> TagMatch:
    score = 0.0
    matched_dimensions: dict[str, list[str]] = {}
    for dimension, weight in WEIGHTS.items():
        query_values = canonical_tags(query, dimension)
        target_values = canonical_tags(card.tags, dimension)
        matched = sorted(query_values & target_values)
        if matched:
            matched_dimensions[dimension] = matched
            score += len(matched) * weight
    return TagMatch(score=score, matched_dimensions=matched_dimensions, card=card)


def _average_confidence(tag_profiles: Iterable[TagProfile]) -> float:
    values = [profile.confidence for profile in tag_profiles]
    if not values:
        return 0.35
    return round(sum(values) / len(values), 3)


def build_tag_candidates(
    phrases: Iterable[str],
    *,
    source_id: str,
    source_type: str,
    field: str,
    quote: str,
) -> list[TagCandidate]:
    candidates: list[TagCandidate] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = normalize_candidate_tag(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            TagCandidate(
                candidate=phrase,
                normalized=normalized,
                source_id=source_id,
                source_type=source_type,
                field=field,
                quote=quote,
            )
        )
    return candidates


def append_tag_candidates(path: Path, candidates: Iterable[TagCandidate]) -> None:
    rows = list(candidates)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CANDIDATE_LOG_LOCK:
        existing_keys: set[tuple[str, str, str]] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_keys.add((str(data.get("source_id", "")), str(data.get("field", "")), str(data.get("normalized", ""))))
        new_lines: list[str] = []
        for candidate in rows:
            key = (candidate.source_id, candidate.field, candidate.normalized)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_lines.append(json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False))
        if not new_lines:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(new_lines) + "\n")


def _clip(text: str, max_chars: int = 160) -> str:
    clean = " ".join(text.strip().split())
    if len(clean) <= max_chars:
        return clean or "No evidence text provided."
    return clean[: max_chars - 1].rstrip() + "..."
