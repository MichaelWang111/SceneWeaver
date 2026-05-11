from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import re
from typing import Callable

from pydantic import Field

from sceneweaver.analysis.associate_analyzer import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    AssociateLLMClient,
    associate_input,
)
from sceneweaver.analysis.semantic import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_WEIGHT,
    EmbeddingBackend,
    SentenceTransformerBackend,
    build_query_embedding_text,
    semantic_scores,
)
from sceneweaver.analysis.tags import ExperienceCardMatch, RetrievalResult, match_experience_cards, score_experience_card
from sceneweaver.schemas import AssociationAnalysis, ExperienceCard
from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.storage.json_store import read_jsonl, write_json

LogFn = Callable[[str], None]


class KeywordLoopResult(StrictBaseModel):
    input_text: str = Field(min_length=1)
    association_path: str
    candidate_log_path: str
    experience_cards_path: str
    experience_cards_paths: list[str] = Field(default_factory=list)
    searched_card_count: int = 0
    matched_card_count: int = 0
    semantic_enabled: bool = False
    embedding_model: str | None = None
    semantic_weight: float = 0.0
    top_matches: list["KeywordLoopMatchSummary"] = Field(default_factory=list)
    association_analysis: AssociationAnalysis
    retrieval: RetrievalResult
    next_actions: list[str] = Field(default_factory=list)


class KeywordLoopMatchSummary(StrictBaseModel):
    card_id: str
    source_video_id: str
    source_scene_ids: list[str]
    score: float
    tag_score: float
    semantic_score: float | None = None
    matched_dimensions: dict[str, list[str]]
    keywords: list[str]
    reuse_condition: str


def run_keyword_loop(
    input_text: str,
    card_source: Path,
    *,
    client: AssociateLLMClient | None = None,
    association_output_path: Path | None = None,
    result_output_path: Path | None = None,
    top_k: int = 5,
    semantic: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    embedding_backend: EmbeddingBackend | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    prompt_path: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    log: LogFn | None = None,
    stream_callback: Callable[[str], None] | None = None,
    reasoning_callback: Callable[[str], None] | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
) -> KeywordLoopResult:
    clean_input = input_text.strip()
    if not clean_input:
        raise ValueError("input_text cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if semantic_weight < 0:
        raise ValueError("semantic_weight must be >= 0")

    card_paths = discover_experience_card_paths(card_source)
    if not card_paths:
        raise FileNotFoundError(
            f"experience cards not found under: {card_source.resolve()}. "
            "Pass an experience_cards.jsonl file, a film output directory, or a collection directory."
        )
    cards = read_experience_cards(card_paths)
    if not cards:
        raise ValueError(f"no experience cards found in: {', '.join(str(path) for path in card_paths)}")
    _log(log, f"Experience card files discovered: {len(card_paths)}")
    _log(log, f"Experience cards loaded: {len(cards)}")

    default_output_dir = _default_loop_output_dir(card_source)

    association_path = association_output_path or build_keyword_loop_association_path(
        default_output_dir,
        clean_input,
    )

    _log(log, "Phase 1/2: expanding keyword with LLM and refreshing query tags.")
    association = associate_input(
        clean_input,
        client=client,
        prompt_path=prompt_path,
        output_path=association_path,
        max_items=max_items,
        timeout_seconds=timeout_seconds,
        retries=retries,
        log=log,
        stream_callback=stream_callback,
        reasoning_callback=reasoning_callback,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )

    _log(log, "Phase 2/2: matching query tags against experience cards.")
    if semantic:
        _log(log, f"Semantic rerank enabled: model={embedding_model}, weight={semantic_weight:g}")
        retrieval = match_experience_cards_semantic(
            association,
            cards,
            top_k=top_k,
            input_text=clean_input,
            embedding_model=embedding_model,
            semantic_weight=semantic_weight,
            embedding_backend=embedding_backend,
        )
    else:
        retrieval = RetrievalResult(
            query_tags=association.query_tags,
            results=match_experience_cards(association.query_tags, cards, top_k=top_k),
        )
    _log(log, f"Experience card matches: {len(retrieval.results)}")
    for match in retrieval.results[:3]:
        _log(
            log,
            "Top match: "
            f"card_id={match.card_id}, score={match.score:g}, "
            f"source_video_id={match.card.source_video_id}, "
            f"matched_dimensions={','.join(match.matched_dimensions.keys())}",
        )
    candidate_log_path = association_path.parent / "tag_candidates.jsonl"
    result = KeywordLoopResult(
        input_text=clean_input,
        association_path=str(association_path.resolve()),
        candidate_log_path=str(candidate_log_path.resolve()),
        experience_cards_path=str(card_paths[0].resolve()),
        experience_cards_paths=[str(path.resolve()) for path in card_paths],
        searched_card_count=len(cards),
        matched_card_count=len(retrieval.results),
        semantic_enabled=semantic,
        embedding_model=embedding_model if semantic else None,
        semantic_weight=semantic_weight if semantic else 0.0,
        top_matches=summarize_matches(retrieval),
        association_analysis=association,
        retrieval=retrieval,
        next_actions=_next_actions(retrieval, candidate_log_path),
    )
    if result_output_path is not None:
        write_json(result_output_path, result)
    return result


def match_experience_cards_semantic(
    association: AssociationAnalysis,
    cards: list[ExperienceCard],
    *,
    input_text: str,
    top_k: int,
    embedding_model: str,
    semantic_weight: float,
    embedding_backend: EmbeddingBackend | None = None,
) -> RetrievalResult:
    backend = embedding_backend or SentenceTransformerBackend(embedding_model)
    query_text = build_query_embedding_text(input_text, _association_embedding_text(association))
    scores = semantic_scores(query_text, cards, backend=backend)
    rows: list[tuple[ExperienceCardMatch, int]] = []
    for index, (card, semantic_score) in enumerate(zip(cards, scores)):
        tag_match = score_experience_card(association.query_tags, card)
        final_score = tag_match.score + max(0.0, semantic_score) * semantic_weight
        if final_score <= 0:
            continue
        rows.append(
            (
                ExperienceCardMatch(
                    card_id=card.card_id,
                    score=round(final_score, 3),
                    tag_score=round(tag_match.score, 3),
                    semantic_score=round(semantic_score, 4),
                    matched_dimensions=tag_match.matched_dimensions,
                    evidence=card.tags.evidence,
                    card=card,
                ),
                index,
            )
        )
    rows.sort(key=lambda row: (-row[0].score, row[1]))
    return RetrievalResult(
        query_tags=association.query_tags,
        results=[match for match, _ in rows[:top_k]],
    )


def discover_experience_card_paths(card_source: Path) -> list[Path]:
    source = card_source.resolve()
    if source.is_file():
        if source.name != "experience_cards.jsonl":
            raise ValueError(f"card source file must be named experience_cards.jsonl: {source}")
        return [source]
    direct_path = source / "analysis" / "experience_cards.jsonl"
    if direct_path.exists():
        return [direct_path]
    return sorted(source.glob("**/analysis/experience_cards.jsonl"))


def read_experience_cards(paths: list[Path]) -> list[ExperienceCard]:
    cards: list[ExperienceCard] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for card in read_jsonl(path, ExperienceCard):
            key = (card.source_video_id, card.card_id)
            if key in seen:
                continue
            seen.add(key)
            cards.append(card)
    return cards


def summarize_matches(retrieval: RetrievalResult, *, limit: int = 5) -> list[KeywordLoopMatchSummary]:
    return [
        KeywordLoopMatchSummary(
            card_id=match.card_id,
            source_video_id=match.card.source_video_id,
            source_scene_ids=match.card.source_scene_ids,
            score=match.score,
            tag_score=match.tag_score,
            semantic_score=match.semantic_score,
            matched_dimensions=match.matched_dimensions,
            keywords=match.card.keywords,
            reuse_condition=match.card.reuse_condition,
        )
        for match in retrieval.results[:limit]
    ]


def build_keyword_loop_association_path(
    loop_output_dir: Path,
    input_text: str,
    *,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(input_text.strip().encode("utf-8")).hexdigest()[:8]
    slug = _safe_filename_slug(input_text)
    return loop_output_dir.resolve() / f"{timestamp}_{slug}_{digest}_association.json"


def _next_actions(retrieval: RetrievalResult, candidate_log_path: Path) -> list[str]:
    actions: list[str] = []
    if not retrieval.results:
        actions.append("No experience cards matched. Review query_tags and expand taxonomy aliases or add more cards.")
    if candidate_log_path.exists():
        actions.append(f"Review pending tag candidates in {candidate_log_path}.")
    else:
        actions.append("No unmanaged tag candidates were recorded for this query.")
    return actions


def _association_embedding_text(association: AssociationAnalysis) -> str:
    parts = [association.core_reading, association.emotional_arc.arc_summary]
    parts.extend(association.avoid_cliches)
    for items in association.association_map.model_dump().values():
        for item in items:
            parts.extend(
                str(item.get(field, ""))
                for field in ("term", "meaning", "emotion", "image_hint", "usage_hint")
            )
    for possibility in association.director_possibilities:
        parts.extend(
            [
                possibility.name,
                possibility.concept,
                possibility.emotional_direction,
                possibility.visual_direction,
                possibility.narrative_direction,
            ]
        )
    return "\n".join(part for part in parts if part)


def _safe_filename_slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    if not slug:
        return "keyword"
    return slug[:32].strip("_") or "keyword"


def _default_loop_output_dir(card_source: Path) -> Path:
    source = card_source.resolve()
    if source.is_file():
        return source.parent / "keyword_loops"
    if (source / "analysis" / "experience_cards.jsonl").exists():
        return source / "analysis"
    return source / "keyword_loops"


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)
