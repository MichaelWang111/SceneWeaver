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
from sceneweaver.analysis.intent_analyzer import analyze_creative_intent
from sceneweaver.analysis.semantic import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_WEIGHT,
    EmbeddingBackend,
    SentenceTransformerBackend,
    build_query_embedding_text,
    semantic_channel_scores,
)
from sceneweaver.analysis.tag_expander import expand_input_tags
from sceneweaver.analysis.tags import RetrievalResult
from sceneweaver.llm.client import VisionLLMClient
from sceneweaver.retrieval.models import QueryPlan
from sceneweaver.retrieval.query_plan import build_query_plan
from sceneweaver.retrieval.rerank import rerank_matches_with_llm
from sceneweaver.retrieval.service import retrieve_experience_matches
from sceneweaver.schemas import AssociationAnalysis, CreativeIntentAnalysis, ExperienceCard, TagExpansionAnalysis
from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.storage.json_store import read_jsonl, write_json

LogFn = Callable[[str], None]


class KeywordLoopResult(StrictBaseModel):
    input_text: str = Field(min_length=1)
    mode: str = "associate"
    association_path: str
    candidate_log_path: str
    experience_cards_path: str
    experience_cards_paths: list[str] = Field(default_factory=list)
    unindexed_scene_dirs: list[str] = Field(default_factory=list)
    searched_card_count: int = 0
    matched_card_count: int = 0
    semantic_enabled: bool = False
    embedding_model: str | None = None
    semantic_weight: float = 0.0
    lexical_weight: float = 0.0
    retrieval_workflow: str = "semantic_constraints"
    rrf_k: int = 60
    intent_weight: float = 0.0
    query_plan: QueryPlan | None = None
    llm_rerank_enabled: bool = False
    llm_rerank_top_n: int = 0
    top_matches: list["KeywordLoopMatchSummary"] = Field(default_factory=list)
    association_analysis: AssociationAnalysis | None = None
    tag_expansion_analysis: TagExpansionAnalysis | None = None
    intent_analysis: CreativeIntentAnalysis | None = None
    retrieval: RetrievalResult
    next_actions: list[str] = Field(default_factory=list)


class KeywordLoopMatchSummary(StrictBaseModel):
    card_id: str
    source_video_id: str
    source_scene_ids: list[str]
    score: float
    tag_score: float
    usecase_score: float = 0.0
    intent_score: float = 0.0
    constraint_score: float = 0.0
    constraint_hits: dict[str, list[str]] = Field(default_factory=dict)
    quality_score: float = 0.0
    semantic_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float = 0.0
    ranking_workflow: str = "semantic_constraints"
    matched_dimensions: dict[str, list[str]]
    matched_usecase: dict[str, list[str]] = Field(default_factory=dict)
    script_stage: str = "general"
    creative_purpose: list[str] = Field(default_factory=list)
    best_usage: str = ""
    risk: str = ""
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
    just_tags: bool = False,
    intent: bool = False,
    intent_weight: float = 3.0,
    semantic: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    lexical_weight: float = 2.0,
    retrieval_workflow: str = "semantic_constraints",
    rrf_k: int = 60,
    embedding_backend: EmbeddingBackend | None = None,
    llm_rerank: bool = False,
    llm_rerank_top_n: int = 20,
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
    if lexical_weight < 0:
        raise ValueError("lexical_weight must be >= 0")
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if intent_weight < 0:
        raise ValueError("intent_weight must be >= 0")
    if llm_rerank_top_n < 1:
        raise ValueError("llm_rerank_top_n must be >= 1")
    if just_tags and intent:
        raise ValueError("just_tags and intent cannot both be enabled")

    card_paths = discover_experience_card_paths(card_source)
    unindexed_scene_dirs = discover_unindexed_scene_dirs(card_source)
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

    if intent:
        _log(log, "Phase 1/2: analyzing creative core intent and refreshing query tags.")
        intent_analysis = analyze_creative_intent(
            clean_input,
            client=client,
            output_path=association_path,
            timeout_seconds=timeout_seconds,
            retries=retries,
            log=log,
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        association = None
        tag_expansion = None
        query_tags = intent_analysis.query_tags
    elif just_tags:
        _log(log, "Phase 1/2: lightweight tag expansion and query tag refresh.")
        tag_expansion = expand_input_tags(
            clean_input,
            client=client,
            output_path=association_path,
            timeout_seconds=timeout_seconds,
            retries=retries,
            log=log,
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        association = None
        intent_analysis = None
        query_tags = tag_expansion.query_tags
    else:
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
        tag_expansion = None
        intent_analysis = None
        query_tags = association.query_tags

    _log(log, "Phase 2/2: matching query tags and script use case against experience cards.")
    extra_text = _extra_text_from_analysis(association, tag_expansion, intent_analysis)
    query_plan = build_query_plan(clean_input)
    query_context_text = build_query_embedding_text(
        clean_input,
        extra_text,
        query_plan=query_plan,
    )
    retrieval_top_k = max(top_k, llm_rerank_top_n) if llm_rerank else top_k
    if semantic:
        _log(log, f"Semantic rerank enabled: model={embedding_model}, weight={semantic_weight:g}")
        retrieval = match_experience_cards_semantic(
            association=association,
            tag_expansion=tag_expansion,
            intent_analysis=intent_analysis,
            cards=cards,
            top_k=retrieval_top_k,
            input_text=clean_input,
            query_plan=query_plan,
            embedding_model=embedding_model,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            retrieval_workflow=retrieval_workflow,
            rrf_k=rrf_k,
            intent_weight=intent_weight,
            embedding_backend=embedding_backend,
        )
    else:
        retrieval = retrieve_experience_matches(
            query_tags=query_tags,
            cards=cards,
            top_k=retrieval_top_k,
            input_text=query_context_text,
            query_plan=query_plan,
            intent_analysis=intent_analysis,
            intent_weight=intent_weight if intent_analysis is not None else 0.0,
            lexical_weight=lexical_weight,
            retrieval_workflow=retrieval_workflow,
            rrf_k=rrf_k,
        )
    if llm_rerank:
        _log(log, f"LLM rerank enabled: candidates={len(retrieval.results)}, top_k={top_k}")
        rerank_client = client or VisionLLMClient()
        retrieval = rerank_matches_with_llm(
            input_text=clean_input,
            query_tags=query_tags,
            query_plan=query_plan,
            matches=retrieval.results,
            client=rerank_client,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
            retries=retries,
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
        mode="intent" if intent else "just-tags" if just_tags else "associate",
        association_path=str(association_path.resolve()),
        candidate_log_path=str(candidate_log_path.resolve()),
        experience_cards_path=str(card_paths[0].resolve()),
        experience_cards_paths=[str(path.resolve()) for path in card_paths],
        unindexed_scene_dirs=[str(path.resolve()) for path in unindexed_scene_dirs],
        searched_card_count=len(cards),
        matched_card_count=len(retrieval.results),
        semantic_enabled=semantic,
        embedding_model=embedding_model if semantic else None,
        semantic_weight=semantic_weight if semantic else 0.0,
        lexical_weight=lexical_weight,
        retrieval_workflow=retrieval_workflow,
        rrf_k=rrf_k,
        intent_weight=intent_weight if intent else 0.0,
        query_plan=query_plan,
        llm_rerank_enabled=llm_rerank,
        llm_rerank_top_n=llm_rerank_top_n if llm_rerank else 0,
        top_matches=summarize_matches(retrieval),
        association_analysis=association,
        tag_expansion_analysis=tag_expansion,
        intent_analysis=intent_analysis,
        retrieval=retrieval,
        next_actions=_next_actions(retrieval, candidate_log_path, unindexed_scene_dirs),
    )
    if result_output_path is not None:
        write_json(result_output_path, result)
    return result


def match_experience_cards_semantic(
    association: AssociationAnalysis | None,
    tag_expansion: TagExpansionAnalysis | None,
    intent_analysis: CreativeIntentAnalysis | None,
    cards: list[ExperienceCard],
    *,
    input_text: str,
    query_plan: QueryPlan | None = None,
    top_k: int,
    embedding_model: str,
    semantic_weight: float,
    lexical_weight: float = 2.0,
    retrieval_workflow: str = "semantic_constraints",
    rrf_k: int = 60,
    intent_weight: float = 0.0,
    embedding_backend: EmbeddingBackend | None = None,
) -> RetrievalResult:
    backend = embedding_backend or SentenceTransformerBackend(embedding_model)
    query_tags = _query_tags_from_analysis(association, tag_expansion, intent_analysis)
    extra_text = _extra_text_from_analysis(association, tag_expansion, intent_analysis)
    query_text = build_query_embedding_text(input_text, extra_text, query_plan=query_plan)
    scores = semantic_channel_scores(
        input_text,
        cards,
        backend=backend,
        association_text=extra_text,
        query_plan=query_plan,
    )
    return retrieve_experience_matches(
        query_tags=query_tags,
        cards=cards,
        top_k=top_k,
        input_text=query_text,
        query_plan=query_plan,
        intent_analysis=intent_analysis,
        intent_weight=intent_weight if intent_analysis is not None else 0.0,
        semantic_scores=scores,
        semantic_weight=semantic_weight,
        lexical_weight=lexical_weight,
        retrieval_workflow=retrieval_workflow,
        rrf_k=rrf_k,
    )


def match_experience_cards_with_intent(
    intent_analysis: CreativeIntentAnalysis,
    cards: list[ExperienceCard],
    *,
    top_k: int,
    intent_weight: float,
) -> RetrievalResult:
    return retrieve_experience_matches(
        query_tags=intent_analysis.query_tags,
        cards=cards,
        top_k=top_k,
        input_text=intent_analysis.expanded_text,
        intent_analysis=intent_analysis,
        intent_weight=intent_weight,
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


def discover_unindexed_scene_dirs(card_source: Path) -> list[Path]:
    source = card_source.resolve()
    if source.is_file():
        return []
    analysis_dirs: list[Path]
    if (source / "analysis" / "scenes.json").exists():
        analysis_dirs = [source / "analysis"]
    else:
        analysis_dirs = [path.parent for path in source.glob("**/analysis/scenes.json")]
    return sorted(
        analysis_dir
        for analysis_dir in analysis_dirs
        if not (analysis_dir / "experience_cards.jsonl").exists()
    )


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
            usecase_score=match.usecase_score,
            intent_score=match.intent_score,
            constraint_score=match.constraint_score,
            constraint_hits=match.constraint_hits,
            quality_score=match.quality_score,
            semantic_score=match.semantic_score,
            lexical_score=match.lexical_score,
            rrf_score=match.rrf_score,
            ranking_workflow=match.ranking_workflow,
            matched_dimensions=match.matched_dimensions,
            matched_usecase=match.matched_usecase,
            script_stage=match.script_stage,
            creative_purpose=match.creative_purpose,
            best_usage=match.best_usage,
            risk=match.risk,
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


def _next_actions(
    retrieval: RetrievalResult,
    candidate_log_path: Path,
    unindexed_scene_dirs: list[Path],
) -> list[str]:
    actions: list[str] = []
    for analysis_dir in unindexed_scene_dirs:
        actions.append(f"Experience cards missing for {analysis_dir}. Run extract-experience on its film directory.")
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


def _query_tags_from_analysis(
    association: AssociationAnalysis | None,
    tag_expansion: TagExpansionAnalysis | None,
    intent_analysis: CreativeIntentAnalysis | None,
):
    if association is not None:
        return association.query_tags
    if tag_expansion is not None:
        return tag_expansion.query_tags
    if intent_analysis is not None:
        return intent_analysis.query_tags
    raise ValueError("one analysis object is required")


def _extra_text_from_analysis(
    association: AssociationAnalysis | None,
    tag_expansion: TagExpansionAnalysis | None,
    intent_analysis: CreativeIntentAnalysis | None,
) -> str:
    if association is not None:
        return _association_embedding_text(association)
    if tag_expansion is not None:
        return tag_expansion.expanded_text
    if intent_analysis is not None:
        return intent_analysis.expanded_text
    return ""


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
