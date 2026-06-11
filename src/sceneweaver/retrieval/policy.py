from __future__ import annotations

from sceneweaver.analysis.semantic import build_card_embedding_text
from sceneweaver.analysis.tags import ExperienceCardMatch, score_experience_card
from sceneweaver.retrieval.models import QueryPlan, QueryUseCase, RetrievalWeights
from sceneweaver.retrieval.query_plan import score_query_constraints
from sceneweaver.schemas import CreativeIntentAnalysis, ExperienceCard
from sceneweaver.schemas.experience_card import ScriptUseCase
from sceneweaver.schemas.tags import TagProfile


def score_experience_match(
    *,
    query_tags: TagProfile,
    query_usecase: QueryUseCase,
    card: ExperienceCard,
    intent_analysis: CreativeIntentAnalysis | None = None,
    intent_weight: float = 0.0,
    semantic_score: float | None = None,
    semantic_weight: float = 0.0,
    lexical_score: float | None = None,
    lexical_weight: float = 0.0,
    rrf_score: float = 0.0,
    ranking_workflow: str = "semantic_constraints",
    query_plan: QueryPlan | None = None,
    constraints_enabled: bool = True,
    weights: RetrievalWeights | None = None,
) -> ExperienceCardMatch:
    active_weights = weights or RetrievalWeights()
    tag_match = score_experience_card(query_tags, card)
    usecase_score, matched_usecase = score_usecase(query_usecase, card.script_usecase, weights=active_weights)
    intent_score = (
        score_card_against_intent(intent_analysis, card, intent_weight=intent_weight)
        if intent_analysis is not None
        else 0.0
    )
    quality_score = card.confidence * active_weights.quality
    semantic_contribution = max(0.0, semantic_score or 0.0) * semantic_weight
    lexical_contribution = max(0.0, lexical_score or 0.0) * lexical_weight
    constraint_score, constraint_hits = (
        score_query_constraints(query_plan, card, weights=active_weights)
        if constraints_enabled
        else (0.0, {})
    )
    final_score = (
        tag_match.score
        + usecase_score
        + intent_score
        + semantic_contribution
        + lexical_contribution
        + quality_score
        + constraint_score
        + rrf_score
    )
    return ExperienceCardMatch(
        card_id=card.card_id,
        score=round(max(0.0, final_score), 3),
        tag_score=round(tag_match.score, 3),
        usecase_score=round(usecase_score, 3),
        intent_score=round(intent_score, 3),
        constraint_score=round(constraint_score, 3),
        constraint_hits=constraint_hits,
        quality_score=round(quality_score, 3),
        semantic_score=round(semantic_score, 4) if semantic_score is not None else None,
        lexical_score=round(lexical_score, 4) if lexical_score is not None else None,
        rrf_score=round(rrf_score, 6),
        ranking_workflow=ranking_workflow,
        matched_dimensions=tag_match.matched_dimensions,
        matched_usecase=matched_usecase,
        script_stage=card.script_usecase.script_stage,
        creative_purpose=card.script_usecase.creative_purpose,
        best_usage=card.script_usecase.best_usage,
        risk=card.script_usecase.risk,
        evidence=card.tags.evidence,
        card=card,
    )


def score_usecase(
    query_usecase: QueryUseCase,
    card_usecase: ScriptUseCase,
    *,
    weights: RetrievalWeights | None = None,
) -> tuple[float, dict[str, list[str]]]:
    active_weights = weights or RetrievalWeights()
    score = 0.0
    matched: dict[str, list[str]] = {}
    if query_usecase.script_stage != "general" and query_usecase.script_stage == card_usecase.script_stage:
        score += active_weights.script_stage_match
        matched["script_stage"] = [card_usecase.script_stage]
    query_purposes = {purpose for purpose in query_usecase.creative_purpose if purpose != "general_expression"}
    card_purposes = {purpose for purpose in card_usecase.creative_purpose if purpose != "general_expression"}
    purpose_matches = sorted(query_purposes & card_purposes)
    if purpose_matches:
        score += len(purpose_matches) * active_weights.creative_purpose_match
        matched["creative_purpose"] = purpose_matches
    return score, matched


def score_card_against_intent(
    intent_analysis: CreativeIntentAnalysis,
    card: ExperienceCard,
    *,
    intent_weight: float,
) -> float:
    if intent_weight <= 0:
        return 0.0
    text = _card_intent_text(card)
    must_hits = _count_term_hits(
        [
            *intent_analysis.must_match,
            *intent_analysis.intent_keywords,
            *intent_analysis.selection_criteria,
        ],
        text,
    )
    nice_hits = _count_term_hits(intent_analysis.nice_to_have, text)
    avoid_hits = _count_term_hits(intent_analysis.avoid, text)
    return (must_hits * intent_weight) + (nice_hits * intent_weight * 0.5) - (avoid_hits * intent_weight)


def _card_intent_text(card: ExperienceCard) -> str:
    tag_parts: list[str] = []
    for values in card.tags.model_dump(mode="json").values():
        if isinstance(values, list):
            tag_parts.extend(str(value) for value in values if isinstance(value, str))
    usecase = card.script_usecase
    return "\n".join(
        part
        for part in (
            build_card_embedding_text(card),
            " ".join(tag_parts),
            usecase.script_stage,
            " ".join(usecase.creative_purpose),
            usecase.best_usage,
            usecase.risk,
            " ".join(card.avoid),
        )
        if part.strip()
    ).lower()


def _count_term_hits(terms: list[str], text: str) -> int:
    hits = 0
    seen: set[str] = set()
    for term in terms:
        clean = str(term).strip().lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        variants = {
            clean,
            clean.replace("_", " "),
            clean.replace(" ", "_"),
        }
        if any(variant and variant in text for variant in variants):
            hits += 1
    return hits
