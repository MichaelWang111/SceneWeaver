from __future__ import annotations

from collections.abc import Sequence

from sceneweaver.analysis.tags import RetrievalResult
from sceneweaver.retrieval.models import QueryUseCase, RetrievalWeights
from sceneweaver.retrieval.policy import score_experience_match
from sceneweaver.retrieval.usecase import infer_query_usecase
from sceneweaver.schemas import CreativeIntentAnalysis, ExperienceCard
from sceneweaver.schemas.tags import TagProfile


def retrieve_experience_matches(
    *,
    query_tags: TagProfile,
    cards: list[ExperienceCard],
    top_k: int = 5,
    input_text: str = "",
    query_usecase: QueryUseCase | None = None,
    intent_analysis: CreativeIntentAnalysis | None = None,
    intent_weight: float = 0.0,
    semantic_scores: Sequence[float] | None = None,
    semantic_weight: float = 0.0,
    weights: RetrievalWeights | None = None,
) -> RetrievalResult:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if semantic_scores is not None and len(semantic_scores) != len(cards):
        raise ValueError("semantic_scores length must match cards length")
    active_query_usecase = query_usecase or infer_query_usecase(input_text, query_tags)
    rows = []
    for index, card in enumerate(cards):
        semantic_score = semantic_scores[index] if semantic_scores is not None else None
        match = score_experience_match(
            query_tags=query_tags,
            query_usecase=active_query_usecase,
            card=card,
            intent_analysis=intent_analysis,
            intent_weight=intent_weight,
            semantic_score=semantic_score,
            semantic_weight=semantic_weight,
            weights=weights,
        )
        signal_score = (
            match.tag_score
            + match.usecase_score
            + max(0.0, match.intent_score)
            + max(0.0, match.semantic_score or 0.0) * semantic_weight
        )
        if signal_score <= 0:
            continue
        rows.append((match, index))
    rows.sort(key=lambda row: (-row[0].score, row[1]))
    return RetrievalResult(
        query_tags=query_tags,
        results=[match for match, _ in rows[:top_k]],
    )
