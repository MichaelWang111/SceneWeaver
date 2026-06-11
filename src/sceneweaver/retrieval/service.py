from __future__ import annotations

from collections.abc import Sequence

from sceneweaver.analysis.tags import RetrievalResult
from sceneweaver.retrieval.lexical import (
    DEFAULT_RRF_K,
    lexical_scores as compute_lexical_scores,
    ranked_indices_from_scores,
    reciprocal_rank_fusion,
)
from sceneweaver.retrieval.models import QueryPlan, QueryUseCase, RetrievalWeights
from sceneweaver.retrieval.policy import score_experience_match
from sceneweaver.retrieval.query_plan import build_query_plan, card_has_forbidden_stage, query_usecase_from_plan
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
    query_plan: QueryPlan | None = None,
    intent_analysis: CreativeIntentAnalysis | None = None,
    intent_weight: float = 0.0,
    semantic_scores: Sequence[float] | None = None,
    semantic_weight: float = 0.0,
    lexical_scores: Sequence[float] | None = None,
    lexical_weight: float = 2.0,
    retrieval_workflow: str = "semantic_constraints",
    rrf_k: int = DEFAULT_RRF_K,
    constraints_enabled: bool = True,
    hard_filter_forbidden_stage: bool = True,
    weights: RetrievalWeights | None = None,
) -> RetrievalResult:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if semantic_scores is not None and len(semantic_scores) != len(cards):
        raise ValueError("semantic_scores length must match cards length")
    if lexical_scores is not None and len(lexical_scores) != len(cards):
        raise ValueError("lexical_scores length must match cards length")
    active_query_plan = query_plan or (
        build_query_plan(input_text) if constraints_enabled and input_text.strip() else None
    )
    if lexical_scores is None and _uses_lexical(retrieval_workflow):
        lexical_scores = compute_lexical_scores(input_text, cards, query_plan=active_query_plan)
    rrf_scores = _rrf_scores(
        semantic_scores=semantic_scores,
        lexical_scores=lexical_scores,
        item_count=len(cards),
        rrf_k=rrf_k,
    ) if retrieval_workflow.startswith("hybrid_rrf") else [0.0 for _card in cards]
    usecase_input_text = active_query_plan.positive_query if active_query_plan is not None else input_text
    inferred_usecase = infer_query_usecase(usecase_input_text, query_tags)
    active_query_usecase = query_usecase or (
        query_usecase_from_plan(active_query_plan, inferred_usecase)
        if active_query_plan is not None
        else inferred_usecase
    )
    rows = []
    for index, card in enumerate(cards):
        if (
            constraints_enabled
            and _uses_constraints(retrieval_workflow)
            and hard_filter_forbidden_stage
            and card_has_forbidden_stage(active_query_plan, card)
        ):
            continue
        semantic_score = semantic_scores[index] if semantic_scores is not None else None
        lexical_score = lexical_scores[index] if lexical_scores is not None else None
        score_semantic_weight = semantic_weight if _uses_semantic(retrieval_workflow) else 0.0
        score_lexical_weight = lexical_weight if retrieval_workflow == "lexical_only" else 0.0
        score_constraints_enabled = constraints_enabled and _uses_constraints(retrieval_workflow)
        match = score_experience_match(
            query_tags=query_tags,
            query_usecase=active_query_usecase,
            card=card,
            intent_analysis=intent_analysis,
            intent_weight=intent_weight,
            semantic_score=semantic_score,
            semantic_weight=score_semantic_weight,
            lexical_score=lexical_score,
            lexical_weight=score_lexical_weight,
            rrf_score=rrf_scores[index] * 100,
            ranking_workflow=retrieval_workflow,
            query_plan=active_query_plan,
            constraints_enabled=score_constraints_enabled,
            weights=weights,
        )
        match = _apply_workflow_score(
            match,
            retrieval_workflow=retrieval_workflow,
            semantic_weight=score_semantic_weight,
            lexical_weight=score_lexical_weight,
        )
        signal_score = match.score - match.quality_score
        if signal_score <= 0:
            continue
        rows.append((match, index))
    rows.sort(key=lambda row: (-row[0].score, row[1]))
    return RetrievalResult(
        query_tags=query_tags,
        results=[match for match, _ in rows[:top_k]],
    )


def _apply_workflow_score(
    match,
    *,
    retrieval_workflow: str,
    semantic_weight: float,
    lexical_weight: float,
):
    if retrieval_workflow == "semantic_only":
        score = max(0.0, match.semantic_score or 0.0) * semantic_weight + match.quality_score
    elif retrieval_workflow == "lexical_only":
        score = max(0.0, match.lexical_score or 0.0) * lexical_weight + match.quality_score
    elif retrieval_workflow == "hybrid_rrf":
        score = match.rrf_score + match.quality_score
    elif retrieval_workflow in {"hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}:
        score = match.rrf_score + match.quality_score + match.constraint_score
    else:
        score = match.score
    return match.model_copy(update={"score": round(max(0.0, score), 3)})


def _uses_semantic(workflow: str) -> bool:
    return workflow in {"semantic_only", "semantic_constraints"}


def _uses_lexical(workflow: str) -> bool:
    return workflow in {"lexical_only", "hybrid_rrf", "hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}


def _uses_constraints(workflow: str) -> bool:
    return workflow in {"semantic_constraints", "hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}


def _rrf_scores(
    *,
    semantic_scores: Sequence[float] | None,
    lexical_scores: Sequence[float] | None,
    item_count: int,
    rrf_k: int,
) -> list[float]:
    ranked_lists = []
    if semantic_scores is not None:
        ranked_lists.append(ranked_indices_from_scores(semantic_scores))
    if lexical_scores is not None:
        ranked_lists.append(ranked_indices_from_scores(lexical_scores))
    if not ranked_lists:
        return [0.0 for _index in range(item_count)]
    return reciprocal_rank_fusion(ranked_lists, item_count=item_count, k=rrf_k)
