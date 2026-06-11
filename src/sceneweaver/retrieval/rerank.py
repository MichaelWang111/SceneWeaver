from __future__ import annotations

import json
from typing import Protocol

from sceneweaver.analysis.tags import ExperienceCardMatch, RetrievalResult
from sceneweaver.retrieval.models import QueryPlan
from sceneweaver.schemas.tags import TagProfile


class RerankLLMClient(Protocol):
    def analyze_text_json(self, **kwargs) -> dict:
        ...


def rerank_matches_with_llm(
    *,
    input_text: str,
    query_tags: TagProfile,
    query_plan: QueryPlan | None,
    matches: list[ExperienceCardMatch],
    client: RerankLLMClient,
    top_k: int,
    timeout_seconds: float,
    retries: int,
) -> RetrievalResult:
    if not matches:
        return RetrievalResult(query_tags=query_tags, results=[])
    payload = {
        "input_text": input_text,
        "query_plan": query_plan.model_dump(mode="json") if query_plan is not None else None,
        "candidates": [_candidate_payload(match) for match in matches],
    }
    response = client.analyze_text_json(
        system_prompt=(
            "You are a strict retrieval reranker. Rank only the given candidates. "
            "Prefer candidates that satisfy the positive need and veto candidates that violate forbidden constraints. "
            "Return compact JSON."
        ),
        user_prompt=(
            "Rerank these SceneWeaver experience cards for the user brief. "
            "Return JSON with key ranked_results, each item containing card_id, score from 0 to 1, "
            "veto boolean, and short reason.\n"
            + json.dumps(payload, ensure_ascii=False)
        ),
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return RetrievalResult(
        query_tags=query_tags,
        results=_apply_rerank_response(matches, response)[:top_k],
    )


def _candidate_payload(match: ExperienceCardMatch) -> dict:
    card = match.card
    return {
        "card_id": match.card_id,
        "score": match.score,
        "script_stage": match.script_stage,
        "creative_purpose": match.creative_purpose,
        "best_usage": match.best_usage,
        "risk": match.risk,
        "constraint_hits": match.constraint_hits,
        "keywords": card.keywords,
        "narrative_logic": card.narrative_logic,
        "director_strategy": card.director_strategy,
        "visual_symbols": card.visual_symbols,
        "reuse_condition": card.reuse_condition,
    }


def _apply_rerank_response(matches: list[ExperienceCardMatch], response: dict) -> list[ExperienceCardMatch]:
    by_id = {match.card_id: match for match in matches}
    ranked_rows = response.get("ranked_results", [])
    if not isinstance(ranked_rows, list):
        return matches
    scored: list[tuple[float, int, ExperienceCardMatch]] = []
    used: set[str] = set()
    for order, row in enumerate(ranked_rows):
        if not isinstance(row, dict):
            continue
        card_id = str(row.get("card_id", ""))
        match = by_id.get(card_id)
        if match is None or card_id in used or bool(row.get("veto", False)):
            continue
        used.add(card_id)
        score = _coerce_score(row.get("score"))
        scored.append((score, order, match))
    scored.sort(key=lambda item: (-item[0], item[1]))
    reranked = [match for _score, _order, match in scored]
    reranked.extend(match for match in matches if match.card_id not in used)
    return reranked


def _coerce_score(value) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))
