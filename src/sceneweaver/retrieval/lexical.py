from __future__ import annotations

from collections import Counter
import math
import re
from typing import Sequence

from sceneweaver.retrieval.models import QueryPlan
from sceneweaver.retrieval.style import infer_card_style_risks, infer_card_style_traits
from sceneweaver.schemas import ExperienceCard

DEFAULT_RRF_K = 60


def lexical_scores(
    query_text: str,
    cards: Sequence[ExperienceCard],
    *,
    query_plan: QueryPlan | None = None,
) -> list[float]:
    query = _query_lexical_text(query_text, query_plan=query_plan)
    query_terms = tokenize(query)
    if not query_terms or not cards:
        return [0.0 for _card in cards]
    documents = [tokenize(build_card_lexical_text(card)) for card in cards]
    return bm25_scores(query_terms, documents)


def bm25_scores(
    query_terms: list[str],
    documents: Sequence[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not documents:
        return []
    doc_count = len(documents)
    doc_lengths = [len(document) for document in documents]
    avg_length = sum(doc_lengths) / max(1, doc_count)
    df: Counter[str] = Counter()
    for document in documents:
        df.update(set(document))
    query_counts = Counter(query_terms)
    scores: list[float] = []
    for document, doc_length in zip(documents, doc_lengths):
        tf = Counter(document)
        score = 0.0
        for term, query_count in query_counts.items():
            if term not in tf:
                continue
            idf = math.log(1 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_length / max(1.0, avg_length))
            score += query_count * idf * (tf[term] * (k1 + 1) / denom)
        scores.append(round(score, 6))
    return scores


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    item_count: int,
    k: int = DEFAULT_RRF_K,
) -> list[float]:
    scores = [0.0 for _index in range(item_count)]
    for ranked in ranked_lists:
        for rank, item_index in enumerate(ranked, start=1):
            if 0 <= item_index < item_count:
                scores[item_index] += 1.0 / (k + rank)
    return scores


def ranked_indices_from_scores(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def tokenize(text: str) -> list[str]:
    normalized = str(text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", normalized)
    cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
    cjk_tokens: list[str] = []
    for chunk in cjk_chunks:
        cjk_tokens.extend(_cjk_ngrams(chunk))
    return [*ascii_tokens, *cjk_tokens]


def build_card_lexical_text(card: ExperienceCard) -> str:
    usecase = card.script_usecase
    tag_parts: list[str] = []
    for value in card.tags.model_dump(mode="json").values():
        if isinstance(value, list):
            tag_parts.extend(str(item) for item in value if isinstance(item, str))
    high_weight = [
        usecase.script_stage,
        *usecase.creative_purpose,
        usecase.best_usage,
        usecase.best_usage,
        usecase.best_usage,
    ]
    mid_weight = [
        *card.keywords,
        *card.keywords,
        card.narrative_logic,
        card.director_strategy,
        card.reuse_condition,
    ]
    low_weight = [
        *card.visual_symbols,
        " ".join(tag_parts),
    ]
    style_weight = [
        *card.style_traits,
        *card.style_risks,
        *card.avoid,
        *infer_card_style_traits(card),
        *infer_card_style_risks(card),
    ]
    return " ".join(part for part in [*high_weight, *mid_weight, *low_weight, *style_weight] if part)


def _query_lexical_text(query_text: str, *, query_plan: QueryPlan | None) -> str:
    if query_plan is None:
        return query_text
    parts = [
        query_plan.positive_query,
        " ".join(query_plan.desired_stage),
        " ".join(query_plan.positive_purposes),
        " ".join(query_plan.positive_style),
        " ".join(query_plan.visual_hints),
    ]
    return " ".join(part for part in parts if part)


def _cjk_ngrams(text: str) -> list[str]:
    if len(text) <= 2:
        return [text]
    tokens: list[str] = []
    for size in (2, 3):
        tokens.extend(text[index : index + size] for index in range(0, len(text) - size + 1))
    tokens.append(text)
    return tokens
