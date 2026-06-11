from __future__ import annotations

from math import sqrt
from typing import Protocol, Sequence

from sceneweaver.retrieval.models import QueryPlan
from sceneweaver.schemas import ExperienceCard

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_SEMANTIC_WEIGHT = 4.0
DEFAULT_SEMANTIC_CHANNEL_WEIGHTS = {
    "script_use": 0.50,
    "experience": 0.25,
    "visual_tags": 0.15,
    "combined": 0.10,
}


class EmbeddingBackend(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SentenceTransformerBackend:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic retrieval requires sentence-transformers. "
                'Install with: python -m pip install -e ".[semantic]"'
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in embeddings]


def build_query_embedding_text(
    input_text: str,
    association_text: str = "",
    *,
    query_plan: QueryPlan | None = None,
) -> str:
    positive_text = query_plan.positive_query if query_plan is not None else input_text
    return "\n".join(part for part in (positive_text.strip(), association_text.strip()) if part)


def build_card_embedding_text(card: ExperienceCard) -> str:
    channels = build_card_embedding_channels(card)
    return "\n".join(channels[channel] for channel in ("script_use", "experience", "visual_tags") if channels[channel])


def build_query_embedding_channels(
    input_text: str,
    association_text: str = "",
    *,
    query_plan: QueryPlan | None = None,
) -> dict[str, str]:
    positive_text = query_plan.positive_query if query_plan is not None else input_text.strip()
    visual_hints = " ".join(query_plan.visual_hints) if query_plan is not None else ""
    desired_stage = " ".join(query_plan.desired_stage) if query_plan is not None else ""
    positive_purposes = " ".join(query_plan.positive_purposes) if query_plan is not None else ""
    positive_style = " ".join(query_plan.positive_style) if query_plan is not None else ""
    script_use = _clean_join(
        [
            positive_text,
            f"desired stage {desired_stage}",
            f"creative purpose {positive_purposes}",
            f"positive style {positive_style}",
        ]
    )
    experience = _clean_join([positive_text, association_text])
    visual_tags = _clean_join([visual_hints or positive_text])
    combined = _clean_join([script_use, experience, visual_tags])
    return {
        "script_use": script_use,
        "experience": experience,
        "visual_tags": visual_tags,
        "combined": combined,
    }


def build_card_embedding_channels(card: ExperienceCard) -> dict[str, str]:
    usecase = card.script_usecase
    tag_text = _tag_text(card)
    keyword_text = " ".join(card.keywords)
    script_use = _clean_join(
        [
            keyword_text,
            usecase.script_stage,
            " ".join(usecase.creative_purpose),
            " ".join(card.style_traits),
            usecase.best_usage,
            card.reuse_condition,
        ]
    )
    experience = _clean_join(
        [
            keyword_text,
            card.underlying_emotion,
            card.narrative_logic,
            card.director_strategy,
            " ".join(card.shooting_techniques),
            card.copywriting_tone,
            " ".join(card.style_traits),
            usecase.risk,
        ]
    )
    visual_tags = _clean_join(
        [
            keyword_text,
            " ".join(card.visual_symbols),
            tag_text,
        ]
    )
    combined = _clean_join([script_use, experience, visual_tags])
    return {
        "script_use": script_use,
        "experience": experience,
        "visual_tags": visual_tags,
        "combined": combined,
    }


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_scores(
    query_text: str,
    cards: Sequence[ExperienceCard],
    *,
    backend: EmbeddingBackend,
) -> list[float]:
    texts = [query_text, *(build_card_embedding_text(card) for card in cards)]
    embeddings = backend.encode(texts)
    if len(embeddings) != len(cards) + 1:
        raise ValueError("embedding backend returned an unexpected number of vectors")
    query_embedding = embeddings[0]
    return [cosine_similarity(query_embedding, card_embedding) for card_embedding in embeddings[1:]]


def semantic_channel_scores(
    input_text: str,
    cards: Sequence[ExperienceCard],
    *,
    backend: EmbeddingBackend,
    association_text: str = "",
    query_plan: QueryPlan | None = None,
    channel_weights: dict[str, float] | None = None,
) -> list[float]:
    weights = channel_weights or DEFAULT_SEMANTIC_CHANNEL_WEIGHTS
    query_channels = build_query_embedding_channels(input_text, association_text, query_plan=query_plan)
    card_channels = [build_card_embedding_channels(card) for card in cards]
    texts: list[str] = []
    for channel in weights:
        texts.append(query_channels[channel])
        for channels in card_channels:
            texts.append(channels[channel])
    embeddings = backend.encode(texts)
    expected_count = len(weights) * (len(cards) + 1)
    if len(embeddings) != expected_count:
        raise ValueError("embedding backend returned an unexpected number of vectors")

    scores = [0.0 for _card in cards]
    offset = 0
    for channel, weight in weights.items():
        query_embedding = embeddings[offset]
        offset += 1
        for card_index in range(len(cards)):
            card_embedding = embeddings[offset]
            offset += 1
            scores[card_index] += weight * cosine_similarity(query_embedding, card_embedding)
    return scores


def _tag_text(card: ExperienceCard) -> str:
    parts: list[str] = []
    for value in card.tags.model_dump(mode="json").values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value if isinstance(item, str))
    return " ".join(parts)


def _clean_join(parts: list[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())
