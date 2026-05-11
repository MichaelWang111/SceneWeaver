from __future__ import annotations

from math import sqrt
from typing import Protocol, Sequence

from sceneweaver.schemas import ExperienceCard

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_SEMANTIC_WEIGHT = 4.0


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


def build_query_embedding_text(input_text: str, association_text: str = "") -> str:
    return "\n".join(part for part in (input_text.strip(), association_text.strip()) if part)


def build_card_embedding_text(card: ExperienceCard) -> str:
    return "\n".join(
        part
        for part in (
            " ".join(card.keywords),
            card.underlying_emotion,
            card.narrative_logic,
            card.director_strategy,
            " ".join(card.shooting_techniques),
            " ".join(card.visual_symbols),
            card.copywriting_tone,
            card.reuse_condition,
        )
        if part.strip()
    )


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
