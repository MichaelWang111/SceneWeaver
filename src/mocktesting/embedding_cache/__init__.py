from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

DEFAULT_MODEL = "text_embedding_v4"
DEFAULT_DIMENSION = 1024
MAX_DASHSCOPE_BATCH_SIZE = 10
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "qwen_text_embedding_v4_1024.jsonl"


class EmbeddingCache:
    def __init__(
        self,
        *,
        cache_path: Path = DEFAULT_CACHE_PATH,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = MAX_DASHSCOPE_BATCH_SIZE,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.cache_path = cache_path
        self.model = model
        self.dimension = dimension
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = min(batch_size, MAX_DASHSCOPE_BATCH_SIZE)
        self.embedder = embedder or self._dashscope_embed
        self._rows = load_cache(cache_path)

    def key_for_text(self, text: str) -> str:
        return cache_key(text, model=self.model, dimension=self.dimension)

    def get(self, text: str) -> list[float] | None:
        row = self._rows.get(self.key_for_text(text))
        if row is None:
            return None
        return row["embedding"]

    def missing_texts(self, texts: list[str]) -> list[str]:
        missing: list[str] = []
        seen: set[str] = set()
        for text in texts:
            key = self.key_for_text(text)
            if key in self._rows or key in seen:
                continue
            seen.add(key)
            missing.append(text)
        return missing

    def embed_texts(self, texts: list[str], *, dry_run: bool = False) -> dict[str, int]:
        missing = self.missing_texts(texts)
        if dry_run:
            return {"requested": len(texts), "missing": len(missing), "written": 0}
        if not missing:
            return {"requested": len(texts), "missing": 0, "written": 0}
        vectors: list[list[float]] = []
        for start in range(0, len(missing), self.batch_size):
            vectors.extend(self.embedder(missing[start : start + self.batch_size]))
        if len(vectors) != len(missing):
            raise ValueError("embedder returned a different number of vectors")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            for text, vector in zip(missing, vectors):
                row = {
                    "key": self.key_for_text(text),
                    "model": self.model,
                    "dimension": self.dimension,
                    "text_sha256": text_sha256(text),
                    "text": text,
                    "embedding": vector,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._rows[row["key"]] = row
        return {"requested": len(texts), "missing": len(missing), "written": len(missing)}

    def require_embedding(self, text: str) -> list[float]:
        vector = self.get(text)
        if vector is None:
            raise KeyError(f"embedding missing for text key={self.key_for_text(text)}")
        return vector

    def _dashscope_embed(self, texts: list[str]) -> list[list[float]]:
        try:
            import dashscope
            from http import HTTPStatus
        except ImportError as exc:
            raise RuntimeError("dashscope package is required for qwen embedding") from exc

        api_key = (
            os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("SCENEWEAVER_API_KEY")
            or os.environ.get("VIDEO_ANALYZER_API_KEY")
        )
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY, SCENEWEAVER_API_KEY, or VIDEO_ANALYZER_API_KEY is required")

        dashscope.base_http_api_url = os.environ.get("DASHSCOPE_HTTP_API_URL", "https://dashscope.aliyuncs.com/api/v1")
        model = getattr(dashscope.TextEmbedding.Models, self.model, self.model)
        response = dashscope.TextEmbedding.call(
            api_key=api_key,
            model=model,
            input=texts,
            dimension=self.dimension,
            output_type="dense",
        )
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"dashscope embedding request failed: {response}")
        return [item["embedding"] for item in response.output["embeddings"]]


def load_cache(cache_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not cache_path.exists():
        return rows
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["key"]] = row
    return rows


def cache_key(text: str, *, model: str, dimension: int) -> str:
    payload = f"{model}|{dimension}|{text_sha256(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
