from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np

DEFAULT_MODEL = "text_embedding_v4"
DEFAULT_DIMENSION = 1024
MAX_DASHSCOPE_BATCH_SIZE = 10
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "qwen_text_embedding_v4_1024.jsonl"
DEFAULT_MATRIX_DTYPE = "float16"


class EmbeddingCache:
    def __init__(
        self,
        *,
        cache_path: Path = DEFAULT_CACHE_PATH,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = MAX_DASHSCOPE_BATCH_SIZE,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        load_all: bool = True,
        prefer_matrix: bool = False,
        matrix_path: Path | None = None,
        matrix_dtype: str = DEFAULT_MATRIX_DTYPE,
    ) -> None:
        self.cache_path = cache_path
        self.matrix_path = matrix_path or default_matrix_path(cache_path)
        self.model = model
        self.dimension = dimension
        self.prefer_matrix = prefer_matrix
        self.matrix_dtype = matrix_dtype
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = min(batch_size, MAX_DASHSCOPE_BATCH_SIZE)
        self.embedder = embedder or self._dashscope_embed
        self._loaded_all = load_all
        self._matrix_vectors: np.ndarray | None = None
        self._matrix_keys: list[str] = []
        self._matrix_key_to_index: dict[str, int] = {}
        self.stats = {
            "load_mode": "full" if load_all else "lazy",
            "loaded_rows": 0,
            "load_all_seconds": 0.0,
            "lazy_scan_count": 0,
            "lazy_scan_seconds": 0.0,
            "lazy_loaded_rows": 0,
            "matrix_enabled": prefer_matrix,
            "matrix_status": "disabled",
            "matrix_rows": 0,
            "matrix_load_seconds": 0.0,
            "matrix_saved_rows": 0,
            "matrix_save_seconds": 0.0,
        }
        self.last_embed_stats = {"requested": 0, "missing": 0, "written": 0}
        if prefer_matrix:
            self._load_matrix_if_available()
        if load_all and not self._matrix_key_to_index:
            started_at = time.perf_counter()
            self._rows = load_cache(cache_path)
            self.stats["loaded_rows"] = len(self._rows)
            self.stats["load_all_seconds"] = round(time.perf_counter() - started_at, 6)
        else:
            self._rows = {}

    def key_for_text(self, text: str) -> str:
        return cache_key(text, model=self.model, dimension=self.dimension)

    def get(self, text: str) -> list[float] | None:
        vector = self.get_array(text)
        if vector is None:
            return None
        return vector.astype(np.float32, copy=False).tolist()

    def get_array(self, text: str) -> np.ndarray | None:
        key = self.key_for_text(text)
        matrix_index = self._matrix_key_to_index.get(key)
        if matrix_index is not None and self._matrix_vectors is not None:
            return self._matrix_vectors[matrix_index].astype(np.float32, copy=False)
        if key not in self._rows and not self._loaded_all:
            self._load_needed_keys({key})
        row = self._rows.get(key)
        if row is None:
            return None
        return np.asarray(row["embedding"], dtype=np.float32)

    def missing_texts(self, texts: list[str]) -> list[str]:
        if not self._loaded_all:
            self._load_needed_keys(
                {
                    self.key_for_text(text)
                    for text in texts
                    if self.key_for_text(text) not in self._matrix_key_to_index
                }
            )
        missing: list[str] = []
        seen: set[str] = set()
        for text in texts:
            key = self.key_for_text(text)
            if key in self._matrix_key_to_index or key in self._rows or key in seen:
                continue
            seen.add(key)
            missing.append(text)
        return missing

    def cache_report(self) -> dict[str, int | float | str]:
        return {
            **self.stats,
            "resident_rows": len(self._rows),
            "cache_path": str(self.cache_path),
            "matrix_path": str(self.matrix_path),
            "matrix_dtype": self.matrix_dtype,
            "last_embed_requested": self.last_embed_stats["requested"],
            "last_embed_missing": self.last_embed_stats["missing"],
            "last_embed_written": self.last_embed_stats["written"],
        }

    def embed_texts(self, texts: list[str], *, dry_run: bool = False) -> dict[str, int]:
        missing = self.missing_texts(texts)
        if dry_run:
            self.last_embed_stats = {"requested": len(texts), "missing": len(missing), "written": 0}
            return dict(self.last_embed_stats)
        if not missing:
            self.last_embed_stats = {"requested": len(texts), "missing": 0, "written": 0}
            return dict(self.last_embed_stats)
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
        self.last_embed_stats = {"requested": len(texts), "missing": len(missing), "written": len(missing)}
        if self.prefer_matrix and self._matrix_vectors is not None and self.stats["matrix_status"] == "loaded":
            self._append_matrix_rows([self._rows[self.key_for_text(text)] for text in missing])
        return dict(self.last_embed_stats)

    def _load_matrix_if_available(self) -> None:
        self.stats["matrix_status"] = "missing"
        if not self.matrix_path.exists():
            return
        if self.cache_path.exists() and self.cache_path.stat().st_mtime > self.matrix_path.stat().st_mtime:
            self.stats["matrix_status"] = "stale"
            return
        started_at = time.perf_counter()
        try:
            with np.load(self.matrix_path, allow_pickle=False) as data:
                keys = [str(key) for key in data["keys"].tolist()]
                vectors = np.asarray(data["embeddings"])
        except Exception:
            self.stats["matrix_status"] = "invalid"
            return
        if vectors.ndim != 2 or (self.dimension and vectors.shape[1] != self.dimension):
            self.stats["matrix_status"] = "dimension_mismatch"
            return
        self._matrix_keys = keys
        self._matrix_key_to_index = {key: index for index, key in enumerate(keys)}
        self._matrix_vectors = vectors
        self.stats["matrix_status"] = "loaded"
        self.stats["matrix_rows"] = len(keys)
        self.stats["matrix_load_seconds"] = round(time.perf_counter() - started_at, 6)

    def _append_matrix_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        new_rows = [row for row in rows if row["key"] not in self._matrix_key_to_index]
        if not new_rows:
            return
        new_keys = [row["key"] for row in new_rows]
        new_vectors = np.asarray([row["embedding"] for row in new_rows], dtype=self.matrix_dtype)
        if self._matrix_vectors is None:
            vectors = new_vectors
            keys = new_keys
        else:
            vectors = np.vstack([self._matrix_vectors.astype(self.matrix_dtype, copy=False), new_vectors])
            keys = [*self._matrix_keys, *new_keys]
        save_seconds = save_matrix_cache(self.matrix_path, keys, vectors)
        self._matrix_keys = keys
        self._matrix_key_to_index = {key: index for index, key in enumerate(keys)}
        self._matrix_vectors = vectors
        self.stats["matrix_rows"] = len(keys)
        self.stats["matrix_saved_rows"] = len(keys)
        self.stats["matrix_save_seconds"] = save_seconds
        self.stats["matrix_status"] = "loaded"

    def _load_needed_keys(self, keys: set[str]) -> None:
        needed = keys - set(self._rows) - set(self._matrix_key_to_index)
        if not needed or not self.cache_path.exists():
            return
        started_at = time.perf_counter()
        loaded = 0
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key = _jsonl_line_key(line)
                if key not in needed:
                    continue
                row = json.loads(line)
                self._rows[key] = row
                loaded += 1
                needed.remove(key)
                if not needed:
                    break
        self.stats["lazy_scan_count"] += 1
        self.stats["lazy_scan_seconds"] = round(
            float(self.stats["lazy_scan_seconds"]) + time.perf_counter() - started_at,
            6,
        )
        self.stats["lazy_loaded_rows"] += loaded

    def require_embedding(self, text: str) -> list[float]:
        vector = self.get_array(text)
        if vector is None:
            raise KeyError(f"embedding missing for text key={self.key_for_text(text)}")
        return vector.astype(np.float32, copy=False).tolist()

    def require_embedding_array(self, text: str) -> np.ndarray:
        vector = self.get_array(text)
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


def default_matrix_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".npz")


def build_matrix_cache(
    cache_path: Path,
    *,
    matrix_path: Path | None = None,
    dtype: str = DEFAULT_MATRIX_DTYPE,
) -> dict[str, int | float | str]:
    started_at = time.perf_counter()
    rows = load_cache(cache_path)
    keys = list(rows)
    vectors = np.asarray([rows[key]["embedding"] for key in keys], dtype=dtype)
    output_path = matrix_path or default_matrix_path(cache_path)
    save_seconds = save_matrix_cache(output_path, keys, vectors)
    return {
        "cache_path": str(cache_path),
        "matrix_path": str(output_path),
        "rows": len(keys),
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "dtype": str(vectors.dtype),
        "jsonl_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
        "matrix_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "save_seconds": save_seconds,
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
    }


def save_matrix_cache(matrix_path: Path, keys: list[str], vectors: np.ndarray) -> float:
    started_at = time.perf_counter()
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        matrix_path,
        keys=np.asarray(keys),
        embeddings=vectors,
    )
    return round(time.perf_counter() - started_at, 6)


def _jsonl_line_key(line: str) -> str | None:
    marker = '"key": "'
    start = line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = line.find('"', start)
    if end < 0:
        return None
    return line[start:end]


def cache_key(text: str, *, model: str, dimension: int) -> str:
    payload = f"{model}|{dimension}|{text_sha256(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
