from __future__ import annotations

from dataclasses import dataclass

from sceneweaver.llm.providers import infer_provider_from_env, model_limits, normalize_provider


@dataclass(frozen=True)
class LLMRunOptions:
    concurrency: int = 1
    timeout_seconds: float | None = None
    retries: int = 0
    enable_thinking: bool | None = None
    thinking_budget: int | None = None

    def validate(self) -> "LLMRunOptions":
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.retries < 0:
            raise ValueError("retries must be >= 0")
        if self.thinking_budget is not None and self.thinking_budget < 0:
            raise ValueError("thinking_budget must be >= 0")
        return self


def effective_concurrency(requested: int, *, provider: str, model: str) -> int:
    if requested < 1:
        raise ValueError("concurrency must be >= 1")
    try:
        normalized = normalize_provider(provider or "auto")
    except Exception:
        return requested
    if normalized == "auto":
        normalized = infer_provider_from_env()
    limits = model_limits(normalized, model)
    if limits.concurrency is None or limits.concurrency < 1:
        return requested
    return max(1, min(requested, int(limits.concurrency)))


__all__ = ["LLMRunOptions", "effective_concurrency"]
