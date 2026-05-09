from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1800
    request_timeout_seconds: float = 180.0
    enable_thinking: bool | None = None
    thinking_budget: int | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # VIDEO_ANALYZER_* aliases keep reuse convenient in the existing conda env.
        api_key = os.environ.get("SCENEWEAVER_API_KEY") or os.environ.get("VIDEO_ANALYZER_API_KEY", "")
        base_url = os.environ.get("SCENEWEAVER_BASE_URL") or os.environ.get(
            "VIDEO_ANALYZER_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        model = os.environ.get("SCENEWEAVER_MODEL") or os.environ.get("VIDEO_ANALYZER_MODEL", "gemini-2.0-flash")
        max_tokens = int(os.environ.get("SCENEWEAVER_MAX_TOKENS") or os.environ.get("VIDEO_ANALYZER_MAX_TOKENS", "1800"))
        request_timeout_seconds = float(
            os.environ.get("SCENEWEAVER_TIMEOUT_SECONDS") or os.environ.get("VIDEO_ANALYZER_TIMEOUT_SECONDS", "180")
        )
        enable_thinking = _parse_optional_bool(
            os.environ.get("SCENEWEAVER_ENABLE_THINKING") or os.environ.get("VIDEO_ANALYZER_ENABLE_THINKING")
        )
        thinking_budget_raw = os.environ.get("SCENEWEAVER_THINKING_BUDGET") or os.environ.get(
            "VIDEO_ANALYZER_THINKING_BUDGET"
        )
        thinking_budget = int(thinking_budget_raw) if thinking_budget_raw else None
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            request_timeout_seconds=request_timeout_seconds,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )


class VisionLLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

    def analyze_text_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        retries: int = 0,
        stream_callback: Callable[[str], None] | None = None,
        reasoning_callback: Callable[[str], None] | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> dict[str, Any]:
        try:
            from openai import APIConnectionError
            from openai import APIStatusError
            from openai import APITimeoutError
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for API mode") from exc

        if not self.config.api_key:
            raise RuntimeError("SCENEWEAVER_API_KEY or VIDEO_ANALYZER_API_KEY is required")
        if retries < 0:
            raise ValueError("retries must be >= 0")

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=timeout_seconds or self.config.request_timeout_seconds,
            max_retries=0,
        )
        request_enable_thinking = self.config.enable_thinking if enable_thinking is None else enable_thinking
        request_thinking_budget = self.config.thinking_budget if thinking_budget is None else thinking_budget
        extra_body: dict[str, Any] = {}
        if request_enable_thinking is not None:
            extra_body["enable_thinking"] = request_enable_thinking
        if request_thinking_budget is not None:
            extra_body["thinking_budget"] = request_thinking_budget

        attempts = retries + 1
        for attempt_index in range(attempts):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "stream": stream_callback is not None or reasoning_callback is not None,
                }
                if extra_body:
                    request_kwargs["extra_body"] = extra_body

                response = client.chat.completions.create(
                    **request_kwargs,
                )
                if stream_callback is None and reasoning_callback is None:
                    text = response.choices[0].message.content or ""
                else:
                    text = _collect_stream_text(response, stream_callback, reasoning_callback=reasoning_callback)
                return extract_json_object(text)
            except APIStatusError as exc:
                if _should_retry_status(exc) and attempt_index < retries:
                    time.sleep(_retry_delay_seconds(attempt_index))
                    continue
                raise RuntimeError(_format_api_status_error(exc, self.config)) from exc
            except PartialStreamError as exc:
                if attempt_index < retries:
                    time.sleep(_retry_delay_seconds(attempt_index))
                    continue
                partial_note = ""
                if exc.partial_text:
                    partial_note = f" partial_chars={len(exc.partial_text)}."
                raise RuntimeError(
                    "LLM text JSON stream failed before a complete JSON object could be parsed. "
                    f"attempts={attempts}, base_url={self.config.base_url!r}, "
                    f"model={self.config.model!r}.{partial_note} Error: {exc.original_error}"
                ) from exc.original_error
            except (APIConnectionError, APITimeoutError, ValueError) as exc:
                if attempt_index < retries:
                    time.sleep(_retry_delay_seconds(attempt_index))
                    continue
                raise RuntimeError(
                    "LLM text JSON request failed. "
                    f"attempts={attempts}, base_url={self.config.base_url!r}, "
                    f"model={self.config.model!r}. Error: {exc}"
                ) from exc

    def analyze_images_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> dict[str, Any]:
        try:
            from openai import APIStatusError
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for API mode") from exc

        if not self.config.api_key:
            raise RuntimeError("SCENEWEAVER_API_KEY or VIDEO_ANALYZER_API_KEY is required")

        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image_path)},
                }
            )

        try:
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except APIStatusError as exc:
            raise RuntimeError(_format_api_status_error(exc, self.config)) from exc
        text = response.choices[0].message.content or ""
        return extract_json_object(text)


class PartialStreamError(RuntimeError):
    def __init__(self, partial_text: str, original_error: Exception) -> None:
        super().__init__(str(original_error))
        self.partial_text = partial_text
        self.original_error = original_error


def _collect_stream_text(
    response: Iterable[Any],
    stream_callback: Callable[[str], None] | None = None,
    *,
    reasoning_callback: Callable[[str], None] | None = None,
) -> str:
    chunks: list[str] = []
    try:
        for chunk in response:
            reasoning_delta = _stream_chunk_reasoning_delta(chunk)
            if reasoning_delta and reasoning_callback is not None:
                reasoning_callback(reasoning_delta)
            delta = _stream_chunk_delta(chunk)
            if not delta:
                continue
            chunks.append(delta)
            if stream_callback is not None:
                stream_callback(delta)
    except Exception as exc:
        partial_text = "".join(chunks)
        if partial_text:
            try:
                extract_json_object(partial_text)
            except ValueError:
                pass
            else:
                return partial_text
        raise PartialStreamError(partial_text, exc) from exc
    return "".join(chunks)


def _stream_chunk_delta(chunk: Any) -> str:
    try:
        choices = getattr(chunk, "choices", None)
    except (IndexError, TypeError, AttributeError):
        return ""
    if not choices:
        return ""
    try:
        choice = choices[0]
    except (IndexError, TypeError, AttributeError):
        return ""

    delta = getattr(choice, "delta", None)
    content = getattr(delta, "content", None)
    if content is None and isinstance(delta, dict):
        content = delta.get("content")
    return content if isinstance(content, str) else ""


def _stream_chunk_reasoning_delta(chunk: Any) -> str:
    try:
        choices = getattr(chunk, "choices", None)
    except (IndexError, TypeError, AttributeError):
        return ""
    if not choices:
        return ""
    try:
        choice = choices[0]
    except (IndexError, TypeError, AttributeError):
        return ""

    delta = getattr(choice, "delta", None)
    reasoning_content = getattr(delta, "reasoning_content", None)
    if reasoning_content is None and isinstance(delta, dict):
        reasoning_content = delta.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else ""


def _image_data_url(path: Path) -> str:
    mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_data}"


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(match.group())


def _format_api_status_error(exc: Exception, config: LLMConfig) -> str:
    status_code = getattr(exc, "status_code", "unknown")
    response = getattr(exc, "response", None)
    body = None
    if response is not None:
        try:
            body = response.json()
        except Exception:
            body = None

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            code = str(error.get("code") or "").strip()
            if "FreeTierOnly" in code or "free tier" in message.lower():
                return (
                    "LLM provider rejected the request with 403 FreeTierOnly. "
                    f"base_url={config.base_url!r}, model={config.model!r}. "
                    "The current API key or provider console is restricted to free-tier routing, "
                    "and that quota has been exhausted. Disable free-tier-only mode, switch to paid routing, "
                    "or use another key/model with available quota, then retry."
                )
            if message:
                return (
                    f"LLM provider request failed with status {status_code}. "
                    f"base_url={config.base_url!r}, model={config.model!r}. "
                    f"Provider message: {message}"
                )

    return (
        f"LLM provider request failed with status {status_code}. "
        f"base_url={config.base_url!r}, model={config.model!r}."
    )


def _should_retry_status(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def _retry_delay_seconds(attempt_index: int) -> float:
    return min(2.0 * (attempt_index + 1), 10.0)


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")
