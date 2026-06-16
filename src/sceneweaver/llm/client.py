from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from sceneweaver.llm.providers import (
    api_key_info,
    infer_provider_from_env,
    model_limits,
    model_pricing,
    model_pricing_dict,
    normalize_provider,
    provider_limits_dict,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)


@dataclass(frozen=True)
class LLMTextJSONResult:
    data: dict[str, Any]
    usage: dict[str, Any]
    request_id: str | None = None
    raw_usage: dict[str, Any] | None = None
    model: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    provider: str = "auto"
    temperature: float = 0.2
    max_tokens: int = 1800
    request_timeout_seconds: float = 180.0
    stream_idle_timeout_seconds: float = 10.0
    enable_thinking: bool | None = None
    thinking_budget: int | None = None
    enable_failure_ping: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = infer_provider_from_env()
        api_key = resolve_api_key(provider)
        base_url = resolve_base_url(provider)
        model = resolve_model(provider)
        max_tokens = int(
            os.environ.get("SCENEWEAVER_MAX_TOKENS")
            or os.environ.get("VIDEO_ANALYZER_MAX_TOKENS")
            or os.environ.get("DASHSCOPE_MAX_TOKENS", "1800")
        )
        request_timeout_seconds = float(
            os.environ.get("SCENEWEAVER_TIMEOUT_SECONDS")
            or os.environ.get("VIDEO_ANALYZER_TIMEOUT_SECONDS")
            or os.environ.get("DASHSCOPE_TIMEOUT_SECONDS", "180")
        )
        stream_idle_timeout_seconds = float(
            os.environ.get("SCENEWEAVER_STREAM_IDLE_TIMEOUT_SECONDS")
            or os.environ.get("VIDEO_ANALYZER_STREAM_IDLE_TIMEOUT_SECONDS")
            or os.environ.get("DASHSCOPE_STREAM_IDLE_TIMEOUT_SECONDS", "10")
        )
        enable_thinking = _parse_optional_bool(
            os.environ.get("SCENEWEAVER_ENABLE_THINKING")
            or os.environ.get("VIDEO_ANALYZER_ENABLE_THINKING")
            or os.environ.get("DASHSCOPE_ENABLE_THINKING")
        )
        thinking_budget_raw = (
            os.environ.get("SCENEWEAVER_THINKING_BUDGET")
            or os.environ.get("VIDEO_ANALYZER_THINKING_BUDGET")
            or os.environ.get("DASHSCOPE_THINKING_BUDGET")
        )
        thinking_budget = int(thinking_budget_raw) if thinking_budget_raw else None
        enable_failure_ping = _parse_optional_bool(
            os.environ.get("SCENEWEAVER_ENABLE_FAILURE_PING")
            or os.environ.get("VIDEO_ANALYZER_ENABLE_FAILURE_PING")
            or os.environ.get("DASHSCOPE_ENABLE_FAILURE_PING")
        )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            request_timeout_seconds=request_timeout_seconds,
            stream_idle_timeout_seconds=stream_idle_timeout_seconds,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            enable_failure_ping=bool(enable_failure_ping),
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
        return self.analyze_text_json_result(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
            stream_callback=stream_callback,
            reasoning_callback=reasoning_callback,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        ).data

    def analyze_text_json_result(
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
    ) -> LLMTextJSONResult:
        try:
            from openai import APIConnectionError
            from openai import APIStatusError
            from openai import APITimeoutError
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for API mode") from exc

        if not self.config.api_key:
            raise RuntimeError(_missing_api_key_message(self.config))
        if retries < 0:
            raise ValueError("retries must be >= 0")

        should_stream = stream_callback is not None or reasoning_callback is not None
        request_timeout = timeout_seconds or self.config.request_timeout_seconds
        sdk_timeout = self.config.stream_idle_timeout_seconds if should_stream else request_timeout

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=sdk_timeout,
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
                    "stream": should_stream,
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
                usage = _response_usage_dict(response) if not should_stream else {}
                return LLMTextJSONResult(
                    data=extract_json_object(text),
                    usage=usage,
                    request_id=_response_request_id(response) if not should_stream else None,
                    raw_usage=usage or None,
                    model=getattr(response, "model", None) if not should_stream else self.config.model,
                )
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
                ping_note = _failure_ping_note(client, self.config)
                raise RuntimeError(
                    "LLM text JSON stream failed before a complete JSON object could be parsed. "
                    f"attempts={attempts}, base_url={self.config.base_url!r}, "
                    f"model={self.config.model!r}.{partial_note} "
                    f"stream_idle_timeout_seconds={self.config.stream_idle_timeout_seconds:g}. "
                    f"{ping_note} Error: {exc.original_error}"
                ) from exc.original_error
            except (APIConnectionError, APITimeoutError, ValueError) as exc:
                if attempt_index < retries:
                    time.sleep(_retry_delay_seconds(attempt_index))
                    continue
                ping_note = _failure_ping_note(client, self.config)
                raise RuntimeError(
                    "LLM text JSON request failed. "
                    f"attempts={attempts}, base_url={self.config.base_url!r}, "
                    f"model={self.config.model!r}. {ping_note} Error: {exc}"
                ) from exc

    def analyze_images_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> dict[str, Any]:
        try:
            from openai import APIStatusError
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for API mode") from exc

        if not self.config.api_key:
            raise RuntimeError(_missing_api_key_message(self.config))

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


def llm_config_metadata(config: LLMConfig) -> dict[str, Any]:
    provider = normalize_provider(getattr(config, "provider", "auto") or "auto")
    if provider == "auto":
        provider = infer_provider_from_env()
    model = str(getattr(config, "model", ""))
    base_url = str(getattr(config, "base_url", ""))
    pricing = model_pricing(provider, model, required=False)
    limits = model_limits(provider, model)
    key_info = api_key_info(provider)
    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": key_info.active_env_name,
        "pricing": model_pricing_dict(pricing),
        "limits": provider_limits_dict(limits),
    }


def _missing_api_key_message(config: LLMConfig) -> str:
    provider = normalize_provider(config.provider or "auto")
    if provider == "auto":
        provider = infer_provider_from_env()
    info = api_key_info(provider)
    names = ", ".join(info.accepted_env_names) or "SCENEWEAVER_API_KEY"
    return f"LLM provider {provider!r} requires one of these API key env vars: {names}"


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
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("LLM response did not contain a JSON object")


def _response_usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, dict) else {}
    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def _response_request_id(response: Any) -> str | None:
    value = getattr(response, "id", None)
    return value if isinstance(value, str) else None


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


def _failure_ping_note(client: Any, config: LLMConfig) -> str:
    if config.enable_failure_ping:
        return _format_ping_note(client, config)
    return "Failure ping disabled; set SCENEWEAVER_ENABLE_FAILURE_PING=1 to run an extra diagnostic LLM call."


def _format_ping_note(client: Any, config: LLMConfig) -> str:
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": 'Return {"reply":"pong"} now.'},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=64,
            timeout=config.stream_idle_timeout_seconds,
        )
        text = response.choices[0].message.content or ""
        data = extract_json_object(text)
    except Exception as exc:
        return f"Ping failed: {type(exc).__name__}: {exc}."
    return f"Ping ok: {data}."


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
