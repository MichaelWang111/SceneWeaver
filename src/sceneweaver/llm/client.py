from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 1800

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # VIDEO_ANALYZER_* aliases keep reuse convenient in the existing conda env.
        api_key = os.environ.get("SCENEWEAVER_API_KEY") or os.environ.get("VIDEO_ANALYZER_API_KEY", "")
        base_url = os.environ.get("SCENEWEAVER_BASE_URL") or os.environ.get(
            "VIDEO_ANALYZER_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        model = os.environ.get("SCENEWEAVER_MODEL") or os.environ.get("VIDEO_ANALYZER_MODEL", "gemini-2.0-flash")
        return cls(api_key=api_key, base_url=base_url, model=model)


class VisionLLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

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
