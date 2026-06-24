from __future__ import annotations

from dataclasses import replace
import os
from typing import Literal

from sceneweaver.llm.client import LLMConfig, VisionLLMClient
from sceneweaver.llm.providers import default_base_url, infer_provider_from_model, normalize_model_id, normalize_provider
from sceneweaver.llm.settings import config_for_role_settings

LLMRole = Literal["default", "scene_analysis", "script_generation", "retrieval_judge", "rerank"]

DEFAULT_ROLE_MODELS: dict[str, str] = {
    "scene_analysis": "qwen3.7-plus",
}


def config_for_role(*, role: LLMRole = "default", model: str | None = None, base_config: LLMConfig | None = None) -> LLMConfig:
    settings_config = config_for_role_settings(role=role, model=model, base_config=base_config)
    config = settings_config or base_config or LLMConfig.from_env()
    default_model = "" if settings_config is not None else DEFAULT_ROLE_MODELS.get(role, "")
    resolved_model = normalize_model_id(model or default_model)
    if resolved_model:
        config = replace(config, model=resolved_model)
    if resolved_model and infer_provider_from_model(resolved_model) == "dashscope":
        config = replace(
            config,
            provider="dashscope",
            api_key=_dashscope_api_key(config),
            base_url=_dashscope_base_url(config),
        )
    return config


def client_for_role(*, role: LLMRole = "default", model: str | None = None, base_config: LLMConfig | None = None) -> VisionLLMClient:
    return VisionLLMClient(config_for_role(role=role, model=model, base_config=base_config))


def _dashscope_api_key(config: LLMConfig) -> str:
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("VIDEO_ANALYZER_API_KEY")
        or (config.api_key if normalize_provider(config.provider) == "dashscope" else "")
    )


def _dashscope_base_url(config: LLMConfig) -> str:
    return (
        os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("VIDEO_ANALYZER_BASE_URL")
        or (config.base_url if normalize_provider(config.provider) == "dashscope" else default_base_url("dashscope"))
    )


__all__ = ["DEFAULT_ROLE_MODELS", "LLMRole", "client_for_role", "config_for_role"]
