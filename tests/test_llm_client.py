from __future__ import annotations

import os

import pytest

from sceneweaver.llm.client import LLMConfig
from sceneweaver.llm.client import PartialStreamError, _collect_stream_text, extract_json_object, llm_config_metadata
from sceneweaver.llm.providers import infer_provider_from_base_url, infer_provider_from_model, provider_status, provider_status_dict


class _Delta:
    def __init__(self, content: str | None, reasoning_content: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, content: str | None, reasoning_content: str | None = None) -> None:
        self.delta = _Delta(content, reasoning_content)


class _Chunk:
    def __init__(self, choices) -> None:
        self.choices = choices


class _BadChoicesChunk:
    @property
    def choices(self):
        raise IndexError("list index out of range")


def test_collect_stream_text_ignores_empty_and_malformed_choice_chunks():
    streamed: list[str] = []
    response = [
        _Chunk([]),
        _BadChoicesChunk(),
        _Chunk([_Choice('{"ok":')]),
        _Chunk([_Choice("true}")]),
        _Chunk([_Choice(None)]),
    ]

    text = _collect_stream_text(response, streamed.append)

    assert text == '{"ok":true}'
    assert streamed == ['{"ok":', "true}"]


def test_collect_stream_text_uses_complete_json_when_stream_ends_with_error():
    streamed: list[str] = []

    def response():
        yield _Chunk([_Choice('{"ok":')])
        yield _Chunk([_Choice("true}")])
        raise TimeoutError("provider stream timed out")

    text = _collect_stream_text(response(), streamed.append)

    assert text == '{"ok":true}'
    assert streamed == ['{"ok":', "true}"]


def test_collect_stream_text_keeps_reasoning_separate_from_json_text():
    streamed: list[str] = []
    reasoning: list[str] = []
    response = [
        _Chunk([_Choice(None, "think ")]),
        _Chunk([_Choice(None, "first")]),
        _Chunk([_Choice('{"ok":')]),
        _Chunk([_Choice("true}")]),
    ]

    text = _collect_stream_text(response, streamed.append, reasoning_callback=reasoning.append)

    assert text == '{"ok":true}'
    assert streamed == ['{"ok":', "true}"]
    assert reasoning == ["think ", "first"]


def test_collect_stream_text_raises_partial_stream_error_for_incomplete_json():
    def response():
        yield _Chunk([_Choice('{"ok":')])
        raise TimeoutError("provider stream timed out")

    with pytest.raises(PartialStreamError) as exc_info:
        _collect_stream_text(response(), lambda _chunk: None)

    assert exc_info.value.partial_text == '{"ok":'
    assert isinstance(exc_info.value.original_error, TimeoutError)


def test_extract_json_object_uses_first_complete_object_with_extra_data():
    text = 'preface {"judgements":[{"grade":2}]} {"extra": true}'

    assert extract_json_object(text) == {"judgements": [{"grade": 2}]}


def test_llm_config_reads_dashscope_aliases(monkeypatch):
    monkeypatch.delenv("SCENEWEAVER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("VIDEO_ANALYZER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.delenv("VIDEO_ANALYZER_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen3.7-max")
    monkeypatch.setenv("DASHSCOPE_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("DASHSCOPE_STREAM_IDLE_TIMEOUT_SECONDS", "7")

    config = LLMConfig.from_env()

    assert config.api_key == "dash-key"
    assert config.provider == "dashscope"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.model == "qwen3.7-max"
    assert config.request_timeout_seconds == 42.0
    assert config.stream_idle_timeout_seconds == 7.0


def test_llm_config_reads_deepseek_provider_aliases(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("VIDEO_ANALYZER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = LLMConfig.from_env()

    assert config.api_key == "deepseek-key"
    assert config.provider == "deepseek"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"


def test_provider_inference_and_status_do_not_expose_key_values(monkeypatch):
    assert infer_provider_from_base_url("https://api.deepseek.com") == "deepseek"
    assert infer_provider_from_base_url("https://dashscope.aliyuncs.com/compatible-mode/v1") == "dashscope"
    assert infer_provider_from_model("deepseek-v4-pro") == "deepseek"
    assert infer_provider_from_model("qwen3.6-flash") == "dashscope"

    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "dashscope")
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-key-value")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen3.6-flash")
    status = provider_status(provider="auto", env=dict(os.environ), check_balance=False, live_models=False)
    data = provider_status_dict(status, include_models=False)

    assert data["provider"] == "dashscope"
    assert data["api_key"]["configured"] is True
    assert data["api_key"]["active_env_name"] == "DASHSCOPE_API_KEY"
    assert "secret-key-value" not in repr(data)
    assert data["model_count"] >= 1
    assert data["limits"]["rpm"] == 30000


def test_llm_config_metadata_reports_provider_limits_and_pricing(monkeypatch):
    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    config = LLMConfig.from_env()

    metadata = llm_config_metadata(config)

    assert metadata["provider"] == "deepseek"
    assert metadata["api_key_env"] == "DEEPSEEK_API_KEY"
    assert metadata["pricing"]["input_cny_per_million"] == 1.0
    assert metadata["limits"]["concurrency"] == 2500
