from __future__ import annotations

import json
import os
import sys
import types

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


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _ResponseChoice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_ResponseChoice(content)]
        self.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.id = "response-id"
        self.model = "mock-model"


class _MockCompletions:
    calls = 0
    last_kwargs = None

    def create(self, **kwargs):
        type(self).calls += 1
        type(self).last_kwargs = kwargs
        if type(self).calls == 1:
            return _Response("not json")
        return _Response('{"reply":"pong"}')


class _MockChat:
    def __init__(self) -> None:
        self.completions = _MockCompletions()


class _MockOpenAI:
    init_kwargs = []

    def __init__(self, **kwargs) -> None:
        type(self).init_kwargs.append(kwargs)
        self.chat = _MockChat()


def _install_mock_openai(monkeypatch) -> None:
    _MockCompletions.calls = 0
    _MockCompletions.last_kwargs = None
    _MockOpenAI.init_kwargs = []
    module = types.SimpleNamespace(
        OpenAI=_MockOpenAI,
        APIConnectionError=RuntimeError,
        APIStatusError=RuntimeError,
        APITimeoutError=TimeoutError,
    )
    monkeypatch.setitem(sys.modules, "openai", module)


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


def test_llm_client_does_not_ping_on_parse_failure_by_default(monkeypatch):
    from sceneweaver.llm.client import VisionLLMClient

    _install_mock_openai(monkeypatch)
    config = LLMConfig(api_key="key", base_url="https://example.test", model="mock-model")

    with pytest.raises(RuntimeError) as exc_info:
        VisionLLMClient(config).analyze_text_json_result(system_prompt="", user_prompt="")

    assert _MockCompletions.calls == 1
    assert "Failure ping disabled" in str(exc_info.value)


def test_llm_client_ping_on_parse_failure_is_explicit(monkeypatch):
    from sceneweaver.llm.client import VisionLLMClient

    _install_mock_openai(monkeypatch)
    config = LLMConfig(
        api_key="key",
        base_url="https://example.test",
        model="mock-model",
        enable_failure_ping=True,
    )

    with pytest.raises(RuntimeError) as exc_info:
        VisionLLMClient(config).analyze_text_json_result(system_prompt="", user_prompt="")

    assert _MockCompletions.calls == 2
    assert "Ping ok" in str(exc_info.value)


def test_llm_client_image_json_uses_timeout_and_retries(monkeypatch, tmp_path):
    from sceneweaver.llm.client import VisionLLMClient

    _install_mock_openai(monkeypatch)
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image")
    image_path_2 = tmp_path / "frame_2.jpg"
    image_path_2.write_bytes(b"fake image 2")
    image_path_3 = tmp_path / "frame_3.jpg"
    image_path_3.write_bytes(b"fake image 3")
    config = LLMConfig(api_key="key", base_url="https://example.test", model="mock-model")

    result = VisionLLMClient(config).analyze_images_json(
        system_prompt="return json",
        user_prompt="look",
        image_paths=[image_path, image_path_2, image_path_3],
        image_labels=["start", "middle", "end"],
        timeout_seconds=12,
        retries=1,
    )

    assert result == {"reply": "pong"}
    assert _MockCompletions.calls == 2
    assert _MockOpenAI.init_kwargs[0]["timeout"] == 12
    assert _MockOpenAI.init_kwargs[0]["max_retries"] == 0
    content = _MockCompletions.last_kwargs["messages"][1]["content"]
    text_items = [item["text"] for item in content if item["type"] == "text"]
    assert text_items[1] == "Scene frame triplet: treat the following 3 images as one scene input, ordered as start, middle, end. Analyze them together as a single scene."
    assert text_items[2:] == ["start frame of the same scene", "middle frame of the same scene", "end frame of the same scene"]
    assert len([item for item in content if item["type"] == "image_url"]) == 3


def test_llm_client_image_json_leaves_generic_images_unlabeled(monkeypatch, tmp_path):
    from sceneweaver.llm.client import VisionLLMClient

    _install_mock_openai(monkeypatch)
    _MockCompletions.calls = 1
    image_path = tmp_path / "ping.jpg"
    image_path.write_bytes(b"fake image")
    config = LLMConfig(api_key="key", base_url="https://example.test", model="mock-model")

    result = VisionLLMClient(config).analyze_images_json(
        system_prompt="return json",
        user_prompt="look",
        image_paths=[image_path],
    )

    assert result == {"reply": "pong"}
    content = _MockCompletions.last_kwargs["messages"][1]["content"]
    text_items = [item["text"] for item in content if item["type"] == "text"]
    assert text_items == ["look"]
    assert len([item for item in content if item["type"] == "image_url"]) == 1


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


def test_llm_config_defaults_dashscope_to_qwen_plus(monkeypatch):
    monkeypatch.delenv("SCENEWEAVER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    monkeypatch.delenv("VIDEO_ANALYZER_API_KEY", raising=False)
    monkeypatch.delenv("VIDEO_ANALYZER_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")

    config = LLMConfig.from_env()

    assert config.provider == "dashscope"
    assert config.model == "qwen3.6-plus"


def test_llm_config_normalizes_qwen_plus_alias(monkeypatch):
    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "dashscope")
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-3.6plus")

    config = LLMConfig.from_env()

    assert config.model == "qwen3.6-plus"


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
    assert infer_provider_from_model("qwen-3.6plus") == "dashscope"

    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "dashscope")
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-key-value")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-3.6plus")
    status = provider_status(provider="auto", env=dict(os.environ), check_balance=False, live_models=False)
    data = provider_status_dict(status, include_models=False)

    assert data["provider"] == "dashscope"
    assert data["api_key"]["configured"] is True
    assert data["api_key"]["active_env_name"] == "DASHSCOPE_API_KEY"
    assert data["configured_model"] == "qwen3.6-plus"
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


def test_llm_profile_scene_analysis_forces_dashscope_for_qwen(monkeypatch, tmp_path):
    from sceneweaver.llm import client_for_role, config_for_role

    monkeypatch.setattr("sceneweaver.llm.settings.DEFAULT_LLM_CONFIG_PATH", tmp_path / "llm_config.json")
    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")

    config = config_for_role(role="scene_analysis")
    client = client_for_role(role="scene_analysis")

    assert config.provider == "dashscope"
    assert config.api_key == "dashscope-key"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.model == "qwen3.7-plus"
    assert client.config == config


def test_llm_status_payload_uses_sceneweaver_method_and_hides_key(monkeypatch):
    from sceneweaver.llm import llm_status_payload

    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    payload = llm_status_payload(provider="auto", include_models=False)

    assert payload["method"] == "sceneweaver_llm_provider_status"
    assert payload["summary"]["provider"] == "deepseek"
    assert payload["summary"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert "deepseek-secret" not in repr(payload)


def test_llm_settings_save_masks_key_and_configures_roles(monkeypatch, tmp_path):
    from sceneweaver.llm.settings import config_for_role_settings, llm_settings_payload, save_llm_settings

    config_path = tmp_path / "llm_config.json"
    save_llm_settings(
        {
            "profiles": {
                "basic": {
                    "mode": "preset",
                    "scheme": "deepseek",
                    "provider": "deepseek",
                    "api_key": "basic-secret-value",
                    "model": "deepseek-v4-flash",
                    "models": ["deepseek-v4-flash", "custom-basic-model"],
                    "base_url": "https://api.deepseek.com",
                },
                "vision": {
                    "mode": "preset",
                    "scheme": "dashscope",
                    "provider": "dashscope",
                    "api_key": "vision-secret-value",
                    "model": "qwen3.7-plus",
                    "models": ["qwen3.7-plus", "custom-vision-model"],
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                },
            }
        },
        config_path,
    )

    payload = llm_settings_payload(config_path)
    assert payload["profiles"]["basic"]["api_key_configured"] is True
    assert payload["profiles"]["basic"]["api_key_mask"] == "basi...alue"
    assert "custom-basic-model" in payload["profiles"]["basic"]["models"]
    assert "custom-vision-model" in payload["profiles"]["vision"]["models"]
    assert payload["profiles"]["basic"]["mode"] == "preset"
    assert payload["profiles"]["vision"]["scheme"] == "dashscope"
    assert "basic-secret-value" not in repr(payload)
    assert "vision-secret-value" not in repr(payload)

    basic_config = config_for_role_settings(role="script_generation", path=config_path)
    vision_config = config_for_role_settings(role="scene_analysis", path=config_path)
    assert basic_config is not None
    assert vision_config is not None
    assert basic_config.provider == "deepseek"
    assert basic_config.model == "deepseek-v4-flash"
    assert basic_config.api_key == "basic-secret-value"
    assert vision_config.provider == "dashscope"
    assert vision_config.model == "qwen3.7-plus"
    assert vision_config.api_key == "vision-secret-value"


def test_llm_settings_accepts_custom_provider_and_metadata(tmp_path):
    from sceneweaver.llm.runtime import effective_concurrency
    from sceneweaver.llm.settings import config_for_role_settings, llm_settings_payload, save_llm_settings

    config_path = tmp_path / "llm_config.json"
    save_llm_settings(
        {
            "profiles": {
                "basic": {
                    "mode": "custom",
                    "scheme": "custom",
                    "provider": "openrouter",
                    "providers": ["auto", "openrouter"],
                    "api_key": "custom-secret",
                    "model": "openai/gpt-4.1-mini",
                    "models": ["openai/gpt-4.1-mini"],
                    "base_url": "https://openrouter.ai/api/v1",
                }
            }
        },
        config_path,
    )

    payload = llm_settings_payload(config_path)
    config = config_for_role_settings(role="script_generation", path=config_path)
    assert payload["profiles"]["basic"]["mode"] == "custom"
    assert payload["profiles"]["basic"]["scheme"] == "custom"
    assert "openrouter" in payload["profiles"]["basic"]["providers"]
    assert config is not None
    assert config.provider == "openrouter"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.model == "openai/gpt-4.1-mini"
    metadata = llm_config_metadata(config)
    assert metadata["provider"] == "openrouter"
    assert metadata["pricing"] is None
    assert metadata["limits"] is None
    assert effective_concurrency(4, provider="openrouter", model="openai/gpt-4.1-mini") == 4


def test_llm_settings_default_provider_scheme_sets_base_url_and_allows_custom_model(tmp_path):
    from sceneweaver.llm.settings import config_for_role_settings, llm_settings_payload, save_llm_settings

    config_path = tmp_path / "llm_config.json"
    save_llm_settings(
        {
            "profiles": {
                "basic": {
                    "mode": "preset",
                    "scheme": "minimax",
                    "api_key": "minimax-secret",
                    "model": "MiniMax-Text-01-custom-preview",
                    "base_url": "https://wrong.example/v1",
                }
            }
        },
        config_path,
    )

    payload = llm_settings_payload(config_path)
    config = config_for_role_settings(role="script_generation", path=config_path)
    assert payload["provider_schemes"]
    assert payload["profiles"]["basic"]["mode"] == "preset"
    assert payload["profiles"]["basic"]["scheme"] == "minimax"
    assert payload["profiles"]["basic"]["provider"] == "minimax"
    assert payload["profiles"]["basic"]["base_url"] == "https://api.minimax.chat/v1"
    assert "MiniMax-Text-01-custom-preview" in payload["profiles"]["basic"]["models"]
    assert config is not None
    assert config.provider == "minimax"
    assert config.base_url == "https://api.minimax.chat/v1"
    assert config.model == "MiniMax-Text-01-custom-preview"


def test_llm_settings_treats_legacy_unknown_base_url_as_custom(tmp_path):
    from sceneweaver.llm.settings import config_for_role_settings, llm_settings_payload

    config_path = tmp_path / "llm_config.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "basic": {
                        "provider": "auto",
                        "api_key": "legacy-secret",
                        "model": "deepseek-v4-pro",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                        "models": ["deepseek-v4-pro"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    payload = llm_settings_payload(config_path)
    config = config_for_role_settings(role="script_generation", path=config_path)
    assert payload["profiles"]["basic"]["mode"] == "custom"
    assert payload["profiles"]["basic"]["scheme"] == "custom"
    assert payload["profiles"]["basic"]["provider"] == "auto"
    assert payload["profiles"]["basic"]["base_url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert config is not None
    assert config.base_url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert config.api_key == "legacy-secret"


def test_llm_settings_can_clear_key_without_dropping_other_profiles(tmp_path):
    from sceneweaver.llm.settings import llm_settings_payload, save_llm_settings

    config_path = tmp_path / "llm_config.json"
    save_llm_settings(
        {
            "profiles": {
                "basic": {"provider": "deepseek", "api_key": "basic-secret", "model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com"},
                "vision": {"provider": "dashscope", "api_key": "vision-secret", "model": "qwen3.7-plus", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
            }
        },
        config_path,
    )
    save_llm_settings({"profiles": {"basic": {"provider": "deepseek", "model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com", "clear_api_key": True}}}, config_path)

    payload = llm_settings_payload(config_path, include_secrets=True)
    assert payload["profiles"]["basic"]["api_key"] == ""
    assert payload["profiles"]["vision"]["api_key"] == "vision-secret"


def test_vision_ping_attaches_white_png(monkeypatch, tmp_path):
    from sceneweaver.llm.settings import ping_llm_profile

    calls = {}
    stale_ping_image = tmp_path / "llm_ping_pixel.png"
    stale_ping_image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c49444154789c63f8ffff3f0005fe02fea7c93da70000000049454e44ae426082"
        )
    )

    class FakeVisionClient:
        def __init__(self, config):
            self.config = config

        def analyze_images_json(self, **kwargs):
            calls["image"] = kwargs
            return {"reply": "pong", "vision": True}

        def analyze_text_json_result(self, **_kwargs):
            raise AssertionError("vision ping must use an image request")

    monkeypatch.setattr("sceneweaver.llm.settings.VisionLLMClient", FakeVisionClient)

    result = ping_llm_profile(
        "vision",
        profile_data={
            "mode": "custom",
            "scheme": "custom",
            "provider": "dashscope",
            "api_key": "vision-key",
            "model": "custom-vision",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        path=tmp_path / "llm_config.json",
    )

    image_path = calls["image"]["image_paths"][0]
    assert image_path.suffix == ".png"
    assert image_path.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
    image_data = image_path.read_bytes()
    width = int.from_bytes(image_data[16:20], "big")
    height = int.from_bytes(image_data[20:24], "big")
    assert width > 10
    assert height > 10
    assert result["ping_mode"] == "image"
    assert result["image_attached"] is True
    assert result["image_path"].endswith("llm_ping_pixel.png")
    assert result["response"] == {"reply": "pong", "vision": True}


def test_retrieval_lab_budget_guard_path_stays_compatible():
    from retrieval_lab.llm.budget_guard import DEFAULT_LLM_USAGE_LEDGER as retrieval_lab_ledger
    from sceneweaver.llm import DEFAULT_LLM_USAGE_LEDGER as sceneweaver_ledger

    assert str(sceneweaver_ledger) == str(os.path.join(".tmp", "sceneweaver", "llm_usage_ledger.jsonl"))
    assert str(retrieval_lab_ledger) == str(os.path.join(".tmp", "retrieval_lab", "llm_usage_ledger.jsonl"))
