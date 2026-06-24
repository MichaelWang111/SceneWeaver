from __future__ import annotations

import json
import struct
import time
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from sceneweaver.llm.client import LLMConfig, VisionLLMClient
from sceneweaver.llm.providers import (
    default_base_url,
    infer_provider_from_base_url,
    infer_provider_from_env,
    infer_provider_from_model,
    normalize_model_id,
    normalize_provider,
    resolve_api_key,
)

DEFAULT_LLM_CONFIG_PATH = Path(".tmp") / "sceneweaver" / "llm_config.json"
PING_IMAGE_SIZE = 32

DEFAULT_PROVIDER_SCHEMES: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "models": ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"),
    },
    "minimax": {
        "label": "MiniMax",
        "provider": "minimax",
        "base_url": "https://api.minimax.chat/v1",
        "models": ("MiniMax-M1", "MiniMax-Text-01", "MiniMax-VL-01"),
    },
    "dashscope": {
        "label": "DashScope / 百炼",
        "provider": "dashscope",
        "base_url": default_base_url("dashscope"),
        "models": ("qwen3.7-plus", "qwen3.7-max", "qwen3.6-plus", "qwen3.6-flash", "qwen-vl-plus", "qwen-vl-max"),
    },
}

DEFAULT_PROVIDER_SCHEME_ORDER = ("deepseek", "minimax", "dashscope")
DEFAULT_PROVIDERS = tuple(DEFAULT_PROVIDER_SCHEMES)

DEFAULT_BASIC_MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
)

DEFAULT_VISION_MODELS = (
    "qwen3.7-plus",
    "qwen3.7-max",
    "qwen3.6-plus",
    "qwen-vl-plus",
    "qwen-vl-max",
)


LLM_CONFIG_PROFILES: dict[str, dict[str, Any]] = {
    "basic": {
        "label": "基础 API",
        "description": "文本生成、检索规划和脚本生成任务使用。",
        "mode": "preset",
        "scheme": "deepseek",
        "provider": "deepseek",
        "providers": DEFAULT_PROVIDERS,
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "models": DEFAULT_BASIC_MODELS,
    },
    "vision": {
        "label": "图片理解 API",
        "description": "视频入库时的画面理解和场景分析任务使用。",
        "mode": "preset",
        "scheme": "dashscope",
        "provider": "dashscope",
        "providers": DEFAULT_PROVIDERS,
        "model": "qwen3.7-plus",
        "base_url": default_base_url("dashscope"),
        "models": DEFAULT_VISION_MODELS,
    },
}

LLM_PROFILE_ROLE_MAP = {
    "default": "basic",
    "script_generation": "basic",
    "retrieval_judge": "basic",
    "rerank": "basic",
    "scene_analysis": "vision",
}


def llm_config_path(path: str | Path | None = None) -> Path:
    return Path(path or DEFAULT_LLM_CONFIG_PATH)


def load_llm_settings(path: str | Path | None = None) -> dict[str, Any]:
    config_path = llm_config_path(path)
    if not config_path.exists():
        return {"version": 1, "profiles": {}}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid LLM config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"LLM config must be a JSON object: {config_path}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        payload["profiles"] = {}
    return payload


def llm_settings_payload(path: str | Path | None = None, *, include_secrets: bool = False) -> dict[str, Any]:
    config_path = llm_config_path(path)
    raw = load_llm_settings(config_path)
    raw_profiles = raw.get("profiles", {}) if isinstance(raw.get("profiles"), dict) else {}
    profiles: dict[str, Any] = {}
    for name in LLM_CONFIG_PROFILES:
        profile = _display_profile(name, raw_profiles.get(name))
        profiles[name] = _profile_payload(name, profile, saved=name in raw_profiles, include_secret=include_secrets)
    return {
        "version": int(raw.get("version") or 1),
        "config_path": str(config_path.resolve()),
        "updated_at": raw.get("updated_at"),
        "provider_schemes": provider_schemes_payload(),
        "profiles": profiles,
    }


def save_llm_settings(payload: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    config_path = llm_config_path(path)
    current = load_llm_settings(config_path)
    current_profiles = current.get("profiles", {}) if isinstance(current.get("profiles"), dict) else {}
    incoming_profiles = payload.get("profiles", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(incoming_profiles, Mapping):
        raise ValueError("profiles must be an object")

    profiles: dict[str, Any] = {}
    for name in LLM_CONFIG_PROFILES:
        incoming = incoming_profiles.get(name)
        existing = current_profiles.get(name) if isinstance(current_profiles.get(name), Mapping) else {}
        if incoming is None:
            if existing:
                profiles[name] = _persisted_profile(name, existing)
            continue
        if not isinstance(incoming, Mapping):
            raise ValueError(f"profile {name!r} must be an object")
        profiles[name] = _persisted_profile(name, incoming, existing=existing)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    stored = {"version": 1, "updated_at": int(time.time()), "profiles": profiles}
    config_path.write_text(json.dumps(stored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return llm_settings_payload(config_path)


def config_for_role_settings(
    *,
    role: str = "default",
    model: str | None = None,
    base_config: LLMConfig | None = None,
    path: str | Path | None = None,
) -> LLMConfig | None:
    profile_name = LLM_PROFILE_ROLE_MAP.get(role, "basic")
    return config_for_llm_profile(
        profile_name,
        model=model,
        base_config=base_config,
        path=path,
        require_saved=True,
    )


def config_for_llm_profile(
    profile_name: str,
    *,
    profile_data: Mapping[str, Any] | None = None,
    model: str | None = None,
    base_config: LLMConfig | None = None,
    path: str | Path | None = None,
    require_saved: bool = False,
) -> LLMConfig | None:
    normalized_name = _normalize_profile_name(profile_name)
    raw = load_llm_settings(path)
    raw_profiles = raw.get("profiles", {}) if isinstance(raw.get("profiles"), dict) else {}
    saved_profile = raw_profiles.get(normalized_name)
    if require_saved and not isinstance(saved_profile, Mapping):
        return None

    profile = dict(LLM_CONFIG_PROFILES[normalized_name])
    if isinstance(saved_profile, Mapping):
        profile.update(_stored_profile(saved_profile))
    if profile_data:
        profile.update(_incoming_profile(profile_data, existing=profile))
    profile = _apply_scheme_defaults(profile)

    fallback = base_config or LLMConfig.from_env()
    selected_model = normalize_model_id(model or str(profile.get("model") or "") or fallback.model)
    explicit_base_url = str(profile.get("base_url") or "").strip()
    selected_base_url = explicit_base_url or fallback.base_url
    provider = _resolve_profile_provider(str(profile.get("provider") or "auto"), selected_base_url, selected_model, fallback)
    if not selected_base_url and provider != "auto":
        try:
            selected_base_url = default_base_url(provider)
        except Exception:
            selected_base_url = ""
    api_key = _api_key_for_profile(profile, provider, fallback)
    return replace(
        fallback,
        provider=provider,
        api_key=api_key,
        base_url=selected_base_url,
        model=selected_model,
    )


def ping_llm_profile(
    profile_name: str,
    *,
    profile_data: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    normalized_name = _normalize_profile_name(profile_name)
    config = config_for_llm_profile(
        normalized_name,
        profile_data=profile_data,
        path=path,
        require_saved=False,
    )
    if config is None:
        raise ValueError(f"unknown LLM profile: {profile_name}")
    if not config.api_key:
        raise ValueError(f"{LLM_CONFIG_PROFILES[normalized_name]['label']} requires an API key before ping")

    started = time.perf_counter()
    client = VisionLLMClient(config)
    if normalized_name == "vision":
        image_path = _ping_image_path(path)
        data = client.analyze_images_json(
            system_prompt="Return JSON only.",
            user_prompt='Read the attached all-white 32x32 PNG and return {"reply":"pong","vision":true}.',
            image_paths=[image_path],
            timeout_seconds=timeout_seconds,
            retries=0,
        )
        ping_mode = "image"
        image_attached = True
        image_path_text = str(image_path.resolve())
    else:
        result = client.analyze_text_json_result(
            system_prompt="Return JSON only.",
            user_prompt='Return {"reply":"pong"} now.',
            max_tokens=64,
            timeout_seconds=timeout_seconds,
            retries=0,
        )
        data = result.data
        ping_mode = "text"
        image_attached = False
        image_path_text = ""
    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "profile": normalized_name,
        "label": LLM_CONFIG_PROFILES[normalized_name]["label"],
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "ping_mode": ping_mode,
        "image_attached": image_attached,
        "image_path": image_path_text,
        "latency_ms": latency_ms,
        "response": data,
    }


def _display_profile(name: str, raw_profile: Any) -> dict[str, Any]:
    profile = dict(LLM_CONFIG_PROFILES[name])
    if isinstance(raw_profile, Mapping):
        profile.update(_stored_profile(raw_profile))
    else:
        fallback = _default_display_config(name)
        if not profile.get("model"):
            profile["model"] = fallback.model
        if not profile.get("base_url"):
            profile["base_url"] = fallback.base_url
        if profile.get("provider") == "auto":
            profile["provider"] = fallback.provider
    return profile


def _default_display_config(name: str) -> LLMConfig:
    fallback = LLMConfig.from_env()
    if name == "vision":
        provider = "dashscope"
        return replace(
            fallback,
            provider=provider,
            api_key="",
            base_url=default_base_url(provider),
            model=LLM_CONFIG_PROFILES[name]["model"],
        )
    return replace(fallback, api_key="")


def _profile_payload(name: str, profile: Mapping[str, Any], *, saved: bool, include_secret: bool) -> dict[str, Any]:
    api_key = str(profile.get("api_key") or "")
    profile = _apply_scheme_defaults(profile)
    payload = {
        "name": name,
        "label": LLM_CONFIG_PROFILES[name]["label"],
        "description": LLM_CONFIG_PROFILES[name]["description"],
        "mode": str(profile.get("mode") or "preset"),
        "scheme": str(profile.get("scheme") or _scheme_for_provider(str(profile.get("provider") or "")) or "custom"),
        "schemes": provider_schemes_payload(),
        "provider": str(profile.get("provider") or "auto"),
        "providers": _providers_for_payload(name, profile),
        "model": str(profile.get("model") or ""),
        "base_url": str(profile.get("base_url") or ""),
        "models": _models_for_payload(name, profile),
        "api_key_configured": bool(api_key),
        "api_key_mask": mask_api_key(api_key),
        "saved": saved,
    }
    if include_secret:
        payload["api_key"] = api_key
    return payload


def _persisted_profile(name: str, incoming: Mapping[str, Any], *, existing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    profile = _incoming_profile(incoming, existing=existing)
    profile = _apply_scheme_defaults(profile)
    if profile["mode"] == "custom":
        profile["providers"] = _normalize_providers(profile.get("providers", ()), defaults=(), extra=profile.get("provider", ""))
        profile["models"] = _normalize_models(profile.get("models", ()), extra=profile.get("model", ""))
    else:
        profile["providers"] = list(DEFAULT_PROVIDERS)
        profile["models"] = _normalize_models(profile.get("models", ()), extra=profile.get("model", ""))
    if bool(incoming.get("clear_api_key")):
        profile["api_key"] = ""
    else:
        new_key = str(incoming.get("api_key") or "")
        profile["api_key"] = new_key.strip() or str(existing.get("api_key") or "")
    return {
        "mode": profile["mode"],
        "scheme": profile["scheme"],
        "provider": profile["provider"],
        "providers": profile["providers"],
        "model": profile["model"],
        "base_url": profile["base_url"],
        "api_key": profile["api_key"],
        "models": profile["models"],
    }


def _incoming_profile(incoming: Mapping[str, Any], *, existing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    existing = existing or {}
    mode_value = str(incoming.get("mode") or existing.get("mode") or "").strip().lower()
    scheme = str(incoming.get("scheme") or existing.get("scheme") or "").strip().lower()
    provider = str(incoming.get("provider") or existing.get("provider") or "auto").strip() or "auto"
    provider = _normalize_provider_value(provider)
    base_url = str(incoming.get("base_url") or existing.get("base_url") or "").strip()
    if mode_value in {"custom", "preset"}:
        mode = mode_value
    elif scheme == "custom" or _looks_like_custom_profile(provider, base_url):
        mode = "custom"
    else:
        mode = "preset"
    model = normalize_model_id(str(incoming.get("model") or existing.get("model") or "").strip())
    return {
        "mode": mode,
        "scheme": scheme or _scheme_for_provider(provider) or "custom",
        "provider": provider,
        "providers": _normalize_providers(incoming.get("providers", existing.get("providers", ())), extra=provider),
        "model": model,
        "base_url": base_url,
        "api_key": str(incoming.get("api_key") or existing.get("api_key") or "").strip(),
        "models": _normalize_models(incoming.get("models", existing.get("models", ())), extra=model),
    }


def _stored_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    return _incoming_profile(profile)


def _looks_like_custom_profile(provider: str, base_url: str) -> bool:
    scheme_name = _scheme_for_provider(provider)
    if provider != "auto" and scheme_name is None:
        return True
    if not base_url:
        return False
    normalized_base_url = base_url.rstrip("/").lower()
    if scheme_name:
        scheme_base_url = str(DEFAULT_PROVIDER_SCHEMES[scheme_name]["base_url"]).rstrip("/").lower()
        return normalized_base_url != scheme_base_url
    return infer_provider_from_base_url(base_url) is None


def provider_schemes_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "label": str(DEFAULT_PROVIDER_SCHEMES[name]["label"]),
            "provider": str(DEFAULT_PROVIDER_SCHEMES[name]["provider"]),
            "base_url": str(DEFAULT_PROVIDER_SCHEMES[name]["base_url"]),
            "models": list(DEFAULT_PROVIDER_SCHEMES[name]["models"]),
        }
        for name in DEFAULT_PROVIDER_SCHEME_ORDER
    ]


def _scheme_for_provider(provider: str) -> str | None:
    normalized = _normalize_provider_value(provider)
    for name, scheme in DEFAULT_PROVIDER_SCHEMES.items():
        if _normalize_provider_value(str(scheme.get("provider") or "")) == normalized:
            return name
    return None


def _scheme(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    return DEFAULT_PROVIDER_SCHEMES.get(str(name).strip().lower())


def _apply_scheme_defaults(profile: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(profile)
    mode = str(data.get("mode") or "preset").strip().lower()
    data["mode"] = "custom" if mode == "custom" else "preset"
    if data["mode"] == "custom":
        data["scheme"] = str(data.get("scheme") or "custom").strip().lower() or "custom"
        data["provider"] = _normalize_provider_value(str(data.get("provider") or "custom"))
        data["models"] = _normalize_models(data.get("models", ()), extra=str(data.get("model") or ""))
        data["providers"] = _normalize_providers(data.get("providers", ()), extra=str(data.get("provider") or ""))
        return data

    scheme_name = str(data.get("scheme") or _scheme_for_provider(str(data.get("provider") or "")) or "deepseek").strip().lower()
    scheme = _scheme(scheme_name) or DEFAULT_PROVIDER_SCHEMES["deepseek"]
    if scheme_name not in DEFAULT_PROVIDER_SCHEMES:
        scheme_name = "deepseek"
    data["scheme"] = scheme_name
    data["provider"] = _normalize_provider_value(str(scheme["provider"]))
    data["base_url"] = str(scheme["base_url"])
    scheme_models = _normalize_models(scheme.get("models", ()))
    model = normalize_model_id(str(data.get("model") or "").strip())
    data["model"] = model or (scheme_models[0] if scheme_models else model)
    data["models"] = _normalize_models(data.get("models", ()), defaults=scheme_models, extra=data["model"])
    data["providers"] = list(DEFAULT_PROVIDERS)
    return data


def _models_for_payload(name: str, profile: Mapping[str, Any]) -> list[str]:
    profile = _apply_scheme_defaults(profile)
    return _normalize_models(profile.get("models", ()), extra=str(profile.get("model") or ""))


def _providers_for_payload(name: str, profile: Mapping[str, Any]) -> list[str]:
    profile = _apply_scheme_defaults(profile)
    return _normalize_providers(profile.get("providers", ()), defaults=LLM_CONFIG_PROFILES[name].get("providers", DEFAULT_PROVIDERS), extra=str(profile.get("provider") or ""))


def _normalize_provider_value(provider: str) -> str:
    value = str(provider or "auto").strip().lower() or "auto"
    try:
        return normalize_provider(value)
    except Exception:
        return value


def _normalize_providers(
    providers: Any,
    *,
    defaults: Sequence[str] = DEFAULT_PROVIDERS,
    extra: str | None = None,
) -> list[str]:
    values: list[str] = []
    if isinstance(defaults, Sequence) and not isinstance(defaults, (str, bytes)):
        values.extend(str(item) for item in defaults)
    if isinstance(providers, str):
        values.extend(part.strip() for part in providers.replace("\n", ",").split(","))
    elif isinstance(providers, Sequence) and not isinstance(providers, (bytes, bytearray)):
        values.extend(str(item) for item in providers)
    if extra:
        values.append(extra)

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        provider = _normalize_provider_value(value)
        if not provider or provider in seen:
            continue
        seen.add(provider)
        normalized.append(provider)
    return normalized


def _normalize_models(
    models: Any,
    *,
    defaults: Sequence[str] = (),
    extra: str | None = None,
) -> list[str]:
    values: list[str] = []
    if isinstance(defaults, Sequence) and not isinstance(defaults, (str, bytes)):
        values.extend(str(item) for item in defaults)
    if isinstance(models, str):
        values.extend(part.strip() for part in models.replace("\n", ",").split(","))
    elif isinstance(models, Sequence) and not isinstance(models, (bytes, bytearray)):
        values.extend(str(item) for item in models)
    if extra:
        values.append(extra)

    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        model = normalize_model_id(str(value or "").strip())
        if not model or model in seen:
            continue
        seen.add(model)
        normalized.append(model)
    return normalized


def _resolve_profile_provider(provider: str, base_url: str, model: str, fallback: LLMConfig) -> str:
    normalized = _normalize_provider_value(provider)
    if normalized != "auto":
        return normalized
    fallback_provider = _normalize_provider_value(fallback.provider)
    if fallback_provider == "auto":
        fallback_provider = infer_provider_from_env()
    return infer_provider_from_base_url(base_url) or infer_provider_from_model(model) or fallback_provider


def _api_key_for_profile(profile: Mapping[str, Any], provider: str, fallback: LLMConfig) -> str:
    explicit_key = str(profile.get("api_key") or "").strip()
    if explicit_key:
        return explicit_key
    provider_key = _resolve_provider_api_key(provider)
    if provider_key:
        return provider_key
    fallback_provider = _normalize_provider_value(fallback.provider)
    if fallback_provider == "auto":
        fallback_provider = infer_provider_from_env()
    return fallback.api_key if fallback_provider == _normalize_provider_value(provider) else ""


def _resolve_provider_api_key(provider: str) -> str:
    try:
        return resolve_api_key(provider)
    except Exception:
        return ""


def _normalize_profile_name(profile_name: str) -> str:
    normalized = str(profile_name or "").strip().lower()
    if normalized not in LLM_CONFIG_PROFILES:
        raise ValueError(f"unknown LLM profile: {profile_name}")
    return normalized


def _ping_image_path(path: str | Path | None = None) -> Path:
    target = llm_config_path(path).parent / "llm_ping_pixel.png"
    if not _ping_image_is_usable(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_white_png_bytes(PING_IMAGE_SIZE))
    return target


def _ping_image_is_usable(path: Path) -> bool:
    dimensions = _png_dimensions(path)
    if dimensions is None:
        return False
    width, height = dimensions
    return width > 10 and height > 10


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _white_png_bytes(size: int) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    raw_rows = b"".join(b"\x00" + (b"\xff\xff\xff" * size) for _ in range(size))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(raw_rows)) + _png_chunk(b"IEND", b"")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def mask_api_key(api_key: str) -> str:
    value = str(api_key or "")
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


__all__ = [
    "DEFAULT_LLM_CONFIG_PATH",
    "DEFAULT_PROVIDERS",
    "LLM_CONFIG_PROFILES",
    "LLM_PROFILE_ROLE_MAP",
    "config_for_llm_profile",
    "config_for_role_settings",
    "llm_config_path",
    "llm_settings_payload",
    "load_llm_settings",
    "mask_api_key",
    "provider_schemes_payload",
    "ping_llm_profile",
    "save_llm_settings",
]
