from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ModelPricing:
    input_cny_per_million: Decimal
    output_cny_per_million: Decimal
    input_cached_cny_per_million: Decimal | None = None
    currency: str = "CNY"
    source: str = "static"


@dataclass(frozen=True)
class ProviderLimits:
    concurrency: int | None = None
    rpm: int | None = None
    tpm: int | None = None
    source: str = "static"


@dataclass(frozen=True)
class ProviderProfile:
    provider: str
    model: str
    pricing: ModelPricing
    limits: ProviderLimits
    balance_available: Decimal | None = None
    currency: str = "CNY"
    account_available: bool = True


@dataclass(frozen=True)
class ApiKeyInfo:
    provider: str
    configured: bool
    active_env_name: str | None
    accepted_env_names: tuple[str, ...]


@dataclass(frozen=True)
class LLMModelInfo:
    id: str
    provider: str
    owner: str = ""
    object: str = "model"
    created: int | None = None
    pricing: ModelPricing | None = None
    limits: ProviderLimits | None = None
    source: str = "static"


@dataclass(frozen=True)
class LLMProviderStatus:
    provider: str
    base_url: str
    configured_model: str
    api_key: ApiKeyInfo
    balance_cny: Decimal | None
    balance_status: str
    account_available: bool | None
    models: tuple[LLMModelInfo, ...]
    model_source: str
    limits: ProviderLimits
    pricing: ModelPricing | None
    notes: tuple[str, ...] = ()


class ProviderCapabilityError(RuntimeError):
    pass


_PROVIDER_ALIASES = {
    "aliyun": "dashscope",
    "bailian": "dashscope",
    "dashscope": "dashscope",
    "deepseek": "deepseek",
}

_PROVIDER_KEY_ENV_NAMES = {
    "deepseek": ("SCENEWEAVER_API_KEY", "DEEPSEEK_API_KEY"),
    "dashscope": ("SCENEWEAVER_API_KEY", "DASHSCOPE_API_KEY", "VIDEO_ANALYZER_API_KEY"),
}

_PROVIDER_BASE_URL_ENV_NAMES = {
    "deepseek": ("SCENEWEAVER_BASE_URL", "DEEPSEEK_BASE_URL"),
    "dashscope": ("SCENEWEAVER_BASE_URL", "DASHSCOPE_BASE_URL", "VIDEO_ANALYZER_BASE_URL"),
}

_PROVIDER_MODEL_ENV_NAMES = {
    "deepseek": ("SCENEWEAVER_MODEL", "DEEPSEEK_MODEL"),
    "dashscope": ("SCENEWEAVER_MODEL", "DASHSCOPE_MODEL", "VIDEO_ANALYZER_MODEL"),
}

_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "dashscope": "qwen3.6-plus",
}

_MODEL_ALIASES = {
    "qwen-3.6plus": "qwen3.6-plus",
    "qwen-3.6-plus": "qwen3.6-plus",
    "qwen3.6plus": "qwen3.6-plus",
    "qwen-3.6flash": "qwen3.6-flash",
    "qwen-3.6-flash": "qwen3.6-flash",
    "qwen3.6flash": "qwen3.6-flash",
}

_STATIC_MODEL_IDS = {
    "deepseek": ("deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"),
    "dashscope": (
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "qwen3.5-plus",
        "qwen3.5-flash",
        "qwen-plus",
        "qwen-turbo",
        "qwen-max",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ),
}

_DASHSCOPE_LIMITS = {
    "qwen3.7-max": ProviderLimits(rpm=1200, tpm=1_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen3.7-plus": ProviderLimits(rpm=5000, tpm=5_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen3.6-plus": ProviderLimits(rpm=30000, tpm=5_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen3.6-flash": ProviderLimits(rpm=30000, tpm=5_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen3.5-plus": ProviderLimits(rpm=5000, tpm=5_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen3.5-flash": ProviderLimits(rpm=5000, tpm=5_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen-plus": ProviderLimits(rpm=1200, tpm=1_000_000, source="aliyun_model_studio_rate_limit_doc"),
    "qwen-turbo": ProviderLimits(rpm=1200, tpm=1_000_000, source="aliyun_model_studio_rate_limit_doc"),
}


_DEEPSEEK_DOCS = {
    "models": "https://api-docs.deepseek.com/zh-cn/api/list-models",
    "balance": "https://api-docs.deepseek.com/zh-cn/api/get-user-balance",
    "pricing": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
    "rate_limit": "https://api-docs.deepseek.com/zh-cn/quick_start/rate_limit",
}

_DASHSCOPE_DOCS = {
    "models": "https://help.aliyun.com/zh/model-studio/models",
    "rate_limit": "https://help.aliyun.com/zh/model-studio/rate-limit",
    "compatible_api": "https://help.aliyun.com/zh/model-studio/first-api-call-to-qwen",
    "balance": "https://help.aliyun.com/zh/bssopenapi/",
}


DocsByProvider = Mapping[str, str]


def normalize_provider(provider: str | None) -> str:
    value = (provider or "auto").strip().lower()
    if not value or value == "auto":
        return "auto"
    normalized = _PROVIDER_ALIASES.get(value)
    if normalized is None:
        raise ProviderCapabilityError(f"unsupported LLM provider: {provider!r}")
    return normalized


def normalize_model_id(model: str | None) -> str:
    value = str(model or "").strip()
    return _MODEL_ALIASES.get(value.lower(), value)


def infer_provider_from_env(env: Mapping[str, str] | None = None) -> str:
    values = env or os.environ
    explicit = normalize_provider(values.get("SCENEWEAVER_LLM_PROVIDER", "auto"))
    if explicit != "auto":
        return explicit

    base_url = first_env_value(("SCENEWEAVER_BASE_URL", "DASHSCOPE_BASE_URL", "VIDEO_ANALYZER_BASE_URL", "DEEPSEEK_BASE_URL"), values)
    if base_url:
        inferred = infer_provider_from_base_url(base_url)
        if inferred:
            return inferred

    model = first_env_value(("SCENEWEAVER_MODEL", "DASHSCOPE_MODEL", "VIDEO_ANALYZER_MODEL", "DEEPSEEK_MODEL"), values)
    if model:
        inferred = infer_provider_from_model(model)
        if inferred:
            return inferred

    if values.get("DEEPSEEK_API_KEY") and not (values.get("DASHSCOPE_API_KEY") or values.get("VIDEO_ANALYZER_API_KEY")):
        return "deepseek"
    if values.get("DASHSCOPE_API_KEY") or values.get("VIDEO_ANALYZER_API_KEY"):
        return "dashscope"
    return "dashscope"


def infer_provider_from_base_url(base_url: str) -> str | None:
    value = str(base_url or "").lower()
    if "deepseek" in value:
        return "deepseek"
    if "dashscope" in value or "aliyuncs.com" in value or "maas.aliyuncs.com" in value:
        return "dashscope"
    return None


def infer_provider_from_model(model: str) -> str | None:
    value = normalize_model_id(model).lower()
    if value.startswith("deepseek"):
        return "deepseek"
    if value.startswith(("qwen", "qwq", "qvq", "wan", "text-embedding", "tongyi")):
        return "dashscope"
    return None


def provider_docs(provider: str) -> DocsByProvider:
    normalized = normalize_provider(provider)
    if normalized == "deepseek":
        return _DEEPSEEK_DOCS
    if normalized == "dashscope":
        return _DASHSCOPE_DOCS
    return {}


def default_base_url(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized not in _DEFAULT_BASE_URLS:
        raise ProviderCapabilityError(f"missing default base URL for provider {provider!r}")
    return _DEFAULT_BASE_URLS[normalized]


def default_model(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized not in _DEFAULT_MODELS:
        raise ProviderCapabilityError(f"missing default model for provider {provider!r}")
    return _DEFAULT_MODELS[normalized]


def resolve_api_key(provider: str, env: Mapping[str, str] | None = None) -> str:
    info = api_key_info(provider, env=env)
    if not info.configured or info.active_env_name is None:
        return ""
    return (env or os.environ).get(info.active_env_name, "")


def api_key_info(provider: str, env: Mapping[str, str] | None = None) -> ApiKeyInfo:
    normalized = normalize_provider(provider)
    values = env or os.environ
    names = _PROVIDER_KEY_ENV_NAMES.get(normalized, ())
    for name in names:
        if values.get(name):
            return ApiKeyInfo(provider=normalized, configured=True, active_env_name=name, accepted_env_names=names)
    return ApiKeyInfo(provider=normalized, configured=False, active_env_name=None, accepted_env_names=names)


def resolve_base_url(provider: str, env: Mapping[str, str] | None = None) -> str:
    normalized = normalize_provider(provider)
    return first_env_value(_PROVIDER_BASE_URL_ENV_NAMES.get(normalized, ()), env or os.environ) or default_base_url(normalized)


def resolve_model(provider: str, env: Mapping[str, str] | None = None) -> str:
    normalized = normalize_provider(provider)
    model = first_env_value(_PROVIDER_MODEL_ENV_NAMES.get(normalized, ()), env or os.environ) or default_model(normalized)
    return normalize_model_id(model)


def first_env_value(names: Iterable[str], env: Mapping[str, str]) -> str:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return ""


def static_models(provider: str) -> tuple[LLMModelInfo, ...]:
    normalized = normalize_provider(provider)
    return tuple(
        LLMModelInfo(
            id=model_id,
            provider=normalized,
            owner=normalized,
            pricing=model_pricing(normalized, model_id, required=False),
            limits=model_limits(normalized, model_id),
            source="static_registry",
        )
        for model_id in _STATIC_MODEL_IDS.get(normalized, ())
    )


def list_models(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    live: bool = False,
    timeout_seconds: float = 20.0,
) -> tuple[LLMModelInfo, ...]:
    normalized = normalize_provider(provider)
    if not live:
        return static_models(normalized)
    if not api_key:
        raise ProviderCapabilityError(f"{normalized} live model listing requires an API key")
    payload = openai_compatible_get_json(base_url=base_url, path="/models", api_key=api_key, timeout_seconds=timeout_seconds)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ProviderCapabilityError(f"{normalized} /models response missing data[]")
    models: list[LLMModelInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id", "")).strip()
        if not model_id:
            continue
        models.append(
            LLMModelInfo(
                id=model_id,
                provider=normalized,
                owner=str(row.get("owned_by", "")),
                object=str(row.get("object", "model")),
                created=optional_int(row.get("created")),
                pricing=model_pricing(normalized, model_id, required=False),
                limits=model_limits(normalized, model_id),
                source="live_provider_models_api",
            )
        )
    return tuple(models)


def openai_compatible_get_json(*, base_url: str, path: str, api_key: str, timeout_seconds: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        method="GET",
    )
    return urlopen_json(request, timeout_seconds=timeout_seconds)


def deepseek_balance_cny(*, api_key: str, base_url: str = "https://api.deepseek.com", timeout_seconds: float = 20.0) -> Decimal:
    if not api_key:
        raise ProviderCapabilityError("DeepSeek balance query requires SCENEWEAVER_API_KEY or DEEPSEEK_API_KEY")
    payload = openai_compatible_get_json(
        base_url=base_url,
        path="/user/balance",
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if not bool(payload.get("is_available", False)):
        raise ProviderCapabilityError("DeepSeek account is not available for API calls")
    return balance_from_infos(payload.get("balance_infos"), currency="CNY", provider="DeepSeek")


def dashscope_balance_cny_from_env(timeout_seconds: float = 20.0) -> Decimal:
    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "")
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")
    if not access_key_id or not access_key_secret:
        raise ProviderCapabilityError(
            "DashScope/Bailian balance query requires Alibaba Cloud BSS credentials "
            "(ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET). "
            "A DASHSCOPE_API_KEY can call models but cannot query account balance."
        )
    endpoint = os.environ.get("ALIBABA_CLOUD_BSS_ENDPOINT", "https://business.aliyuncs.com/")
    payload = aliyun_rpc_json(
        endpoint=endpoint,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        action="QueryAccountBalance",
        version="2017-12-14",
        timeout_seconds=timeout_seconds,
    )
    return parse_aliyun_account_balance_cny(payload)


def balance_cny(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    timeout_seconds: float = 20.0,
) -> Decimal:
    normalized = normalize_provider(provider)
    if normalized == "deepseek":
        return deepseek_balance_cny(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)
    if normalized == "dashscope":
        return dashscope_balance_cny_from_env(timeout_seconds=timeout_seconds)
    raise ProviderCapabilityError(f"unsupported balance query provider: {provider!r}")


def provider_status(
    *,
    provider: str = "auto",
    live_models: bool = False,
    check_balance: bool = False,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 20.0,
) -> LLMProviderStatus:
    values = env or os.environ
    normalized = infer_provider_from_env(values) if normalize_provider(provider) == "auto" else normalize_provider(provider)
    base_url = resolve_base_url(normalized, values)
    configured_model = resolve_model(normalized, values)
    key_info = api_key_info(normalized, values)
    api_key = resolve_api_key(normalized, values)
    notes: list[str] = []
    model_source = "static_registry"
    try:
        models = list_models(provider=normalized, base_url=base_url, api_key=api_key, live=live_models, timeout_seconds=timeout_seconds)
        model_source = "live_provider_models_api" if live_models else "static_registry"
    except Exception as exc:
        models = static_models(normalized)
        model_source = "static_registry_after_live_error"
        notes.append(f"model list live query failed: {exc}")

    balance_value: Decimal | None = None
    account_available: bool | None = None
    balance_status = "not_checked"
    if check_balance:
        try:
            balance_value = balance_cny(provider=normalized, api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)
            account_available = True
            balance_status = "ok"
        except Exception as exc:
            account_available = False
            balance_status = "unavailable"
            notes.append(f"balance query failed: {exc}")

    return LLMProviderStatus(
        provider=normalized,
        base_url=base_url,
        configured_model=configured_model,
        api_key=key_info,
        balance_cny=balance_value,
        balance_status=balance_status,
        account_available=account_available,
        models=models,
        model_source=model_source,
        limits=model_limits(normalized, configured_model),
        pricing=model_pricing(normalized, configured_model, required=False),
        notes=tuple(notes),
    )


def provider_status_dict(status: LLMProviderStatus, *, include_models: bool = True) -> dict[str, Any]:
    data = {
        "provider": status.provider,
        "base_url": status.base_url,
        "configured_model": status.configured_model,
        "api_key": {
            "configured": status.api_key.configured,
            "active_env_name": status.api_key.active_env_name,
            "accepted_env_names": list(status.api_key.accepted_env_names),
        },
        "balance_cny": float(status.balance_cny) if status.balance_cny is not None else None,
        "balance_status": status.balance_status,
        "account_available": status.account_available,
        "model_count": len(status.models),
        "model_source": status.model_source,
        "limits": provider_limits_dict(status.limits),
        "pricing": model_pricing_dict(status.pricing),
        "docs": dict(provider_docs(status.provider)),
        "notes": list(status.notes),
    }
    if include_models:
        data["models"] = [model_info_dict(model) for model in status.models]
    return data


def model_info_dict(model: LLMModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "provider": model.provider,
        "owner": model.owner,
        "object": model.object,
        "created": model.created,
        "pricing": model_pricing_dict(model.pricing),
        "limits": provider_limits_dict(model.limits),
        "source": model.source,
    }


def model_pricing_dict(pricing: ModelPricing | None) -> dict[str, Any] | None:
    if pricing is None:
        return None
    return {
        "input_cny_per_million": float(pricing.input_cny_per_million),
        "output_cny_per_million": float(pricing.output_cny_per_million),
        "input_cached_cny_per_million": float(pricing.input_cached_cny_per_million) if pricing.input_cached_cny_per_million is not None else None,
        "currency": pricing.currency,
        "source": pricing.source,
    }


def provider_limits_dict(limits: ProviderLimits | None) -> dict[str, Any] | None:
    if limits is None:
        return None
    return {
        "concurrency": limits.concurrency,
        "rpm": limits.rpm,
        "tpm": limits.tpm,
        "source": limits.source,
    }


def model_pricing(provider: str, model: str, *, required: bool = True) -> ModelPricing | None:
    normalized = normalize_provider(provider)
    if normalized == "deepseek":
        pricing = deepseek_pricing(model, required=required)
    elif normalized == "dashscope":
        pricing = dashscope_pricing(model, required=required)
    else:
        pricing = None
    if pricing is None and required:
        raise ProviderCapabilityError(f"missing pricing for provider={provider!r} model={model!r}")
    return pricing


def deepseek_pricing(model: str, *, required: bool = True) -> ModelPricing | None:
    normalized = normalize_model_id(model).lower()
    if normalized in {"deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"}:
        return ModelPricing(
            input_cny_per_million=Decimal("1"),
            output_cny_per_million=Decimal("2"),
            input_cached_cny_per_million=Decimal("0.02"),
            source="deepseek_zh_pricing_doc_2026_06",
        )
    if normalized == "deepseek-v4-pro":
        return ModelPricing(
            input_cny_per_million=Decimal("3"),
            output_cny_per_million=Decimal("6"),
            input_cached_cny_per_million=Decimal("0.025"),
            source="deepseek_zh_pricing_doc_2026_06",
        )
    if required:
        raise ProviderCapabilityError(f"missing DeepSeek pricing for model {model!r}")
    return None


def dashscope_pricing(model: str, *, required: bool = True) -> ModelPricing | None:
    normalized = normalize_model_id(model).lower()
    env_pricing = pricing_from_env(normalized)
    if env_pricing is not None:
        return env_pricing
    pricing = _dashscope_static_pricing(normalized)
    if pricing is not None:
        return pricing
    if required:
        raise ProviderCapabilityError(
            f"missing DashScope/Bailian pricing for model {model!r}; set "
            "SCENEWEAVER_INPUT_CNY_PER_MILLION and SCENEWEAVER_OUTPUT_CNY_PER_MILLION "
            "or add the model to the local registry after confirming the official price"
        )
    return None


def _dashscope_static_pricing(model: str) -> ModelPricing | None:
    if model.startswith("qwen3.6-flash") or model.startswith("qwen3.5-flash"):
        return ModelPricing(
            input_cny_per_million=Decimal("0.3"),
            output_cny_per_million=Decimal("0.6"),
            source="local_dashscope_pricing_registry",
        )
    if model.startswith("qwen3.6-plus") or model.startswith("qwen3.5-plus") or model == "qwen-plus":
        return ModelPricing(
            input_cny_per_million=Decimal("0.8"),
            output_cny_per_million=Decimal("2"),
            source="local_dashscope_pricing_registry",
        )
    return None


def pricing_from_env(model: str) -> ModelPricing | None:
    input_value = os.environ.get(f"SCENEWEAVER_{env_model_key(model)}_INPUT_CNY_PER_MILLION") or os.environ.get("SCENEWEAVER_INPUT_CNY_PER_MILLION")
    output_value = os.environ.get(f"SCENEWEAVER_{env_model_key(model)}_OUTPUT_CNY_PER_MILLION") or os.environ.get("SCENEWEAVER_OUTPUT_CNY_PER_MILLION")
    cached_value = os.environ.get(f"SCENEWEAVER_{env_model_key(model)}_INPUT_CACHED_CNY_PER_MILLION") or os.environ.get("SCENEWEAVER_INPUT_CACHED_CNY_PER_MILLION")
    if not input_value or not output_value:
        return None
    return ModelPricing(
        input_cny_per_million=_decimal(input_value, field="SCENEWEAVER_INPUT_CNY_PER_MILLION"),
        output_cny_per_million=_decimal(output_value, field="SCENEWEAVER_OUTPUT_CNY_PER_MILLION"),
        input_cached_cny_per_million=_decimal(cached_value, field="SCENEWEAVER_INPUT_CACHED_CNY_PER_MILLION") if cached_value else None,
        source="environment_override",
    )


def model_limits(provider: str, model: str) -> ProviderLimits:
    normalized = normalize_provider(provider)
    env_limits = limits_from_env(model)
    if env_limits is not None:
        return env_limits
    if normalized == "deepseek":
        return deepseek_limits(model)
    if normalized == "dashscope":
        return dashscope_limits(model)
    return ProviderLimits(source="unknown")


def deepseek_limits(model: str) -> ProviderLimits:
    normalized = normalize_model_id(model).lower()
    if normalized in {"deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"}:
        return ProviderLimits(concurrency=2500, source="deepseek_zh_rate_limit_doc_2026_06")
    if normalized == "deepseek-v4-pro":
        return ProviderLimits(concurrency=500, source="deepseek_zh_rate_limit_doc_2026_06")
    return ProviderLimits(source="unknown")


def dashscope_limits(model: str) -> ProviderLimits:
    normalized = normalize_model_id(model).lower()
    if normalized in _DASHSCOPE_LIMITS:
        return _DASHSCOPE_LIMITS[normalized]
    for prefix, limits in _DASHSCOPE_LIMITS.items():
        if normalized.startswith(prefix + "-"):
            return limits
    return ProviderLimits(source="aliyun_model_studio_rate_limit_doc_unmapped")


def limits_from_env(model: str) -> ProviderLimits | None:
    prefix = env_model_key(model)
    concurrency = optional_int(os.environ.get(f"SCENEWEAVER_{prefix}_CONCURRENCY_LIMIT") or os.environ.get("SCENEWEAVER_CONCURRENCY_LIMIT"))
    rpm = optional_int(os.environ.get(f"SCENEWEAVER_{prefix}_RPM_LIMIT") or os.environ.get("SCENEWEAVER_RPM_LIMIT"))
    tpm = optional_int(os.environ.get(f"SCENEWEAVER_{prefix}_TPM_LIMIT") or os.environ.get("SCENEWEAVER_TPM_LIMIT"))
    if concurrency is None and rpm is None and tpm is None:
        return None
    return ProviderLimits(concurrency=concurrency, rpm=rpm, tpm=tpm, source="environment_override")


def estimate_request_cost_cny(
    pricing: ModelPricing,
    *,
    prompt_tokens: int,
    max_completion_tokens: int,
) -> Decimal:
    input_cost = Decimal(max(0, prompt_tokens)) * pricing.input_cny_per_million / Decimal(1_000_000)
    output_cost = Decimal(max(0, max_completion_tokens)) * pricing.output_cny_per_million / Decimal(1_000_000)
    return round_decimal(input_cost + output_cost)


def balance_from_infos(value: Any, *, currency: str, provider: str) -> Decimal:
    if not isinstance(value, list):
        raise ProviderCapabilityError(f"{provider} balance response missing balance_infos[]")
    for row in value:
        if not isinstance(row, dict):
            continue
        if str(row.get("currency", "")).upper() == currency.upper():
            return _decimal(row.get("total_balance"), field=f"{provider} total_balance")
    raise ProviderCapabilityError(f"{provider} balance response did not include {currency.upper()} total_balance")


def parse_aliyun_account_balance_cny(payload: dict[str, Any]) -> Decimal:
    if str(payload.get("Code", "")).lower() not in {"", "success"}:
        raise ProviderCapabilityError(f"Alibaba Cloud BSS balance query failed: {payload.get('Code')}: {payload.get('Message')}")
    if payload.get("Success") is False:
        raise ProviderCapabilityError(f"Alibaba Cloud BSS balance query failed: {payload.get('Message')}")
    data = payload.get("Data") if isinstance(payload.get("Data"), dict) else payload
    for key in ("AvailableAmount", "AvailableCashAmount", "TotalAvailableAmount", "Balance"):
        if isinstance(data, dict) and data.get(key) not in (None, ""):
            return _decimal(data.get(key), field=f"Alibaba Cloud BSS {key}")
    raise ProviderCapabilityError("Alibaba Cloud BSS balance response did not include an available CNY amount")


def aliyun_rpc_json(
    *,
    endpoint: str,
    access_key_id: str,
    access_key_secret: str,
    action: str,
    version: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    params = {
        "Action": action,
        "Version": version,
        "Format": "JSON",
        "AccessKeyId": access_key_id,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": str(uuid.uuid4()),
        "SignatureVersion": "1.0",
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    signature = aliyun_rpc_signature(params, access_key_secret)
    query = urllib.parse.urlencode({**params, "Signature": signature}, quote_via=urllib.parse.quote)
    url = endpoint.rstrip("/") + "/?" + query
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    return urlopen_json(request, timeout_seconds=timeout_seconds)


def aliyun_rpc_signature(params: Mapping[str, str], access_key_secret: str) -> str:
    canonical = "&".join(
        f"{percent_encode(key)}={percent_encode(params[key])}" for key in sorted(params)
    )
    string_to_sign = "GET&%2F&" + percent_encode(canonical)
    digest = hmac.new((access_key_secret + "&").encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def percent_encode(value: Any) -> str:
    return urllib.parse.quote(str(value), safe="~-_.")


def urlopen_json(request: urllib.request.Request, *, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ProviderCapabilityError(f"HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise ProviderCapabilityError(str(exc)) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderCapabilityError(f"provider returned non-JSON response: {text[:300]}") from exc
    if not isinstance(payload, dict):
        raise ProviderCapabilityError("provider returned a non-object JSON response")
    return payload


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ProviderCapabilityError(f"invalid decimal for {field}: {value!r}") from exc


def round_decimal(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.000001"))


def format_cny(value: Decimal) -> str:
    return f"{round_decimal(value):f}"


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def env_model_key(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(model).upper()).strip("_")


__all__ = [
    "ApiKeyInfo",
    "LLMModelInfo",
    "LLMProviderStatus",
    "ModelPricing",
    "ProviderCapabilityError",
    "ProviderLimits",
    "ProviderProfile",
    "api_key_info",
    "balance_cny",
    "dashscope_balance_cny_from_env",
    "deepseek_balance_cny",
    "deepseek_limits",
    "deepseek_pricing",
    "default_base_url",
    "default_model",
    "estimate_request_cost_cny",
    "format_cny",
    "infer_provider_from_base_url",
    "infer_provider_from_env",
    "infer_provider_from_model",
    "list_models",
    "model_limits",
    "model_pricing",
    "normalize_model_id",
    "normalize_provider",
    "provider_docs",
    "provider_limits_dict",
    "provider_status",
    "provider_status_dict",
    "resolve_api_key",
    "resolve_base_url",
    "resolve_model",
    "round_decimal",
    "static_models",
]
