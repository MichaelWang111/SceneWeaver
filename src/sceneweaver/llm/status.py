from __future__ import annotations

from typing import Any

from sceneweaver.llm.providers import provider_status, provider_status_dict


def llm_status_payload(
    *,
    provider: str = "auto",
    live_models: bool = False,
    check_balance: bool = False,
    include_models: bool = False,
    timeout_seconds: float = 20.0,
    method: str = "sceneweaver_llm_provider_status",
) -> dict[str, Any]:
    status = provider_status(
        provider=provider,
        live_models=live_models,
        check_balance=check_balance,
        timeout_seconds=timeout_seconds,
    )
    result = provider_status_dict(status, include_models=include_models)
    summary = {
        "provider": result["provider"],
        "configured_model": result["configured_model"],
        "api_key_configured": result["api_key"]["configured"],
        "api_key_env": result["api_key"]["active_env_name"],
        "model_count": result["model_count"],
        "model_source": result["model_source"],
        "balance_status": result["balance_status"],
        "balance_cny": result["balance_cny"],
        "account_available": result["account_available"],
        "limits": result["limits"],
        "pricing": result["pricing"],
        "notes": result["notes"],
    }
    return {
        "method": method,
        "summary": summary,
        "status": result,
    }


def llm_status_command(args: Any) -> dict[str, Any]:
    return llm_status_payload(
        provider=str(getattr(args, "provider", "auto") or "auto"),
        live_models=bool(getattr(args, "live_models", False)),
        check_balance=bool(getattr(args, "check_balance", False)),
        include_models=bool(getattr(args, "include_models", False)),
        timeout_seconds=float(getattr(args, "timeout_seconds", 20.0) or 20.0),
    )


__all__ = ["llm_status_command", "llm_status_payload"]
