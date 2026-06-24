from __future__ import annotations

from typing import Any

from sceneweaver.llm.status import llm_status_payload


def llm_status_command(args: Any) -> dict[str, Any]:
    return llm_status_payload(
        provider=str(getattr(args, "provider", "auto") or "auto"),
        live_models=bool(getattr(args, "live_models", False)),
        check_balance=bool(getattr(args, "check_balance", False)),
        include_models=bool(getattr(args, "include_models", False)),
        timeout_seconds=float(getattr(args, "timeout_seconds", 20.0) or 20.0),
        method="retrieval_lab_llm_provider_status",
    )


__all__ = ["llm_status_command"]
