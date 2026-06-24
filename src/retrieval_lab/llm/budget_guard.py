from __future__ import annotations

from pathlib import Path

from sceneweaver.llm.budget import *  # noqa: F403
from sceneweaver.llm.budget import (
    __all__ as _SCENEWEAVER_LLM_BUDGET_ALL,
    ProviderBudgetClient,
    ProviderBudgetGuard as _SceneWeaverProviderBudgetGuard,
)

DEFAULT_LLM_USAGE_LEDGER = Path(".tmp") / "retrieval_lab" / "llm_usage_ledger.jsonl"


class ProviderBudgetGuard(_SceneWeaverProviderBudgetGuard):
    def __init__(
        self,
        *,
        client: ProviderBudgetClient,
        hard_budget_cny: float,
        ledger_path: Path = DEFAULT_LLM_USAGE_LEDGER,
        safety_cny: float = 0.05,
        balance_check_interval_seconds: float = 10.0,
    ) -> None:
        super().__init__(
            client=client,
            hard_budget_cny=hard_budget_cny,
            ledger_path=ledger_path,
            safety_cny=safety_cny,
            balance_check_interval_seconds=balance_check_interval_seconds,
        )


__all__ = [*_SCENEWEAVER_LLM_BUDGET_ALL]
