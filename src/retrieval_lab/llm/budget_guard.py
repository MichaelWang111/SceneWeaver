from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from sceneweaver.llm.client import LLMConfig
from sceneweaver.llm.providers import (
    ModelPricing,
    ProviderCapabilityError,
    ProviderLimits,
    ProviderProfile,
    balance_cny as provider_balance_cny,
    dashscope_balance_cny_from_env,
    deepseek_balance_cny,
    deepseek_limits,
    deepseek_pricing,
    estimate_request_cost_cny,
    format_cny,
    infer_provider_from_base_url,
    infer_provider_from_model,
    model_limits,
    model_pricing,
    normalize_provider,
    round_decimal,
)

DEFAULT_LLM_USAGE_LEDGER = Path(".tmp") / "retrieval_lab" / "llm_usage_ledger.jsonl"


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    provider: str
    model: str
    batch_id: int
    sample_count: int
    prompt_tokens_upper_bound: int
    max_completion_tokens: int
    reserved_cny: Decimal
    balance_before_cny: Decimal | None
    started_at: float


class ProviderBudgetClient(Protocol):
    provider: str
    model: str

    def profile(self) -> ProviderProfile:
        ...

    def balance_cny(self) -> Decimal | None:
        ...


class HardBudgetError(RuntimeError):
    pass


class DeepSeekBudgetClient:
    provider = "deepseek"

    def __init__(self, *, api_key: str, model: str, base_url: str = "https://api.deepseek.com") -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def profile(self) -> ProviderProfile:
        balance = self.balance_cny()
        pricing = model_pricing(self.provider, self.model, required=True)
        if pricing is None:
            raise HardBudgetError(f"missing DeepSeek pricing for model {self.model!r}")
        return ProviderProfile(
            provider=self.provider,
            model=self.model,
            pricing=pricing,
            limits=model_limits(self.provider, self.model),
            balance_available=balance,
            account_available=balance is not None,
        )

    def balance_cny(self) -> Decimal | None:
        try:
            return deepseek_balance_cny(api_key=self.api_key, base_url=self.base_url)
        except ProviderCapabilityError as exc:
            raise HardBudgetError(str(exc)) from exc


class DashScopeBudgetClient:
    provider = "dashscope"

    def __init__(self, *, model: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def profile(self) -> ProviderProfile:
        balance = self.balance_cny()
        pricing = model_pricing(self.provider, self.model, required=True)
        if pricing is None:
            raise HardBudgetError(f"missing DashScope/Bailian pricing for model {self.model!r}")
        return ProviderProfile(
            provider=self.provider,
            model=self.model,
            pricing=pricing,
            limits=model_limits(self.provider, self.model),
            balance_available=balance,
            account_available=balance is not None,
        )

    def balance_cny(self) -> Decimal | None:
        try:
            return dashscope_balance_cny_from_env()
        except ProviderCapabilityError as exc:
            message = str(exc)
            if "BSS credentials" in message:
                message = f"{message} DashScope hard budget is fail-closed without account-billing credentials."
            raise HardBudgetError(message) from exc


class ProviderBudgetGuard:
    def __init__(
        self,
        *,
        client: ProviderBudgetClient,
        hard_budget_cny: float,
        ledger_path: Path = DEFAULT_LLM_USAGE_LEDGER,
        safety_cny: float = 0.05,
        balance_check_interval_seconds: float = 10.0,
    ) -> None:
        if hard_budget_cny <= 0:
            raise ValueError("hard_budget_cny must be > 0")
        self.client = client
        self.hard_budget_cny = _decimal(hard_budget_cny, field="hard_budget_cny")
        self.safety_cny = _decimal(max(0.0, safety_cny), field="safety_cny")
        self.ledger_path = ledger_path
        self.balance_check_interval_seconds = max(0.0, balance_check_interval_seconds)
        self.lock = threading.Lock()
        self.profile_data: ProviderProfile | None = None
        self.initial_balance_cny: Decimal | None = None
        self.last_balance_cny: Decimal | None = None
        self.last_balance_check_time = 0.0
        self.reserved_cny = Decimal("0")
        self.actual_usage_cost_cny = Decimal("0")
        self.observed_balance_delta_cny = Decimal("0")
        self.reserved_cny_peak = Decimal("0")
        self.stop_reason = ""
        self._next_id = 0

    @property
    def enabled(self) -> bool:
        return True

    def preflight(self) -> ProviderProfile:
        profile = self.client.profile()
        if profile.balance_available is None:
            raise HardBudgetError("hard budget requires a provider balance check")
        if profile.currency.upper() != "CNY":
            raise HardBudgetError(f"hard budget requires CNY balance, got {profile.currency!r}")
        required = self.hard_budget_cny + self.safety_cny
        if profile.balance_available < required:
            raise HardBudgetError(
                f"provider balance {format_cny(profile.balance_available)} CNY is below required "
                f"{format_cny(required)} CNY"
            )
        self.profile_data = profile
        self.initial_balance_cny = profile.balance_available
        self.last_balance_cny = profile.balance_available
        self.last_balance_check_time = time.time()
        return profile

    def reserve(
        self,
        *,
        batch_id: int,
        sample_count: int,
        prompt_tokens_upper_bound: int,
        max_completion_tokens: int,
    ) -> BudgetReservation:
        with self.lock:
            if self.stop_reason:
                raise HardBudgetError(f"hard budget guard is stopped: {self.stop_reason}")
            profile = self._profile_or_preflight()
            reserved = estimate_request_cost_cny(
                profile.pricing,
                prompt_tokens=prompt_tokens_upper_bound,
                max_completion_tokens=max_completion_tokens,
            )
            projected = self._charged_cny_locked() + self.reserved_cny + reserved
            if projected > self.hard_budget_cny:
                self.stop_reason = "hard_budget_reserved_exhausted"
                raise HardBudgetError(
                    f"hard budget would be exceeded: projected={format_cny(projected)} CNY "
                    f"budget={format_cny(self.hard_budget_cny)} CNY"
                )
            self._next_id += 1
            reservation = BudgetReservation(
                reservation_id=f"llm-{self._next_id}",
                provider=profile.provider,
                model=profile.model,
                batch_id=batch_id,
                sample_count=sample_count,
                prompt_tokens_upper_bound=prompt_tokens_upper_bound,
                max_completion_tokens=max_completion_tokens,
                reserved_cny=reserved,
                balance_before_cny=self.last_balance_cny,
                started_at=time.time(),
            )
            self.reserved_cny += reserved
            self.reserved_cny_peak = max(self.reserved_cny_peak, self.reserved_cny)
            return reservation

    def settle_success(self, reservation: BudgetReservation, *, usage: dict[str, Any], request_id: str | None) -> None:
        status = "ok"
        error = ""
        try:
            actual_cost = self.actual_cost_cny(usage)
        except Exception as exc:
            actual_cost = reservation.reserved_cny
            status = "usage_missing"
            error = str(exc)[:1000]
        with self.lock:
            self.reserved_cny = max(Decimal("0"), self.reserved_cny - reservation.reserved_cny)
            self.actual_usage_cost_cny += actual_cost
            if status != "ok":
                self.stop_reason = self.stop_reason or "provider_usage_missing"
            try:
                balance_after = self._maybe_check_balance_locked(force=True)
            except Exception as exc:
                balance_after = self.last_balance_cny
                self.stop_reason = self.stop_reason or "balance_check_failed"
                error = (error + "; " if error else "") + str(exc)[:1000]
            self._update_observed_delta_locked(balance_after)
            self._write_ledger_locked(
                reservation,
                status=status,
                usage=usage,
                actual_cost_cny=actual_cost,
                request_id=request_id,
                error=error,
                balance_after_cny=balance_after,
            )
            if self._charged_cny_locked() + self.reserved_cny >= self.hard_budget_cny:
                self.stop_reason = self.stop_reason or "hard_budget_spent_exhausted"

    def settle_failure(self, reservation: BudgetReservation, *, error: Exception) -> None:
        with self.lock:
            self.reserved_cny = max(Decimal("0"), self.reserved_cny - reservation.reserved_cny)
            error_text = str(error)[:1000]
            try:
                balance_after = self._maybe_check_balance_locked(force=True)
            except Exception as exc:
                balance_after = self.last_balance_cny
                self.stop_reason = self.stop_reason or "balance_check_failed"
                error_text = (error_text + "; " if error_text else "") + str(exc)[:1000]
            self._update_observed_delta_locked(balance_after)
            self._write_ledger_locked(
                reservation,
                status="error",
                usage={},
                actual_cost_cny=Decimal("0"),
                request_id=None,
                error=error_text,
                balance_after_cny=balance_after,
            )

    def actual_cost_cny(self, usage: dict[str, Any]) -> Decimal:
        profile = self._profile_or_preflight()
        prompt_tokens = _int_usage(usage, "prompt_tokens")
        completion_tokens = _int_usage(usage, "completion_tokens")
        total_tokens = _int_usage(usage, "total_tokens")
        if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
            raise HardBudgetError("provider response missing token usage in hard budget mode")
        if prompt_tokens <= 0 and total_tokens > completion_tokens:
            prompt_tokens = max(0, total_tokens - completion_tokens)
        return estimate_request_cost_cny(
            profile.pricing,
            prompt_tokens=prompt_tokens,
            max_completion_tokens=completion_tokens,
        )

    def summary(self) -> dict[str, Any]:
        with self.lock:
            charged = self._charged_cny_locked()
            profile = self.profile_data
            limits = profile.limits if profile else None
            pricing = profile.pricing if profile else None
            return {
                "hard_budget_enabled": True,
                "hard_budget_cny": float(self.hard_budget_cny),
                "budget_safety_cny": float(self.safety_cny),
                "provider": profile.provider if profile else self.client.provider,
                "model": profile.model if profile else self.client.model,
                "provider_concurrency_limit": limits.concurrency if limits else None,
                "provider_rpm_limit": limits.rpm if limits else None,
                "provider_tpm_limit": limits.tpm if limits else None,
                "provider_limit_source": limits.source if limits else "",
                "provider_pricing_source": pricing.source if pricing else "",
                "initial_balance_cny": float(self.initial_balance_cny) if self.initial_balance_cny is not None else None,
                "last_balance_cny": float(self.last_balance_cny) if self.last_balance_cny is not None else None,
                "actual_usage_cost_cny": float(round_decimal(self.actual_usage_cost_cny)),
                "observed_balance_delta_cny": float(round_decimal(self.observed_balance_delta_cny)),
                "charged_cost_cny": float(round_decimal(charged)),
                "reserved_cny_peak": float(round_decimal(self.reserved_cny_peak)),
                "inflight_reserved_cny": float(round_decimal(self.reserved_cny)),
                "usage_ledger": str(self.ledger_path),
                "budget_stop_reason": self.stop_reason,
            }

    def _profile_or_preflight(self) -> ProviderProfile:
        if self.profile_data is None:
            return self.preflight()
        return self.profile_data

    def _charged_cny_locked(self) -> Decimal:
        return max(self.actual_usage_cost_cny, self.observed_balance_delta_cny)

    def _maybe_check_balance_locked(self, *, force: bool) -> Decimal | None:
        now = time.time()
        if not force and self.balance_check_interval_seconds > 0:
            if now - self.last_balance_check_time < self.balance_check_interval_seconds:
                return self.last_balance_cny
        balance = self.client.balance_cny()
        self.last_balance_cny = balance
        self.last_balance_check_time = now
        return balance

    def _update_observed_delta_locked(self, balance_after: Decimal | None) -> None:
        if self.initial_balance_cny is not None and balance_after is not None:
            self.observed_balance_delta_cny = max(Decimal("0"), self.initial_balance_cny - balance_after)

    def _write_ledger_locked(
        self,
        reservation: BudgetReservation,
        *,
        status: str,
        usage: dict[str, Any],
        actual_cost_cny: Decimal,
        request_id: str | None,
        error: str,
        balance_after_cny: Decimal | None,
    ) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "reservation_id": reservation.reservation_id,
            "provider": reservation.provider,
            "model": reservation.model,
            "batch_id": reservation.batch_id,
            "sample_count": reservation.sample_count,
            "request_id": request_id,
            "status": status,
            "prompt_tokens_upper_bound": reservation.prompt_tokens_upper_bound,
            "max_completion_tokens": reservation.max_completion_tokens,
            "reserved_cny": float(round_decimal(reservation.reserved_cny)),
            "actual_cost_cny": float(round_decimal(actual_cost_cny)),
            "usage": usage,
            "balance_before_cny": float(reservation.balance_before_cny) if reservation.balance_before_cny is not None else None,
            "balance_after_cny": float(balance_after_cny) if balance_after_cny is not None else None,
            "elapsed_seconds": round(time.time() - reservation.started_at, 6),
            "error": error,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def budget_client_from_env(config: LLMConfig | None = None, provider: str = "auto") -> ProviderBudgetClient:
    llm_config = config or LLMConfig.from_env()
    resolved = normalize_provider(provider)
    if resolved == "auto":
        resolved = normalize_provider(getattr(llm_config, "provider", "auto"))
    if resolved == "auto":
        resolved = infer_provider(llm_config)
    if resolved == "deepseek":
        return DeepSeekBudgetClient(api_key=llm_config.api_key, model=llm_config.model, base_url=llm_config.base_url)
    if resolved == "dashscope":
        return DashScopeBudgetClient(model=llm_config.model, base_url=llm_config.base_url)
    raise HardBudgetError(f"unsupported hard budget provider: {provider!r}")


def infer_provider(config: LLMConfig) -> str:
    inferred = infer_provider_from_base_url(config.base_url) or infer_provider_from_model(config.model)
    if inferred:
        return inferred
    raise HardBudgetError(
        "could not infer hard budget provider; pass --provider deepseek/dashscope or configure SCENEWEAVER_LLM_PROVIDER"
    )


def balance_cny(*, provider: str, api_key: str, base_url: str, timeout_seconds: float = 20.0) -> Decimal:
    try:
        return provider_balance_cny(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
    except ProviderCapabilityError as exc:
        raise HardBudgetError(str(exc)) from exc


def dashscope_limits(model: str) -> ProviderLimits:
    return model_limits("dashscope", model)


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise HardBudgetError(f"invalid decimal for {field}: {value!r}") from exc


def _int_usage(usage: dict[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DEFAULT_LLM_USAGE_LEDGER",
    "BudgetReservation",
    "DashScopeBudgetClient",
    "DeepSeekBudgetClient",
    "HardBudgetError",
    "ModelPricing",
    "ProviderBudgetClient",
    "ProviderBudgetGuard",
    "ProviderLimits",
    "ProviderProfile",
    "balance_cny",
    "budget_client_from_env",
    "dashscope_limits",
    "deepseek_limits",
    "deepseek_pricing",
    "estimate_request_cost_cny",
    "format_cny",
    "infer_provider",
    "round_decimal",
]
