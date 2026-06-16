from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_jsonl, write_json, write_jsonl
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retrieval_lab.llm.budget_guard import (
    DEFAULT_LLM_USAGE_LEDGER,
    BudgetReservation,
    ProviderBudgetGuard,
    budget_client_from_env,
)
from retrieval_lab.qrels import load_qrels, merge_adjudicated_qrels, qrels_audit_summary, write_qrels


DEFAULT_LLM_ADJUDICATION_OUTPUT = Path(".tmp") / "retrieval_lab" / "llm_qrels_adjudications.jsonl"
DEFAULT_LLM_ADJUDICATION_REPORT = Path(".tmp") / "retrieval_lab" / "llm_qrels_adjudication_report.json"
DEFAULT_LLM_JUDGE_CACHE = Path(".tmp") / "retrieval_lab" / "llm_judge_cache.jsonl"
DEFAULT_LLM_NATURAL_FUZZY_OUTPUT = Path(".tmp") / "retrieval_lab" / "llm_natural_fuzzy.jsonl"
DEFAULT_LLM_NATURAL_FUZZY_REPORT = Path(".tmp") / "retrieval_lab" / "llm_natural_fuzzy_report.json"
PROMPT_VERSION = "retrieval_lab_qrels_judge_v1"
NATURAL_FUZZY_PROMPT_VERSION = "retrieval_lab_natural_fuzzy_generator_v1"

NATURAL_PURPOSE_TEXT = {
    "opening": "开场建立问题和真实处境",
    "setup": "前情铺垫和需求建立",
    "technology_showcase": "技术能力被自然看见",
    "team_work": "团队一起推进事情",
    "value_expression": "价值被具体场景承接",
    "outcome": "结果和成效自然显现",
    "growth": "成长变化和长期积累被看见",
    "character_intro": "人物进入故事并让人产生共情",
    "ending": "结尾收束并留下信任",
    "establish_problem": "把问题和现实压力讲清楚",
    "establish_need": "让需求自然成立",
    "build_reality": "真实处境和现场感",
    "build_trust": "建立可信感",
    "show_pressure": "现实压力",
    "show_distance": "距离和阻隔",
    "close_loop": "前后呼应",
    "show_outcome": "结果和成效",
    "scale_reveal": "规模逐渐展开",
    "show_scale": "规模感",
    "show_network": "关系网络和协作链路",
    "show_growth": "成长变化",
    "show_long_termism": "长期积累",
    "introduce_people": "具体人物进入故事",
    "build_empathy": "共情",
    "show_team": "团队能力",
    "show_collaboration": "协作过程",
    "show_technology": "技术能力",
    "prove_capability": "能力和可靠性",
    "express_value": "价值和意义",
    "land_value": "价值落到具体场景",
    "leave_trust": "留下信任",
    "humanize_professional": "专业表达里的人味",
    "humanize_technology": "技术表达里的人味",
    "stabilize_emotion": "稳定情绪",
    "avoid_overclaim": "克制不过度承诺",
    "connect_feedback_to_mission": "反馈和使命之间的连接",
    "show_face_to_face_communication": "面对面沟通",
    "keep_human_warmth": "人的温度",
}

INTERNAL_LABELS = set(NATURAL_PURPOSE_TEXT) | {
    "opening",
    "setup",
    "technology_showcase",
    "team_work",
    "value_expression",
    "outcome",
    "scale_reveal",
    "growth",
    "character_intro",
    "ending",
}


@dataclass(frozen=True)
class JudgeBatchRequest:
    batch_id: int
    samples: list[dict[str, Any]]
    prompt: str
    estimated_tokens: int
    estimated_cost_cny: float
    budget_reservation: BudgetReservation | None = None


@dataclass(frozen=True)
class JudgeBatchResult:
    batch_id: int
    samples: list[dict[str, Any]]
    judgements: list[dict[str, Any]]
    elapsed_seconds: float
    estimated_tokens: int
    estimated_cost_cny: float
    usage: dict[str, Any] | None = None
    request_id: str | None = None
    budget_reservation: BudgetReservation | None = None


@dataclass(frozen=True)
class LLMJudgementResponse:
    judgements: list[dict[str, Any]]
    usage: dict[str, Any]
    request_id: str | None = None


@dataclass(frozen=True)
class LLMNaturalFuzzyResponse:
    rows: list[dict[str, Any]]
    usage: dict[str, Any]
    request_id: str | None = None


def llm_adjudicate_qrels_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    samples_path = Path(getattr(args, "samples"))
    samples = read_jsonl(samples_path)
    sample_limit = int(getattr(args, "llm_sample_size", 80))
    samples = samples[:sample_limit] if sample_limit > 0 else []
    budget = CostBudget(
        budget_cny=float(getattr(args, "budget_cny", 20.0)),
        cny_per_1k_tokens=float(getattr(args, "cny_per_1k_tokens", 0.01)),
    )
    cache = JudgeCache(Path(getattr(args, "judge_cache", DEFAULT_LLM_JUDGE_CACHE)))
    use_fake = bool(getattr(args, "fake_client", False))
    require_llm = bool(getattr(args, "require_llm", False))
    if not use_fake and not require_llm:
        raise ValueError("llm adjudication is opt-in: pass --require-llm for real calls or --fake-client for tests")
    hard_budget_cny = float(getattr(args, "hard_budget_cny", 0.0) or 0.0)
    budget_guard = build_budget_guard(args) if hard_budget_cny > 0 and not use_fake else None
    if budget_guard is not None:
        budget_guard.preflight()

    judgements: list[dict[str, Any]] = []
    fallback_count = 0
    budget_stopped = False
    indexed_judgements: dict[int, dict[str, Any]] = {}
    pending_requests: list[JudgeBatchRequest] = []
    failed_batches: list[dict[str, Any]] = []
    batch_diagnostics: list[dict[str, Any]] = []
    concurrency = max(1, int(getattr(args, "concurrency", 1) or 1))
    requested_batch_size = int(getattr(args, "batch_size", 0) or 0)
    max_batch_size = max(1, int(getattr(args, "max_batch_size", 10) or 10))
    max_batch_tokens = max(512, int(getattr(args, "max_batch_tokens", 6000) or 6000))
    min_batch_size_for_concurrency = max(
        1, int(getattr(args, "min_batch_size_for_concurrency", 5) or 5)
    )
    uncached_indexed_samples: list[tuple[int, dict[str, Any]]] = []
    for sample_index, sample in enumerate(samples):
        cached = cache.get(sample)
        if cached is not None:
            indexed_judgements[sample_index] = cached
        else:
            uncached_indexed_samples.append((sample_index, sample))
    if requested_batch_size > 0:
        planned_batches = chunks(uncached_indexed_samples, requested_batch_size)
        batching_strategy = "manual"
    else:
        planned_batches = adaptive_judge_batches(
            uncached_indexed_samples,
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            min_batch_size_for_concurrency=min_batch_size_for_concurrency,
        )
        batching_strategy = "auto_balanced"
    batch_sizes = [len(batch) for batch in planned_batches]
    batch_size = max(batch_sizes) if batch_sizes else max(1, requested_batch_size or max_batch_size)
    for batch_id, uncached in enumerate(planned_batches):
        uncached_samples = [{**sample, "_sample_index": sample_index} for sample_index, sample in uncached]
        prompt = judge_prompt(uncached_samples)
        estimated_tokens = estimate_tokens(prompt)
        estimated_cost = budget.estimate(estimated_tokens)
        if not budget.can_spend(estimated_cost):
            budget_stopped = True
            break
        budget.spend(estimated_cost, estimated_tokens)
        budget_reservation = None
        if budget_guard is not None:
            try:
                budget_reservation = budget_guard.reserve(
                    batch_id=batch_id,
                    sample_count=len(uncached_samples),
                    prompt_tokens_upper_bound=estimated_tokens,
                    max_completion_tokens=int(getattr(args, "max_tokens", 1800)),
                )
            except Exception:
                budget_stopped = True
                break
        pending_requests.append(
            JudgeBatchRequest(
                batch_id=batch_id,
                samples=uncached_samples,
                prompt=prompt,
                estimated_tokens=estimated_tokens,
                estimated_cost_cny=estimated_cost,
                budget_reservation=budget_reservation,
            )
        )
    max_tokens = int(getattr(args, "max_tokens", 1800))
    timeout_seconds = float(getattr(args, "timeout_seconds", 180.0))
    retries = int(getattr(args, "retries", 0))
    if not pending_requests:
        pass
    elif use_fake or concurrency == 1:
        for request in pending_requests:
            try:
                result = run_judge_batch(
                    request,
                    use_fake=use_fake,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
            except Exception as exc:
                settle_budget_failure(budget_guard, request, exc)
                fallback_count += len(request.samples)
                failed_batches.append(failed_batch_row(request, exc))
                continue
            settle_budget_success(budget_guard, result)
            save_judge_batch_result(result, cache, indexed_judgements, batch_diagnostics)
    else:
        workers = min(concurrency, len(pending_requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_judge_batch,
                    request,
                    use_fake=use_fake,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                ): request
                for request in pending_requests
            }
            for future in as_completed(futures):
                request = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    settle_budget_failure(budget_guard, request, exc)
                    fallback_count += len(request.samples)
                    failed_batches.append(failed_batch_row(request, exc))
                    continue
                settle_budget_success(budget_guard, result)
                save_judge_batch_result(result, cache, indexed_judgements, batch_diagnostics)
    judgements = [indexed_judgements[index] for index in sorted(indexed_judgements)]
    llm_call_count = len(pending_requests) if not use_fake else 0

    output = Path(getattr(args, "output", DEFAULT_LLM_ADJUDICATION_OUTPUT))
    write_jsonl(output, judgements)
    merged_summary: dict[str, Any] = {}
    merged_output = Path(getattr(args, "merged_qrels_output", "")) if getattr(args, "merged_qrels_output", "") else None
    qrels_path = Path(getattr(args, "qrels", "")) if getattr(args, "qrels", "") else None
    if qrels_path is not None and qrels_path.exists() and merged_output is not None:
        merged = merge_adjudicated_qrels(load_qrels(qrels_path), judgements)
        write_qrels(merged_output, merged)
        merged_summary = {**qrels_audit_summary(merged), "merged_qrels_output": str(merged_output)}

    elapsed = round(time.perf_counter() - started, 6)
    summary = {
        "sample_count": len(samples),
        "judgement_count": len(judgements),
        "llm_call_count": llm_call_count,
        "successful_batch_count": len(batch_diagnostics),
        "failed_batch_count": len(failed_batches),
        "submitted_batch_count": len(pending_requests),
        "batch_size": batch_size,
        "batch_sizes": batch_sizes,
        "batching_strategy": batching_strategy,
        "max_batch_size": max_batch_size,
        "max_batch_tokens": max_batch_tokens,
        "min_batch_size_for_concurrency": min_batch_size_for_concurrency,
        "concurrency": concurrency,
        "effective_concurrency": min(concurrency, len(pending_requests)) if pending_requests else 0,
        "fallback_count": fallback_count,
        "cache_hit_count": cache.hit_count,
        "cache_miss_count": cache.miss_count,
        "budget_cny": budget.budget_cny,
        "estimated_tokens": budget.spent_tokens,
        "total_estimated_cost_cny": round(budget.spent_cny, 6),
        "budget_stopped": budget_stopped,
        "status": "partial_failed" if failed_batches else "ok",
        "manual_count": 0,
        "llm_count": len(judgements),
        "output": str(output),
        "elapsed_seconds": elapsed,
        **(budget_guard.summary() if budget_guard is not None else {"hard_budget_enabled": False}),
        **merged_summary,
    }
    report = {
        "method": "retrieval_lab_llm_qrels_adjudication",
        "samples": str(samples_path),
        "output": str(output),
        "summary": summary,
        "batch_diagnostics": sorted(batch_diagnostics, key=lambda row: int(row.get("batch_id", 0))),
        "failed_batches": sorted(failed_batches, key=lambda row: int(row.get("batch_id", 0))),
        "judgement_preview": judgements[:20],
        "fingerprint": data_sha256({"summary": summary, "judgements": judgements}),
    }
    report_output = Path(getattr(args, "report_output", DEFAULT_LLM_ADJUDICATION_REPORT))
    write_json(report_output, report)
    return report


def run_judge_batch(
    request: JudgeBatchRequest,
    *,
    use_fake: bool,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> JudgeBatchResult:
    started = time.perf_counter()
    if use_fake:
        judgements = [fake_judgement(sample) for sample in request.samples]
        usage: dict[str, Any] = {}
        request_id = None
    else:
        if request.budget_reservation is None:
            judgements = real_llm_judgements(
                request.samples,
                prompt=request.prompt,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
            usage = {}
            request_id = None
        else:
            response = real_llm_judgement_response(
                request.samples,
                prompt=request.prompt,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                retries=retries,
            )
            judgements = response.judgements
            usage = response.usage
            request_id = response.request_id
    return JudgeBatchResult(
        batch_id=request.batch_id,
        samples=request.samples,
        judgements=judgements,
        elapsed_seconds=round(time.perf_counter() - started, 6),
        estimated_tokens=request.estimated_tokens,
        estimated_cost_cny=request.estimated_cost_cny,
        usage=usage,
        request_id=request_id,
        budget_reservation=request.budget_reservation,
    )


def save_judge_batch_result(
    result: JudgeBatchResult,
    cache: "JudgeCache",
    indexed_judgements: dict[int, dict[str, Any]],
    batch_diagnostics: list[dict[str, Any]],
) -> None:
    for sample, judgement in zip(result.samples, result.judgements, strict=False):
        normalized = normalize_judgement(sample, judgement)
        sample_index = int(sample.get("_sample_index", len(indexed_judgements)))
        cache.set(sample, normalized)
        indexed_judgements[sample_index] = normalized
    batch_diagnostics.append(
        {
            "batch_id": result.batch_id,
            "sample_count": len(result.samples),
            "elapsed_seconds": result.elapsed_seconds,
            "estimated_tokens": result.estimated_tokens,
            "estimated_cost_cny": round(result.estimated_cost_cny, 6),
        }
    )


def failed_batch_row(request: JudgeBatchRequest, exc: Exception) -> dict[str, Any]:
    return {
        "batch_id": request.batch_id,
        "sample_count": len(request.samples),
        "query_ids": [str(sample.get("query_id", "")) for sample in request.samples],
        "item_ids": [str(sample.get("item_id", "")) for sample in request.samples],
        "error_type": type(exc).__name__,
        "error": str(exc)[:1000],
        "estimated_tokens": request.estimated_tokens,
        "estimated_cost_cny": round(request.estimated_cost_cny, 6),
    }


def settle_budget_success(budget_guard: ProviderBudgetGuard | None, result: JudgeBatchResult) -> None:
    if budget_guard is None or result.budget_reservation is None:
        return
    budget_guard.settle_success(
        result.budget_reservation,
        usage=result.usage or {},
        request_id=result.request_id,
    )



def settle_budget_failure(
    budget_guard: ProviderBudgetGuard | None,
    request: JudgeBatchRequest,
    exc: Exception,
) -> None:
    if budget_guard is None or request.budget_reservation is None:
        return
    budget_guard.settle_failure(request.budget_reservation, error=exc)


def build_budget_guard(args: Any) -> ProviderBudgetGuard:
    return ProviderBudgetGuard(
        client=budget_client_from_env(provider=str(getattr(args, "provider", "auto") or "auto")),
        hard_budget_cny=float(getattr(args, "hard_budget_cny", 0.0) or 0.0),
        ledger_path=Path(getattr(args, "usage_ledger", DEFAULT_LLM_USAGE_LEDGER)),
        safety_cny=float(getattr(args, "budget_safety_cny", 0.05) or 0.0),
        balance_check_interval_seconds=float(
            getattr(args, "balance_check_interval_seconds", 10.0) or 0.0
        ),
    )


def adaptive_judge_batches(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    concurrency: int,
    max_batch_size: int,
    max_batch_tokens: int,
    min_batch_size_for_concurrency: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    sizes = adaptive_judge_batch_sizes(
        len(rows),
        concurrency=concurrency,
        max_batch_size=max_batch_size,
        min_batch_size_for_concurrency=min_batch_size_for_concurrency,
    )
    batches = []
    offset = 0
    for size in sizes:
        batches.extend(split_batch_by_token_budget(rows[offset : offset + size], max_batch_tokens=max_batch_tokens))
        offset += size
    return batches


def adaptive_judge_batch_sizes(
    sample_count: int,
    *,
    concurrency: int,
    max_batch_size: int,
    min_batch_size_for_concurrency: int,
) -> list[int]:
    if sample_count <= 0:
        return []
    workers = max(1, concurrency)
    max_batch = max(1, max_batch_size)
    min_batch = max(1, min_batch_size_for_concurrency)
    if sample_count <= max_batch:
        return [sample_count]

    if sample_count < workers * min_batch:
        request_count = ceil_div(sample_count, min_batch)
    elif sample_count <= workers * max_batch:
        request_count = workers
    else:
        request_count = ceil_div(sample_count, max_batch)
    request_count = max(1, min(request_count, sample_count))
    base = sample_count // request_count
    remainder = sample_count % request_count
    return [base + (1 if index < remainder else 0) for index in range(request_count)]


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def split_batch_by_token_budget(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    max_batch_tokens: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    if not rows:
        return []
    budget = max(512, max_batch_tokens)
    batches: list[list[tuple[int, dict[str, Any]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    for sample_index, sample in rows:
        candidate = [*current, (sample_index, sample)]
        candidate_tokens = estimate_tokens(
            judge_prompt([{**candidate_sample, "_sample_index": candidate_index} for candidate_index, candidate_sample in candidate])
        )
        if current and candidate_tokens > budget:
            batches.append(current)
            current = [(sample_index, sample)]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def llm_generate_natural_fuzzy_command(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    cases = read_cases(
        Path(getattr(args, "dataset", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "test")),
        limit=int(getattr(args, "limit", 0)),
    )
    sample_size = int(getattr(args, "llm_sample_size", 0) or 0)
    require_llm = bool(getattr(args, "require_llm", False))
    budget = CostBudget(
        budget_cny=float(getattr(args, "budget_cny", 20.0)),
        cny_per_1k_tokens=float(getattr(args, "cny_per_1k_tokens", 0.01)),
    )
    hard_budget_cny = float(getattr(args, "hard_budget_cny", 0.0) or 0.0)
    budget_guard = build_budget_guard(args) if hard_budget_cny > 0 else None
    if budget_guard is not None and require_llm and sample_size > 0:
        budget_guard.preflight()
    rows: list[dict[str, Any]] = []
    llm_call_count = 0
    fallback_count = 0
    budget_stopped = False
    if require_llm and sample_size > 0:
        for batch in chunks(cases[:sample_size], max(1, int(getattr(args, "batch_size", 5) or 5))):
            prompt = natural_fuzzy_prompt(batch)
            tokens = estimate_tokens(prompt)
            cost = budget.estimate(tokens)
            if not budget.can_spend(cost):
                budget_stopped = True
                break
            llm_call_count += 1
            budget_reservation = None
            if budget_guard is not None:
                try:
                    budget_reservation = budget_guard.reserve(
                        batch_id=llm_call_count - 1,
                        sample_count=len(batch),
                        prompt_tokens_upper_bound=tokens,
                        max_completion_tokens=int(getattr(args, "max_tokens", 1200)),
                    )
                except Exception:
                    budget_stopped = True
                    llm_call_count -= 1
                    break
            try:
                if budget_reservation is None:
                    rows.extend(real_llm_natural_fuzzy(batch, prompt=prompt, max_tokens=int(getattr(args, "max_tokens", 1200)), timeout_seconds=float(getattr(args, "timeout_seconds", 180.0)), retries=int(getattr(args, "retries", 0))))
                else:
                    response = real_llm_natural_fuzzy_response(batch, prompt=prompt, max_tokens=int(getattr(args, "max_tokens", 1200)), timeout_seconds=float(getattr(args, "timeout_seconds", 180.0)), retries=int(getattr(args, "retries", 0)))
                    rows.extend(response.rows)
                    budget_guard.settle_success(budget_reservation, usage=response.usage, request_id=response.request_id)
            except Exception as exc:
                if budget_guard is not None and budget_reservation is not None:
                    budget_guard.settle_failure(budget_reservation, error=exc)
                fallback_count += len(batch)
                rows.extend(deterministic_natural_fuzzy_row(case) for case in batch)
            budget.spend(cost, tokens)
    else:
        rows = [deterministic_natural_fuzzy_row(case) for case in cases]
    output = Path(getattr(args, "output", DEFAULT_LLM_NATURAL_FUZZY_OUTPUT))
    write_jsonl(output, rows)
    summary = {
        "case_count": len(cases),
        "generated_count": len(rows),
        "llm_call_count": llm_call_count,
        "fallback_count": fallback_count,
        "budget_cny": budget.budget_cny,
        "estimated_tokens": budget.spent_tokens,
        "total_estimated_cost_cny": round(budget.spent_cny, 6),
        "budget_stopped": budget_stopped,
        "metadata_leak_count": sum(1 for row in rows if natural_fuzzy_has_label_leak(str(row.get("user_input", "")))),
        "output": str(output),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        **(budget_guard.summary() if budget_guard is not None else {"hard_budget_enabled": False}),
    }
    summary["metadata_leak_rate"] = round(summary["metadata_leak_count"] / max(1, len(rows)), 6)
    report = {
        "method": "retrieval_lab_llm_natural_fuzzy_generation",
        "summary": summary,
        "rows_preview": rows[:20],
        "fingerprint": data_sha256({"summary": summary, "rows": rows}),
    }
    report_output = Path(getattr(args, "report_output", DEFAULT_LLM_NATURAL_FUZZY_REPORT))
    write_json(report_output, report)
    return report


class CostBudget:
    def __init__(self, *, budget_cny: float, cny_per_1k_tokens: float) -> None:
        self.budget_cny = max(0.0, budget_cny)
        self.cny_per_1k_tokens = max(0.0, cny_per_1k_tokens)
        self.spent_cny = 0.0
        self.spent_tokens = 0

    def estimate(self, tokens: int) -> float:
        return tokens / 1000.0 * self.cny_per_1k_tokens

    def can_spend(self, cost: float) -> bool:
        return self.spent_cny + cost <= self.budget_cny

    def spend(self, cost: float, tokens: int) -> None:
        self.spent_cny += cost
        self.spent_tokens += tokens


class JudgeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows = self._load()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        row = self.rows.get(cache_key(sample))
        if row is None:
            self.miss_count += 1
            return None
        self.hit_count += 1
        return dict(row["judgement"])

    def set(self, sample: dict[str, Any], judgement: dict[str, Any]) -> None:
        key = cache_key(sample)
        row = {"key": key, "prompt_version": PROMPT_VERSION, "judgement": judgement}
        self.rows[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        rows = {}
        for row in read_jsonl(self.path):
            key = str(row.get("key", ""))
            if key:
                rows[key] = row
        return rows


def real_llm_judgements(
    samples: list[dict[str, Any]],
    *,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> list[dict[str, Any]]:
    from sceneweaver.llm.client import VisionLLMClient

    response = VisionLLMClient().analyze_text_json(
        system_prompt=(
            "You are a strict retrieval relevance judge. Return JSON only. "
            "Grade each candidate 0..3: 3 ideal, 2 usable, 1 weak, 0 irrelevant or violates constraints."
        ),
        user_prompt=prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return parse_llm_judgements(samples, response)


def real_llm_judgement_response(
    samples: list[dict[str, Any]],
    *,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> LLMJudgementResponse:
    from sceneweaver.llm.client import VisionLLMClient

    result = VisionLLMClient().analyze_text_json_result(
        system_prompt=(
            "You are a strict retrieval relevance judge. Return JSON only. "
            "Grade each candidate 0..3: 3 ideal, 2 usable, 1 weak, 0 irrelevant or violates constraints."
        ),
        user_prompt=prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return LLMJudgementResponse(
        judgements=parse_llm_judgements(samples, result.data),
        usage=result.usage,
        request_id=result.request_id,
    )


def parse_llm_judgements(samples: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("judgements") or response.get("judgments") or []
    if not isinstance(raw, list):
        raise ValueError("LLM qrels judge response must include judgements[]")
    rows = [row for row in raw if isinstance(row, dict)]
    by_key = {(str(row.get("query_id", "")), str(row.get("item_id", ""))): row for row in rows}
    by_index = {optional_int(row.get("sample_index", row.get("_sample_index"))): row for row in rows}
    judgements = []
    missing = []
    for local_index, sample in enumerate(samples):
        key = (str(sample.get("query_id", "")), str(sample.get("item_id", "")))
        sample_index = optional_int(sample.get("_sample_index"))
        judgement = by_key.get(key) or by_index.get(sample_index) or by_index.get(local_index)
        if judgement is None:
            missing.append({"query_id": key[0], "item_id": key[1], "sample_index": sample_index})
            continue
        judgements.append(judgement)
    if missing:
        preview = [
            {
                "query_id": row.get("query_id", ""),
                "item_id": row.get("item_id", ""),
                "sample_index": row.get("sample_index", row.get("_sample_index")),
            }
            for row in rows[:5]
        ]
        raise ValueError(
            "LLM qrels judge response did not cover every sample: "
            f"missing={len(missing)} returned={len(rows)} missing_preview={missing[:3]} response_preview={preview}"
        )
    return judgements


def real_llm_natural_fuzzy(
    cases: list[dict[str, Any]],
    *,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> list[dict[str, Any]]:
    from sceneweaver.llm.client import VisionLLMClient

    response = VisionLLMClient().analyze_text_json(
        system_prompt=(
            "You generate natural Chinese fuzzy retrieval queries. Return JSON only. "
            "Do not output internal snake_case labels."
        ),
        user_prompt=prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return parse_natural_fuzzy_rows(cases, response)


def real_llm_natural_fuzzy_response(
    cases: list[dict[str, Any]],
    *,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    retries: int,
) -> LLMNaturalFuzzyResponse:
    from sceneweaver.llm.client import VisionLLMClient

    result = VisionLLMClient().analyze_text_json_result(
        system_prompt=(
            "You generate natural Chinese fuzzy retrieval queries. Return JSON only. "
            "Do not output internal snake_case labels."
        ),
        user_prompt=prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return LLMNaturalFuzzyResponse(
        rows=parse_natural_fuzzy_rows(cases, result.data),
        usage=result.usage,
        request_id=result.request_id,
    )


def parse_natural_fuzzy_rows(cases: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    response_rows = natural_fuzzy_response_rows(response)
    by_id = {str(row.get("base_case_id", "")): row for row in response_rows if isinstance(row, dict)}
    single_response = response_rows[0] if len(response_rows) == 1 and isinstance(response_rows[0], dict) else {}
    rows = []
    for case in cases:
        base_id = str(case.get("case_id", ""))
        candidate = by_id.get(base_id, single_response if len(cases) == 1 else {})
        query = str(candidate.get("user_input", "") or candidate.get("query", ""))
        if not query or natural_fuzzy_has_label_leak(query):
            rows.append(deterministic_natural_fuzzy_row(case))
        else:
            rows.append(natural_fuzzy_row(case, query=query, source="llm"))
    return rows


def fake_judgement(sample: dict[str, Any]) -> dict[str, Any]:
    item_id = str(sample.get("item_id", ""))
    target = str(sample.get("target_item_id", ""))
    constraint_hits = sample.get("constraint_hits", {}) if isinstance(sample.get("constraint_hits", {}), dict) else {}
    if constraint_hits.get("negative_style") or constraint_hits.get("forbidden_stage"):
        return {"grade": 0, "reason": "candidate violates explicit negative constraint", "confidence": 0.86}
    if item_id and item_id == target:
        return {"grade": 3, "reason": "candidate is the generated target item", "confidence": 0.9}
    return {"grade": 2, "reason": "candidate is plausible from active retrieval pool", "confidence": 0.72}


def normalize_judgement(sample: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any]:
    grade = max(0, min(3, int(judgement.get("grade", judgement.get("score", 0)))))
    return {
        "query_id": str(sample.get("query_id", "")),
        "item_id": str(sample.get("item_id", "")),
        "grade": grade,
        "reason": str(judgement.get("reason", ""))[:500],
        "judge_type": "llm",
        "judge_id": str(judgement.get("judge_id", "retrieval_lab_llm_judge")),
        "judge_version": PROMPT_VERSION,
        "confidence": float(judgement.get("confidence", 0.75)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def judge_prompt(samples: list[dict[str, Any]]) -> str:
    rows = []
    for sample in samples:
        rows.append(
            {
                "sample_index": sample.get("_sample_index"),
                "query_id": sample.get("query_id", ""),
                "item_id": sample.get("item_id", ""),
                "query": sample.get("user_input", ""),
                "target_item_id": sample.get("target_item_id", ""),
                "candidate_metadata": sample.get("metadata", {}),
                "constraint_hits": sample.get("constraint_hits", {}),
                "pooled_from": sample.get("pooled_from", []),
            }
        )
    return json.dumps(
        {
            "instruction": (
                "Return exactly one JSON object with a judgements array. "
                "Return one judgement for every sample. Copy sample_index, query_id, and item_id exactly. "
                "Do not omit keys and do not rename query_id or item_id."
            ),
            "judgement_schema": {
                "judgements": [
                    {
                        "sample_index": "integer copied from sample_index",
                        "query_id": "string copied exactly",
                        "item_id": "string copied exactly",
                        "grade": "0..3",
                        "reason": "short relevance reason",
                        "confidence": "0..1",
                    }
                ]
            },
            "samples": rows,
        },
        ensure_ascii=False,
    )


def natural_fuzzy_prompt(cases: list[dict[str, Any]]) -> str:
    rows = []
    for case in cases:
        target = target_for_case(case)
        rows.append(
            {
                "base_case_id": case.get("case_id", ""),
                "title": target.get("title", ""),
                "natural_intent": natural_intent_for_case(case),
                "style": target.get("style", ""),
                "summary": target_summary_for_case(case),
            }
        )
    return json.dumps(
        {
            "instruction": "为每条样本写一个自然、模糊但可检索的中文 query，不要包含任何 snake_case 标签。",
            "output_contract": 'Return exactly one JSON object: {"queries":[{"base_case_id":"...","query":"..."}]}',
            "schema": {"queries": [{"base_case_id": "string", "query": "string"}]},
            "samples": rows,
        },
        ensure_ascii=False,
    )


def natural_fuzzy_response_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("queries") or response.get("rows") or response.get("items")
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        rows = []
        for key, value in raw.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("base_case_id", key)
                rows.append(row)
            else:
                rows.append({"base_case_id": key, "query": str(value)})
        return rows
    if "base_case_id" in response or "query" in response or "user_input" in response:
        return [response]
    raise ValueError("LLM natural fuzzy response must include queries[] or a query row")


def deterministic_natural_fuzzy_row(case: dict[str, Any]) -> dict[str, Any]:
    return natural_fuzzy_row(
        case,
        query=f"找一个自然克制、像真实观察到的段落，重点是{natural_intent_for_case(case)}。",
        source="deterministic",
    )


def natural_fuzzy_row(case: dict[str, Any], *, query: str, source: str) -> dict[str, Any]:
    base_id = str(case.get("case_id", ""))
    return {
        "base_case_id": base_id,
        "case_id": f"{base_id}::llm_natural_fuzzy",
        "variant_type": "llm_natural_fuzzy",
        "fuzzy_set_type": "natural_language",
        "user_input": strip_internal_labels(query),
        "source": source,
        "generator_version": NATURAL_FUZZY_PROMPT_VERSION,
    }


def natural_intent_for_case(case: dict[str, Any]) -> str:
    target = target_for_case(case)
    purposes = [str(value) for value in target.get("creative_purpose", []) or []]
    phrases = [NATURAL_PURPOSE_TEXT.get(value, value.replace("_", " ")) for value in purposes]
    if phrases:
        return "、".join(phrases)
    return "这一段的叙事功能"


def target_summary_for_case(case: dict[str, Any]) -> str:
    target = target_for_case(case)
    if isinstance(case.get("expected_prefer"), dict):
        parts = [
            str(target.get("title", "")),
            str(target.get("script_stage", "")),
            " ".join(str(value) for value in target.get("creative_purpose", []) or []),
            str(target.get("industry", "")),
            str(target.get("style", "")),
        ]
        return " ".join(part for part in parts if part).strip()
    return str(case.get("target_summary", ""))


def target_for_case(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_prefer")
    if isinstance(expected, dict):
        return expected
    target = case.get("target")
    return target if isinstance(target, dict) else {}


def natural_fuzzy_has_label_leak(text: str) -> bool:
    lower = str(text or "").lower()
    return any(label in lower for label in INTERNAL_LABELS)


def strip_internal_labels(text: str) -> str:
    result = str(text or "")
    for label in sorted(INTERNAL_LABELS, key=len, reverse=True):
        result = result.replace(label, "")
    return " ".join(result.split()).strip()


def cache_key(sample: dict[str, Any]) -> str:
    return data_sha256({"prompt_version": PROMPT_VERSION, "query_id": sample.get("query_id", ""), "item_id": sample.get("item_id", ""), "target": sample.get("target_item_id", "")})


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5) + 256)


def chunks(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


__all__ = [
    "DEFAULT_LLM_ADJUDICATION_OUTPUT",
    "DEFAULT_LLM_ADJUDICATION_REPORT",
    "DEFAULT_LLM_JUDGE_CACHE",
    "DEFAULT_LLM_NATURAL_FUZZY_OUTPUT",
    "DEFAULT_LLM_NATURAL_FUZZY_REPORT",
    "llm_adjudicate_qrels_command",
    "llm_generate_natural_fuzzy_command",
]
