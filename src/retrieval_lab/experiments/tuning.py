from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, write_json
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, case_fixture_id, read_cases
from retrieval_lab.indexes import index_items_from_cases, target_item_id
from retrieval_lab.planners import DEFAULT_PLANNER_CACHE_PATH
from retrieval_lab.qrels import canonical_stage
from retrieval_lab.retrieval.service import retrieval_run_from_cases


DEFAULT_CONSTRAINT_TUNING_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "constraint_tuning_latest.json"
DEFAULT_CONSTRAINT_PROFILE_PATH = Path(".tmp") / "retrieval_lab" / "constraint_profile_latest.json"
DEFAULT_LEAVE_ONE_FIXTURE_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "leave_one_fixture_latest.json"

DEFAULT_NATIVE_CONSTRAINT_WEIGHTS = {
    "rrf_weight": 10.0,
    "signature_weight": 0.35,
    "purpose_weight": 0.35,
    "desired_stage_bonus": 1.2,
    "forbidden_stage_penalty": 5.0,
    "negative_style_penalty": 1.5,
    "positive_style_bonus": 0.1,
}

NATIVE_TUNING_GRID = {
    "desired_stage_bonus": [0.0, 0.5, 0.9, 1.2, 1.8, 2.4],
    "forbidden_stage_penalty": [0.0, 1.0, 2.5, 5.0, 20.0],
    "negative_style_penalty": [0.0, 0.5, 1.0, 1.5, 3.0],
}


def tune_constraints_report(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    base_profile = load_native_constraint_profile(getattr(args, "constraint_profile", None))
    dataset = dataset_path(args)
    split = str(getattr(args, "split", "dev") or "dev")
    limit = int(getattr(args, "limit", 0) or 0)
    top_k = int(getattr(args, "top_k", 10) or 10)

    dev_cases = read_cases(dataset, split=split, limit=limit)
    test_cases = read_cases(dataset, split="test.md", limit=0)
    dev_rows = retrieval_rows_for_cases(args, dev_cases, run_name="native_tuning_dev")
    test_rows = retrieval_rows_for_cases(args, test_cases, run_name="native_tuning_test")

    tuning = tune_profile_from_rows(dev_rows, dev_cases, base_profile=base_profile, top_k=top_k)
    best_profile = tuning["best"]["profile"]
    dev_baseline = evaluate_rows_with_profile(dev_rows, dev_cases, base_profile, top_k=top_k, constraints_enabled=False)
    dev_after = evaluate_rows_with_profile(dev_rows, dev_cases, best_profile, top_k=top_k, constraints_enabled=True)
    test_baseline = evaluate_rows_with_profile(test_rows, test_cases, base_profile, top_k=top_k, constraints_enabled=False)
    test_after = evaluate_rows_with_profile(test_rows, test_cases, best_profile, top_k=top_k, constraints_enabled=True)

    profile_output = profile_output_path(args)
    profile_artifact = write_native_constraint_profile(
        profile_output,
        best_profile,
        metadata={
            "source_command": "tune-constraints",
            "dataset": str(dataset),
            "split": split,
            "limit": limit,
            "top_k": top_k,
            "selection_score": tuning["best"]["selection_score"],
        },
    )
    summary = {
        "command": "tune-constraints",
        "dataset": str(dataset),
        "split": split,
        "dev_case_count": len(dev_cases),
        "test_case_count": len(test_cases),
        "limit": limit,
        "top_k": top_k,
        "candidate_count": len(tuning["candidates"]),
        "profile_output": str(profile_output),
        "selected_weights": best_profile["weights"],
        "dev_selection_score": tuning["best"]["selection_score"],
        "possible_overfit": possible_overfit(
            dev_baseline["metrics"],
            dev_after["metrics"],
            test_baseline["metrics"],
            test_after["metrics"],
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_constraint_tuning",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "profile": best_profile,
        "profile_artifact": profile_artifact,
        "best": {
            "weights": best_profile["weights"],
            "selection_score": tuning["best"]["selection_score"],
            "metrics": tuning["best"]["metrics"],
        },
        "dev_baseline_metrics": dev_baseline["metrics"],
        "dev_metrics": dev_after["metrics"],
        "test_baseline_metrics": test_baseline["metrics"],
        "test_metrics_after_tuning": test_after["metrics"],
        "candidates": tuning["candidates"],
        "fingerprint": data_sha256({"summary": summary, "profile": best_profile, "candidates": tuning["candidates"]}),
    }


def leave_one_fixture_out_report(args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    base_profile = load_native_constraint_profile(getattr(args, "constraint_profile", None))
    dataset = dataset_path(args)
    split = str(getattr(args, "split", "all") or "all")
    limit = int(getattr(args, "limit", 0) or 0)
    top_k = int(getattr(args, "top_k", 10) or 10)
    cases = read_cases(dataset, split=split, limit=limit)
    rows = retrieval_rows_for_cases(args, cases, run_name="native_leave_one_fixture")
    fixtures = sorted({case_fixture_id(case) for case in cases if case_fixture_id(case)})

    fixture_reports = []
    for fixture_id in fixtures:
        dev_cases, dev_rows = filter_cases_and_rows(cases, rows, fixture_id, include=False)
        test_cases, test_rows = filter_cases_and_rows(cases, rows, fixture_id, include=True)
        tuning = tune_profile_from_rows(dev_rows, dev_cases, base_profile=base_profile, top_k=top_k)
        best_profile = tuning["best"]["profile"]
        test_report = evaluate_rows_with_profile(test_rows, test_cases, best_profile, top_k=top_k, constraints_enabled=True)
        baseline_report = evaluate_rows_with_profile(test_rows, test_cases, base_profile, top_k=top_k, constraints_enabled=False)
        fixture_reports.append(
            {
                "fixture_id": fixture_id,
                "dev_case_count": len(dev_cases),
                "test_case_count": len(test_cases),
                "selected_weights": best_profile["weights"],
                "selection_score": tuning["best"]["selection_score"],
                "test_baseline_metrics": baseline_report["metrics"],
                "test_metrics": test_report["metrics"],
            }
        )

    summary = {
        **summarize_leave_one_fixture(fixture_reports),
        "command": "evaluate-leave-one-fixture-out",
        "dataset": str(dataset),
        "split": split,
        "limit": limit,
        "top_k": top_k,
        "case_count": len(cases),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_leave_one_fixture_out",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "top_k": top_k,
        "fixture_count": len(fixture_reports),
        "summary": summary,
        "fixtures": fixture_reports,
        "fingerprint": data_sha256({"summary": summary, "fixtures": fixture_reports}),
    }


def retrieval_rows_for_cases(args: Any, cases: list[dict[str, Any]], *, run_name: str) -> list[dict[str, Any]]:
    if not cases:
        return []
    top_k = int(getattr(args, "top_k", 10) or 10)
    item_count = len(index_items_from_cases(cases))
    candidate_depth = max(int(getattr(args, "candidate_depth", 100) or 100), item_count, top_k)
    artifact = retrieval_run_from_cases(
        cases,
        dataset=str(dataset_path(args)),
        split=str(getattr(args, "split", "all") or "all"),
        limit=int(getattr(args, "limit", 0) or 0),
        planner=planner_name(args),
        planner_cache=planner_cache(args),
        top_k=top_k,
        candidate_depth=candidate_depth,
        run_name=run_name,
        ranking_key="hybrid_rrf",
        planner_config={"command": run_name},
    )
    run_rows = artifact.get("run_rows", {}) if isinstance(artifact, dict) else {}
    return list(next(iter(run_rows.values()), [])) if isinstance(run_rows, dict) else []


def tune_profile_from_rows(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    base_profile: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    candidates = []
    for desired_stage_bonus in NATIVE_TUNING_GRID["desired_stage_bonus"]:
        for forbidden_stage_penalty in NATIVE_TUNING_GRID["forbidden_stage_penalty"]:
            for negative_style_penalty in NATIVE_TUNING_GRID["negative_style_penalty"]:
                profile = profile_with_weights(
                    base_profile,
                    desired_stage_bonus=desired_stage_bonus,
                    forbidden_stage_penalty=forbidden_stage_penalty,
                    negative_style_penalty=negative_style_penalty,
                )
                report = evaluate_rows_with_profile(rows, cases, profile, top_k=top_k, constraints_enabled=True)
                candidates.append(
                    {
                        "weights": profile["weights"],
                        "profile": profile,
                        "metrics": report["metrics"],
                        "selection_score": tuning_selection_key(report["metrics"]),
                    }
                )
    if not candidates:
        profile = profile_with_weights(base_profile)
        metrics = evaluate_rows_with_profile(rows, cases, profile, top_k=top_k, constraints_enabled=True)["metrics"]
        candidates.append({"weights": profile["weights"], "profile": profile, "metrics": metrics, "selection_score": tuning_selection_key(metrics)})
    return {"best": max(candidates, key=lambda row: row["selection_score"]), "candidates": compact_candidates(candidates)}


def evaluate_rows_with_profile(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    top_k: int,
    constraints_enabled: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_map = {str(case.get("case_id", "")): case for case in cases}
    case_results = []
    for row in rows:
        case = case_map.get(str(row.get("case_id", "")), {})
        ranked = rerank_row_with_profile(row, profile, top_k=top_k, constraints_enabled=constraints_enabled)
        target_id = str(row.get("target_item_id", ""))
        expected_prefer_id = expected_prefer_item_id(case)
        target_rank = rank_of(ranked, target_id)
        expected_prefer_rank = rank_of(ranked, expected_prefer_id)
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id)
        case_results.append(
            {
                "case_id": row.get("case_id", ""),
                "case_type": str(case.get("case_type", row.get("variant_type", "default")) or "default"),
                "expected_relation": str(case.get("expected_relation", "should_match") or "should_match"),
                "query_constraints": row.get("query_constraints", {}),
                "target_item_id": target_id,
                "target_rank": target_rank,
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": expected_prefer_rank,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": margin(expected_prefer_score, target_score),
                "top_results": ranked[:top_k],
            }
        )
    elapsed = round(time.perf_counter() - started, 6)
    return {
        "method": "retrieval_lab_native_constraint_profile_evaluation",
        "case_count": len(case_results),
        "top_k": top_k,
        "constraints_enabled": constraints_enabled,
        "elapsed_seconds": elapsed,
        "seconds_per_case": round(elapsed / max(1, len(case_results)), 6),
        "metrics": build_metrics(case_results),
        "cases": case_results,
    }


def rerank_row_with_profile(
    row: dict[str, Any],
    profile: dict[str, Any],
    *,
    top_k: int,
    constraints_enabled: bool,
) -> list[dict[str, Any]]:
    candidates = list(row.get("all_results") or row.get("top_results") or [])
    weights = profile.get("weights", {}) if isinstance(profile.get("weights"), dict) else DEFAULT_NATIVE_CONSTRAINT_WEIGHTS
    scored = []
    for original_index, result in enumerate(candidates):
        score, components = profile_score(result, row=row, weights=weights, constraints_enabled=constraints_enabled)
        copied = dict(result)
        copied["score"] = round(score, 6)
        copied["final_score"] = round(score, 6)
        copied["ranking_key"] = "native_constraint_profile" if constraints_enabled else "native_unconstrained_baseline"
        copied["native_profile_components"] = components
        scored.append((score, original_index, copied))
    return [result for _score, _index, result in sorted(scored, key=lambda item: (-item[0], item[1]))]


def profile_score(
    result: dict[str, Any],
    *,
    row: dict[str, Any],
    weights: dict[str, Any],
    constraints_enabled: bool,
) -> tuple[float, dict[str, Any]]:
    rrf = numeric(result, "rrf_score")
    semantic = numeric(result, "embedding_score", "semantic_score")
    lexical = numeric(result, "lexical_score")
    signature = numeric(result, "signature_score")
    purpose = numeric(result, "purpose_score")
    base = (
        float(weights.get("rrf_weight", 10.0)) * rrf
        + float(weights.get("semantic_weight", 0.0)) * semantic
        + float(weights.get("lexical_weight", 0.0)) * lexical
        + float(weights.get("signature_weight", 0.35)) * signature
        + float(weights.get("purpose_weight", 0.35)) * purpose
    )
    components: dict[str, Any] = {
        "base": round(base, 6),
        "rrf": round(rrf, 6),
        "semantic": round(semantic, 6),
        "lexical": round(lexical, 6),
        "signature": round(signature, 6),
        "purpose": round(purpose, 6),
    }
    if not constraints_enabled:
        return base, components

    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    stage = canonical_stage(metadata.get("script_stage", ""))
    query_constraints = row.get("query_constraints", {}) if isinstance(row.get("query_constraints"), dict) else {}
    desired = {canonical_stage(value) for value in query_constraints.get("desired_stage", [])}
    forbidden = {canonical_stage(value) for value in query_constraints.get("forbidden_stage", [])}
    score = base
    if desired and stage in desired:
        score += float(weights.get("desired_stage_bonus", 1.2))
        components["desired_stage_bonus"] = float(weights.get("desired_stage_bonus", 1.2))
    if forbidden and stage in forbidden:
        penalty = float(weights.get("forbidden_stage_penalty", 5.0))
        score -= penalty
        components["forbidden_stage_penalty"] = penalty
    negative_hits = negative_style_hits(result, row)
    if negative_hits:
        penalty = float(weights.get("negative_style_penalty", 1.5)) * len(negative_hits)
        score -= penalty
        components["negative_style_penalty"] = round(penalty, 6)
        components["negative_style_hits"] = negative_hits
    positive_hits = positive_style_hits(result)
    if positive_hits:
        bonus = float(weights.get("positive_style_bonus", 0.1)) * len(positive_hits)
        score += bonus
        components["positive_style_bonus"] = round(bonus, 6)
        components["positive_style_hits"] = positive_hits
    return score, components


def build_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in case_results:
        by_type.setdefault(str(row.get("case_type", "default")), []).append(row)
    return {
        "overall": metric_group(case_results),
        "by_case_type": {case_type: metric_group(rows) for case_type, rows in sorted(by_type.items())},
    }


def metric_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if row.get("expected_relation") == "should_match"]
    negative_rows = [row for row in rows if row.get("expected_relation") == "should_not_match"]
    return {
        "count": len(rows),
        "recall_at_1": recall_at(positive_rows, 1),
        "recall_at_3": recall_at(positive_rows, 3),
        "recall_at_10": recall_at(positive_rows, 10),
        "mrr": mean_reciprocal_rank(positive_rows),
        "hard_negative_expected_prefer_margin_positive_rate": margin_positive_rate(negative_rows),
        "forbidden_stage_violation_at_1": forbidden_stage_violation_at(rows, 1),
        "forbidden_stage_violation_at_3": forbidden_stage_violation_at(rows, 3),
        "forbidden_stage_violation_at_10": forbidden_stage_violation_at(rows, 10),
        "desired_stage_hit_at_1": desired_stage_hit_at(rows, 1),
        "desired_stage_hit_at_3": desired_stage_hit_at(rows, 3),
    }


def summarize_leave_one_fixture(fixture_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not fixture_reports:
        return {
            "fixture_count": 0,
            "mean_hard_negative_margin_positive_rate": 0.0,
            "min_hard_negative_margin_positive_rate": 0.0,
            "mean_simple_positive_recall_at_3": 0.0,
            "mean_hard_positive_recall_at_10": 0.0,
            "worst_fixture_id": None,
            "possible_overfit": False,
        }
    hard_negative_rates = [metric_value(row["test_metrics"], "hard_negative", "hard_negative_expected_prefer_margin_positive_rate") for row in fixture_reports]
    if not any(hard_negative_rates):
        hard_negative_rates = [metric_value(row["test_metrics"], "negative", "hard_negative_expected_prefer_margin_positive_rate") for row in fixture_reports]
    simple_recall_at_3 = [metric_value(row["test_metrics"], "simple_positive", "recall_at_3") for row in fixture_reports]
    if not any(simple_recall_at_3):
        simple_recall_at_3 = [metric_value(row["test_metrics"], "positive", "recall_at_3") for row in fixture_reports]
    hard_positive_recall_at_10 = [metric_value(row["test_metrics"], "hard_positive", "recall_at_10") for row in fixture_reports]
    if not any(hard_positive_recall_at_10):
        hard_positive_recall_at_10 = [metric_value(row["test_metrics"], "positive", "recall_at_10") for row in fixture_reports]
    worst_index = min(range(len(fixture_reports)), key=lambda index: hard_negative_rates[index] if hard_negative_rates else 0.0)
    return {
        "fixture_count": len(fixture_reports),
        "mean_hard_negative_margin_positive_rate": round(mean(hard_negative_rates), 6),
        "min_hard_negative_margin_positive_rate": round(min(hard_negative_rates) if hard_negative_rates else 0.0, 6),
        "mean_simple_positive_recall_at_3": round(mean(simple_recall_at_3), 6),
        "mean_hard_positive_recall_at_10": round(mean(hard_positive_recall_at_10), 6),
        "worst_fixture_id": fixture_reports[worst_index]["fixture_id"] if fixture_reports else None,
        "possible_overfit": bool(hard_negative_rates and min(hard_negative_rates) < 0.5),
    }


def tuning_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    by_type = metrics.get("by_case_type", {}) if isinstance(metrics.get("by_case_type"), dict) else {}
    hard_negative = by_type.get("hard_negative") or by_type.get("negative") or metrics.get("overall", {})
    hard_positive = by_type.get("hard_positive") or by_type.get("positive") or metrics.get("overall", {})
    simple_positive = by_type.get("simple_positive") or by_type.get("positive") or metrics.get("overall", {})
    return (
        float(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", 0.0)),
        float(hard_positive.get("recall_at_10", 0.0)),
        float(simple_positive.get("recall_at_3", 0.0)),
    )


def possible_overfit(
    dev_baseline_metrics: dict[str, Any],
    dev_tuned_metrics: dict[str, Any],
    test_baseline_metrics: dict[str, Any],
    test_tuned_metrics: dict[str, Any],
) -> bool:
    return tuning_selection_key(dev_tuned_metrics) > tuning_selection_key(dev_baseline_metrics) and tuning_selection_key(test_tuned_metrics) < tuning_selection_key(test_baseline_metrics)


def recall_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row.get("target_rank") is not None and int(row["target_rank"]) <= k)
    return round(hits / len(rows), 6)


def mean_reciprocal_rank(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = sum(1 / int(row["target_rank"]) for row in rows if row.get("target_rank"))
    return round(total / len(rows), 6)


def margin_positive_rate(rows: list[dict[str, Any]]) -> float:
    margins = [float(row["expected_prefer_margin"]) for row in rows if row.get("expected_prefer_margin") is not None]
    if not margins:
        return 0.0
    return round(sum(1 for value in margins if value > 0) / len(margins), 6)


def forbidden_stage_violation_at(rows: list[dict[str, Any]], k: int) -> float:
    constrained = [row for row in rows if row.get("query_constraints", {}).get("forbidden_stage")]
    if not constrained:
        return 0.0
    violations = 0
    for row in constrained:
        forbidden = {canonical_stage(stage) for stage in row.get("query_constraints", {}).get("forbidden_stage", [])}
        if any(result_stage(result) in forbidden for result in row.get("top_results", [])[:k]):
            violations += 1
    return round(violations / len(constrained), 6)


def desired_stage_hit_at(rows: list[dict[str, Any]], k: int) -> float:
    constrained = [row for row in rows if row.get("query_constraints", {}).get("desired_stage")]
    if not constrained:
        return 0.0
    hits = 0
    for row in constrained:
        desired = {canonical_stage(stage) for stage in row.get("query_constraints", {}).get("desired_stage", [])}
        if any(result_stage(result) in desired for result in row.get("top_results", [])[:k]):
            hits += 1
    return round(hits / len(constrained), 6)


def result_stage(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    return canonical_stage(metadata.get("script_stage", ""))


def rank_of(rows: list[dict[str, Any]], item_id: str | None) -> int | None:
    if not item_id:
        return None
    for index, row in enumerate(rows, start=1):
        if row.get("item_id") == item_id:
            return index
    return None


def score_of(rows: list[dict[str, Any]], item_id: str | None) -> float | None:
    if not item_id:
        return None
    for row in rows:
        if row.get("item_id") == item_id:
            return float(row.get("score", 0.0))
    return None


def margin(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(left - right, 6)


def expected_prefer_item_id(case: dict[str, Any]) -> str | None:
    expected = case.get("expected_prefer") if isinstance(case.get("expected_prefer"), dict) else None
    return target_item_id(expected) if expected else None


def negative_style_hits(result: dict[str, Any], row: dict[str, Any]) -> list[str]:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    existing = [str(value) for value in hits.get("negative_style", []) if value]
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    item_risks = {str(value) for value in metadata.get("style_risks", []) or []}
    plan = row.get("query_plan", {}) if isinstance(row.get("query_plan"), dict) else {}
    requested = {str(value) for value in plan.get("negative_style", []) or []}
    return sorted(set(existing) | (item_risks & requested))


def positive_style_hits(result: dict[str, Any]) -> list[str]:
    hits = result.get("constraint_hits", {}) if isinstance(result.get("constraint_hits"), dict) else {}
    return [str(value) for value in hits.get("positive_style", []) if value]


def numeric(result: dict[str, Any], key: str, *fallback: str) -> float:
    for candidate in (key, *fallback):
        value = result.get(candidate)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def filter_cases_and_rows(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fixture_id: str,
    *,
    include: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_cases = [case for case in cases if (case_fixture_id(case) == fixture_id) is include]
    selected_ids = {str(case.get("case_id", "")) for case in selected_cases}
    selected_rows = [row for row in rows if str(row.get("case_id", "")) in selected_ids]
    return selected_cases, selected_rows


def profile_with_weights(base_profile: dict[str, Any], **overrides: float) -> dict[str, Any]:
    weights = dict(DEFAULT_NATIVE_CONSTRAINT_WEIGHTS)
    if isinstance(base_profile.get("weights"), dict):
        weights.update(base_profile["weights"])
    weights.update({key: value for key, value in overrides.items() if value is not None})
    return {
        "profile_id": "native_constraint_profile",
        "version": "retrieval_lab_constraint_profile_v1",
        "weights": weights,
        "grid": NATIVE_TUNING_GRID,
    }


def load_native_constraint_profile(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return profile_with_weights({})
    path = Path(path_value)
    if not str(path) or not path.exists() or not path.is_file():
        return profile_with_weights({})
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("profile"), dict):
        return profile_with_weights(data["profile"])
    if isinstance(data, dict) and isinstance(data.get("weights"), dict):
        return profile_with_weights(data)
    return profile_with_weights({})


def write_native_constraint_profile(path: Path, profile: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    artifact = {
        "method": "retrieval_lab_native_constraint_profile",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": profile,
        "metadata": metadata,
        "summary": {
            "profile_id": profile.get("profile_id", "native_constraint_profile"),
            "weight_count": len(profile.get("weights", {})),
            "output": str(path),
        },
    }
    artifact["fingerprint"] = data_sha256({"profile": profile, "metadata": metadata})
    write_json(path, artifact)
    return artifact


def compact_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "weights": row["weights"],
            "selection_score": row["selection_score"],
            "metrics": row["metrics"],
        }
        for row in candidates
    ]


def metric_value(metrics: dict[str, Any], case_type: str, metric_name: str) -> float:
    return float(metrics.get("by_case_type", {}).get(case_type, {}).get(metric_name, 0.0))


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def dataset_path(args: Any) -> Path:
    return Path(getattr(args, "dataset", DEFAULT_DATASET_PATH) or DEFAULT_DATASET_PATH)


def profile_output_path(args: Any) -> Path:
    explicit = getattr(args, "profile_output", None)
    if explicit:
        return Path(explicit)
    return DEFAULT_CONSTRAINT_PROFILE_PATH


def planner_name(args: Any) -> str:
    return str(getattr(args, "query_planner", "") or getattr(args, "planner", "") or "multi_query")


def planner_cache(args: Any) -> Path | None:
    if bool(getattr(args, "no_cache", False)):
        return None
    value = getattr(args, "planner_cache", DEFAULT_PLANNER_CACHE_PATH)
    return Path(value) if value else None


__all__ = [
    "DEFAULT_CONSTRAINT_PROFILE_PATH",
    "DEFAULT_CONSTRAINT_TUNING_REPORT_PATH",
    "DEFAULT_LEAVE_ONE_FIXTURE_REPORT_PATH",
    "NATIVE_TUNING_GRID",
    "build_metrics",
    "evaluate_rows_with_profile",
    "leave_one_fixture_out_report",
    "load_native_constraint_profile",
    "summarize_leave_one_fixture",
    "tune_constraints_report",
    "tune_profile_from_rows",
]
