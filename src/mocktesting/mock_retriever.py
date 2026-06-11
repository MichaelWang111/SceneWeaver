from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from mocktesting.constraint_layer import (
    DEFAULT_CONSTRAINT_PROFILE_PATH,
    load_constraint_profile,
    parse_query_constraints,
    profile_with_weights,
    score_constraints,
    write_constraint_profile,
)
from mocktesting.embedding_cache import (
    DEFAULT_CACHE_PATH,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    MAX_DASHSCOPE_BATCH_SIZE,
    EmbeddingCache,
)
from mocktesting.embedding_text_builder import (
    DEFAULT_CHANNEL_WEIGHTS,
    build_item_channel_rows,
    build_query_channels,
    load_review_items,
    target_channel_for_query,
    target_item_id,
)
from mocktesting.eval_input_generator import PURPOSE_WORDS, STAGE_WORDS, canonical_stage
from sceneweaver.retrieval.lexical import (
    DEFAULT_RRF_K,
    bm25_scores,
    ranked_indices_from_scores,
    reciprocal_rank_fusion,
    tokenize,
)
from sceneweaver.retrieval.query_plan import build_query_plan

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_embedding_index.json"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_retrieval_report.json"
DEFAULT_SEARCH_OUTPUT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_search_result.json"
DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "eval_inputs" / "review_generated_inputs.json"
DEFAULT_TUNING_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_constraint_tuning_report.json"
DEFAULT_LEAVE_ONE_FIXTURE_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_leave_one_fixture_report.json"
)
DEFAULT_RANKING_VALIDATION_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_ranking_key_validation_report.json"
)
DEFAULT_PARAPHRASE_STRESS_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_paraphrase_stress_report.json"
)
DEFAULT_WORKFLOW_COMPARISON_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_workflow_comparison_report.json"
)
DEFAULT_STYLE_NEGATIVE_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_style_negative_report.json"
)

VALID_SPLITS = {"dev", "test", "hidden", "all"}
RANKING_KEYS = (
    "final_score",
    "embedding_only",
    "semantic_only",
    "lexical_only",
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "script_use_only",
    "visual_tags_only",
    "experience_only",
    "combined_only",
    "constraints_only",
)

TUNING_GRID = {
    "desired_stage_bonus": [0.06, 0.10, 0.12, 0.16],
    "forbidden_stage_penalty": [0.10, 0.15, 0.18, 0.22, 0.28],
    "negative_constraint_penalty": [0.00, 0.05, 0.08, 0.12],
}
HARD_FORBIDDEN_STAGE_VETO = 1000.0
STYLE_POSITIVE_BONUS = 0.8
STYLE_NEGATIVE_PENALTY = 1.5
CONSTRAINT_RANKING_KEYS = {
    "final_score",
    "constraints_only",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MockTesting multi-channel embedding retriever.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index")
    add_common_paths(build_parser)
    build_parser.add_argument("--dry-run", action="store_true")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("user_input")
    add_common_paths(search_parser)
    add_constraint_args(search_parser)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--output", type=Path, default=DEFAULT_SEARCH_OUTPUT_PATH)

    eval_parser = subparsers.add_parser("evaluate")
    add_common_paths(eval_parser)
    add_constraint_args(eval_parser)
    eval_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    eval_parser.add_argument("--top-k", type=int, default=10)
    eval_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)

    tune_parser = subparsers.add_parser("tune-constraints")
    add_common_paths(tune_parser)
    tune_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    tune_parser.add_argument("--limit", type=int, default=0)
    tune_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="dev")
    tune_parser.add_argument("--top-k", type=int, default=10)
    tune_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    tune_parser.add_argument("--output", type=Path, default=DEFAULT_TUNING_REPORT_PATH)

    lofo_parser = subparsers.add_parser("evaluate-leave-one-fixture-out")
    add_common_paths(lofo_parser)
    lofo_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    lofo_parser.add_argument("--limit", type=int, default=0)
    lofo_parser.add_argument("--top-k", type=int, default=10)
    lofo_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    lofo_parser.add_argument("--output", type=Path, default=DEFAULT_LEAVE_ONE_FIXTURE_REPORT_PATH)

    validate_parser = subparsers.add_parser("validate-ranking-keys")
    add_common_paths(validate_parser)
    validate_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    validate_parser.add_argument("--limit", type=int, default=0)
    validate_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    validate_parser.add_argument("--top-k", type=int, default=10)
    validate_parser.add_argument("--llm-sample-size", type=int, default=0)
    validate_parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    validate_parser.add_argument("--llm-retries", type=int, default=0)
    validate_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_RANKING_VALIDATION_REPORT_PATH)

    hybrid_parser = subparsers.add_parser("evaluate-hybrid")
    add_common_paths(hybrid_parser)
    hybrid_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    hybrid_parser.add_argument("--limit", type=int, default=0)
    hybrid_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    hybrid_parser.add_argument("--top-k", type=int, default=10)
    hybrid_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    hybrid_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    hybrid_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)

    compare_parser = subparsers.add_parser("compare-ranking-workflows")
    add_common_paths(compare_parser)
    compare_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    compare_parser.add_argument("--limit", type=int, default=0)
    compare_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    compare_parser.add_argument("--top-k", type=int, default=10)
    compare_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    compare_parser.add_argument("--output", type=Path, default=DEFAULT_WORKFLOW_COMPARISON_REPORT_PATH)
    compare_parser.add_argument("--markdown-output", type=Path, default=None)

    style_parser = subparsers.add_parser("validate-style-negatives")
    add_common_paths(style_parser)
    style_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    style_parser.add_argument("--limit", type=int, default=0)
    style_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    style_parser.add_argument("--top-k", type=int, default=10)
    style_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    style_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    style_parser.add_argument("--output", type=Path, default=DEFAULT_STYLE_NEGATIVE_REPORT_PATH)

    report_parser = subparsers.add_parser("generate-eval-report")
    report_parser.add_argument("--input", type=Path, default=DEFAULT_WORKFLOW_COMPARISON_REPORT_PATH)
    report_parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "eval_outputs" / "mock_eval_report.md")

    paraphrase_parser = subparsers.add_parser("validate-paraphrase-stress")
    add_common_paths(paraphrase_parser)
    paraphrase_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    paraphrase_parser.add_argument("--limit", type=int, default=10)
    paraphrase_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    paraphrase_parser.add_argument(
        "--case-type",
        choices=["simple_positive", "hard_positive"],
        default="simple_positive",
    )
    paraphrase_parser.add_argument("--top-k", type=int, default=10)
    paraphrase_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="final_score")
    paraphrase_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    paraphrase_parser.add_argument("--output", type=Path, default=DEFAULT_PARAPHRASE_STRESS_REPORT_PATH)
    paraphrase_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "build-index":
        result = build_index_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "search":
        result = search_command(args)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        result = evaluate_command(args)
        write_json(args.output, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "tune-constraints":
        result = tune_constraints_command(args)
        write_json(args.output, result)
        print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-leave-one-fixture-out":
        result = leave_one_fixture_out_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-ranking-keys":
        result = validate_ranking_keys_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-hybrid":
        result = evaluate_hybrid_command(args)
        write_json(args.output, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "compare-ranking-workflows":
        result = compare_ranking_workflows_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-style-negatives":
        result = validate_style_negatives_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "generate-eval-report":
        result = json.loads(args.input.read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown_report(result), encoding="utf-8")
        print(str(args.output))
    elif args.command == "validate-paraphrase-stress":
        result = validate_paraphrase_stress_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--embedding-batch-size", type=int, default=MAX_DASHSCOPE_BATCH_SIZE)


def add_constraint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)


def build_index_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = build_item_channel_rows(load_review_items())
    texts = [channel["text"] for row in rows for channel in row["channels"] if channel.get("enabled", True)]
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    cache_stats = cache.embed_texts(texts, dry_run=args.dry_run)
    result = {
        "item_count": len(rows),
        "channel_count": len(texts),
        "cache": str(args.cache),
        "cache_stats": cache_stats,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return result
    index = {
        "model": args.model,
        "dimension": args.dimension,
        "channel_weights": DEFAULT_CHANNEL_WEIGHTS,
        "items": [
            {
                **row,
                "channels": [
                    {
                        **channel,
                        "embedding": cache.require_embedding(channel["text"]),
                    }
                    for channel in row["channels"]
                    if channel.get("enabled", True)
                ],
            }
            for row in rows
        ],
    }
    write_json(args.index, index)
    result["index"] = str(args.index)
    return result


def search_command(args: argparse.Namespace) -> dict[str, Any]:
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    profile = None if args.no_constraints else load_constraint_profile(args.constraint_profile)
    query_channels = build_query_channels(args.user_input)
    cache.embed_texts([channel["text"] for channel in query_channels])
    results = search_index(
        index,
        query_channels,
        cache,
        top_k=args.top_k,
        user_input=args.user_input,
        constraint_profile=profile,
        constraints_enabled=not args.no_constraints,
    )
    return {
        "user_input": args.user_input,
        "top_k": args.top_k,
        "constraints_enabled": not args.no_constraints,
        "constraint_profile": str(args.constraint_profile),
        "results": results,
    }


def evaluate_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = None if args.no_constraints else load_constraint_profile(args.constraint_profile)
    result = evaluate_cases(
        index_path=args.index,
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        embedding_batch_size=args.embedding_batch_size,
        inputs_path=args.inputs,
        limit=args.limit,
        split=args.split,
        top_k=args.top_k,
        constraint_profile=profile,
        constraint_profile_path=args.constraint_profile,
        constraints_enabled=not args.no_constraints,
    )
    return result


def tune_constraints_command(args: argparse.Namespace) -> dict[str, Any]:
    base_profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    dev_cases = read_cases(args.inputs, args.limit, split=args.split)
    test_cases = read_cases(args.inputs, 0, split="test")
    query_texts = [
        channel["text"]
        for case in dev_cases + test_cases
        for channel in build_query_channels(case["user_input"])
    ]
    cache.embed_texts(query_texts)
    dev_precomputed = precompute_embedding_rankings(index, cache, dev_cases)
    test_precomputed = precompute_embedding_rankings(index, cache, test_cases)

    tuning = tune_profile_from_precomputed(dev_precomputed, base_profile=base_profile, top_k=args.top_k)
    best = tuning["best"]
    best_profile = profile_with_weights(
        desired_stage_bonus=best["weights"]["desired_stage_bonus"],
        forbidden_stage_penalty=best["weights"]["forbidden_stage_penalty"],
        negative_constraint_penalty=best["weights"]["negative_constraint_penalty"],
        base_profile=base_profile,
    )
    write_constraint_profile(args.constraint_profile, best_profile)
    dev_baseline = evaluate_precomputed_cases(
        precomputed=dev_precomputed,
        top_k=args.top_k,
        constraint_profile=base_profile,
        constraints_enabled=False,
        include_cases=False,
    )
    test_baseline = evaluate_precomputed_cases(
        precomputed=test_precomputed,
        top_k=args.top_k,
        constraint_profile=base_profile,
        constraints_enabled=False,
        include_cases=False,
    )
    test_after = evaluate_precomputed_cases(
        precomputed=test_precomputed,
        top_k=args.top_k,
        constraint_profile=best_profile,
        constraints_enabled=True,
        include_cases=False,
    )
    return {
        "method": "mock_constraint_tuning",
        "split": args.split,
        "dev_case_count": len(dev_cases),
        "test_case_count": len(test_cases),
        "limit": args.limit,
        "top_k": args.top_k,
        "candidate_count": len(tuning["candidates"]),
        "profile_output": str(args.constraint_profile),
        "best": {
            "weights": best["weights"],
            "selection_score": best["selection_score"],
            "metrics": best["metrics"],
        },
        "selected_weights": best["weights"],
        "dev_baseline_metrics": dev_baseline["metrics"],
        "dev_metrics": best["metrics"],
        "test_baseline_metrics": test_baseline["metrics"],
        "test_metrics_after_tuning": test_after["metrics"],
        "possible_overfit": possible_overfit(
            dev_baseline["metrics"],
            best["metrics"],
            test_baseline["metrics"],
            test_after["metrics"],
        ),
        "candidates": tuning["candidates"],
    }


def leave_one_fixture_out_command(args: argparse.Namespace) -> dict[str, Any]:
    base_profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    cases = read_cases(args.inputs, args.limit, split="all")
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)
    fixture_ids = sorted({case_fixture_id(row["case"]) for row in precomputed})

    fixture_reports = []
    for fixture_id in fixture_ids:
        dev_precomputed = filter_precomputed_by_fixture(precomputed, fixture_id, include=False)
        test_precomputed = filter_precomputed_by_fixture(precomputed, fixture_id, include=True)
        tuning = tune_profile_from_precomputed(dev_precomputed, base_profile=base_profile, top_k=args.top_k)
        best = tuning["best"]
        best_profile = profile_with_weights(
            desired_stage_bonus=best["weights"]["desired_stage_bonus"],
            forbidden_stage_penalty=best["weights"]["forbidden_stage_penalty"],
            negative_constraint_penalty=best["weights"]["negative_constraint_penalty"],
            base_profile=base_profile,
        )
        test_report = evaluate_precomputed_cases(
            precomputed=test_precomputed,
            top_k=args.top_k,
            constraint_profile=best_profile,
            constraints_enabled=True,
            include_cases=False,
        )
        fixture_reports.append(
            {
                "fixture_id": fixture_id,
                "dev_case_count": len(dev_precomputed),
                "test_case_count": len(test_precomputed),
                "selected_weights": best["weights"],
                "test_metrics": test_report["metrics"],
            }
        )

    summary = summarize_leave_one_fixture(fixture_reports)
    return {
        "method": "mock_leave_one_fixture_out",
        "top_k": args.top_k,
        "fixture_count": len(fixture_reports),
        "summary": summary,
        "fixtures": fixture_reports,
    }


def validate_ranking_keys_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    cases = read_cases(args.inputs, args.limit, split=args.split)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)

    ranked_by_key = {
        key: rank_precomputed_cases(precomputed, ranking_key=key, constraint_profile=profile, top_k=args.top_k)
        for key in RANKING_KEYS
    }
    ranking_reports = {
        key: summarize_ranking_key_results(results)
        for key, results in ranked_by_key.items()
    }
    pairwise = build_pairwise_report(precomputed, ranked_by_key, index=index, top_k=args.top_k)
    margin = build_margin_report(ranked_by_key["final_score"])
    llm_sample = (
        run_llm_sample_judge(
            ranked_by_key=ranked_by_key,
            sample_size=args.llm_sample_size,
            timeout_seconds=args.llm_timeout_seconds,
            retries=args.llm_retries,
        )
        if args.llm_sample_size > 0
        else None
    )
    summary = {
        "case_count": len(cases),
        "split": args.split,
        "final_score_expected_prefer_vs_forbidden_accuracy": pair_metric(
            pairwise,
            "expected_prefer_vs_forbidden",
            "final_score",
            "accuracy",
        ),
        "embedding_only_expected_prefer_vs_forbidden_accuracy": pair_metric(
            pairwise,
            "expected_prefer_vs_forbidden",
            "embedding_only",
            "accuracy",
        ),
        "final_score_positive_vs_random_accuracy": pair_metric(
            pairwise,
            "positive_vs_random",
            "final_score",
            "accuracy",
        ),
        "final_score_positive_vs_wrong_stage_accuracy": pair_metric(
            pairwise,
            "positive_vs_wrong_stage",
            "final_score",
            "accuracy",
        ),
        "final_score_forbidden_stage_violation_at_3": ranking_reports["final_score"]["metrics"]["overall"][
            "forbidden_stage_violation_at_3"
        ],
        "low_confidence_rate": margin["low_confidence_rate"],
    }
    if llm_sample is not None:
        summary["llm_sample_precision_at_3"] = llm_sample.get("precision_at_3")
    return {
        "method": "mock_ranking_key_validation",
        "split": args.split,
        "case_count": len(cases),
        "top_k": args.top_k,
        "ranking_keys": ranking_reports,
        "pairwise": pairwise,
        "margin": margin,
        "llm_sample_judge": llm_sample,
        "worst_cases": select_worst_cases(ranked_by_key["final_score"], limit=20),
        "summary": summary,
    }


def evaluate_hybrid_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    cases = read_cases(args.inputs, args.limit, split=args.split)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)
    rows = rank_precomputed_cases(
        precomputed,
        ranking_key=args.ranking_key,
        constraint_profile=profile,
        top_k=args.top_k,
    )
    compact_rows = [{key: value for key, value in row.items() if key != "all_results"} for row in rows]
    return {
        "method": "mock_hybrid_retrieval",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "case_count": len(rows),
        "top_k": args.top_k,
        "metrics": build_metrics(compact_rows),
        "cases": compact_rows,
    }


def compare_ranking_workflows_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    cases = read_cases(args.inputs, args.limit, split=args.split)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)
    workflow_keys = [
        "semantic_only",
        "semantic_constraints",
        "lexical_only",
        "hybrid_rrf",
        "hybrid_rrf_constraints",
    ]
    key_map = {
        "semantic_constraints": "final_score",
    }
    ranked_by_workflow = {
        workflow: rank_precomputed_cases(
            precomputed,
            ranking_key=key_map.get(workflow, workflow),
            constraint_profile=profile,
            top_k=args.top_k,
        )
        for workflow in workflow_keys
    }
    reports = {
        workflow: summarize_ranking_key_results(rows)
        for workflow, rows in ranked_by_workflow.items()
    }
    baseline = reports["semantic_only"]["metrics"]
    best_workflow = max(
        reports,
        key=lambda workflow: workflow_selection_score(reports[workflow]["metrics"]),
    )
    return {
        "method": "mock_workflow_comparison",
        "split": args.split,
        "case_count": len(cases),
        "top_k": args.top_k,
        "workflows": reports,
        "summary": {
            "best_workflow": best_workflow,
            "best_selection_score": workflow_selection_score(reports[best_workflow]["metrics"]),
            "workflow_delta_vs_baseline": {
                workflow: workflow_delta(reports[workflow]["metrics"], baseline)
                for workflow in reports
            },
        },
        "worst_cases": {
            workflow: select_worst_cases(rows, limit=10)
            for workflow, rows in ranked_by_workflow.items()
        },
    }


def validate_style_negatives_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    index_items = {item["item_id"]: item for item in index["items"]}
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    source_cases = [
        case
        for case in read_cases(args.inputs, args.limit, split=args.split)
        if case["expected_relation"] == "should_match"
    ]
    generated_variants = [build_style_negative_case(case) for case in source_cases]
    skipped_cases = []
    variants = []
    for variant in generated_variants:
        target_hits = target_negative_style_hits(variant, index_items)
        if target_hits:
            skipped_cases.append(
                {
                    "case_id": variant["case_id"],
                    "target_item_id": target_item_id(variant["target"]),
                    "negative_style_hits": target_hits,
                }
            )
            continue
        variants.append(variant)
    query_texts = [channel["text"] for case in variants for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, variants)
    rows = rank_precomputed_cases(
        precomputed,
        ranking_key=args.ranking_key,
        constraint_profile=profile,
        top_k=args.top_k,
    )
    compact_rows = [{key: value for key, value in row.items() if key != "all_results"} for row in rows]
    return {
        "method": "mock_style_negative_validation",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "case_count": len(compact_rows),
        "generated_case_count": len(generated_variants),
        "skipped_target_style_violation_count": len(skipped_cases),
        "skipped_target_style_violation_cases": skipped_cases[:20],
        "top_k": args.top_k,
        "summary": style_negative_summary(compact_rows),
        "cases": compact_rows,
    }


def workflow_selection_score(metrics: dict[str, Any]) -> list[float]:
    overall = metrics.get("overall", {})
    by_type = metrics.get("by_case_type", {})
    hard_negative = by_type.get("hard_negative", {})
    return [
        float(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", 0.0)),
        float(overall.get("recall_at_10", 0.0)),
        -float(overall.get("forbidden_stage_violation_at_3", 0.0)),
    ]


def workflow_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    overall = metrics.get("overall", {})
    base_overall = baseline.get("overall", {})
    keys = ("recall_at_1", "recall_at_3", "recall_at_10", "forbidden_stage_violation_at_3")
    return {
        key: round(float(overall.get(key, 0.0)) - float(base_overall.get(key, 0.0)), 6)
        for key in keys
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Mock Retrieval Evaluation Report",
        "",
        f"- method: `{report.get('method', '')}`",
        f"- split: `{report.get('split', '')}`",
        f"- case_count: `{report.get('case_count', 0)}`",
        f"- top_k: `{report.get('top_k', 0)}`",
        "",
    ]
    summary = report.get("summary", {})
    if summary:
        lines.extend(["## Summary", ""])
        if summary.get("best_workflow") is not None:
            lines.append(f"- best_workflow: `{summary.get('best_workflow', '')}`")
        if summary.get("best_selection_score") is not None:
            lines.append(f"- best_selection_score: `{summary.get('best_selection_score', '')}`")
        metric_rows = simple_metric_rows(summary, skip={"best_workflow", "best_selection_score"})
        if metric_rows:
            lines.extend(["", "| metric | value |", "|---|---:|"])
            lines.extend(f"| {key} | {value} |" for key, value in metric_rows)
        lines.append("")
    metrics = report.get("metrics", {})
    if metrics:
        overall_metrics = metrics.get("overall", metrics)
        metric_rows = simple_metric_rows(overall_metrics)
        if metric_rows:
            lines.extend(["## Metrics", "", "| metric | value |", "|---|---:|"])
            lines.extend(f"| {key} | {value} |" for key, value in metric_rows)
            lines.append("")
    workflows = report.get("workflows", {})
    if workflows:
        lines.extend(["## Workflow Metrics", "", "| workflow | recall@1 | recall@3 | recall@10 | hard_neg_prefer | forbidden@3 |", "|---|---:|---:|---:|---:|---:|"])
        for workflow, row in workflows.items():
            metrics = row.get("metrics", {})
            overall = metrics.get("overall", {})
            hard_negative = metrics.get("by_case_type", {}).get("hard_negative", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        workflow,
                        str(overall.get("recall_at_1", "")),
                        str(overall.get("recall_at_3", "")),
                        str(overall.get("recall_at_10", "")),
                        str(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", "")),
                        str(overall.get("forbidden_stage_violation_at_3", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- LLM judgement is not run by default; use explicit sample flags for token-safe spot checks.",
            "- Treat this report as an experiment comparison, not a production default decision by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def simple_metric_rows(data: dict[str, Any], *, skip: set[str] | None = None) -> list[tuple[str, str]]:
    skip = skip or set()
    rows = []
    for key, value in data.items():
        if key in skip or isinstance(value, (dict, list)):
            continue
        rows.append((key, str(value)))
    return rows


def validate_paraphrase_stress_command(args: argparse.Namespace) -> dict[str, Any]:
    source_cases = [
        case
        for case in read_cases(args.inputs, 0, split=args.split)
        if case["case_type"] == args.case_type and case["expected_relation"] == "should_match"
    ]
    if args.limit > 0:
        source_cases = source_cases[: args.limit]
    variants = [
        variant
        for case in source_cases
        for variant in build_paraphrase_variants(case)
    ]
    if args.dry_run:
        return {
            "method": "mock_paraphrase_stress",
            "dry_run": True,
            "split": args.split,
            "case_type": args.case_type,
            "source_case_count": len(source_cases),
            "variant_count": len(variants),
            "summary": {
                "source_case_count": len(source_cases),
                "variant_count": len(variants),
                "variant_types": sorted({variant["variant_type"] for variant in variants}),
            },
            "cases": variants,
        }

    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
    )
    query_texts = [channel["text"] for variant in variants for channel in build_query_channels(variant["user_input"])]
    cache.embed_texts(query_texts)

    rows = []
    for variant in variants:
        query_channels = build_query_channels(variant["user_input"])
        ranked = search_index(
            index,
            query_channels,
            cache,
            top_k=max(args.top_k, len(index["items"])),
            user_input=variant["user_input"],
            constraint_profile=profile,
            constraints_enabled=True,
        )
        if args.ranking_key != "final_score":
            query_constraints = parse_query_constraints(variant["user_input"], profile)
            ranked = rank_items_for_key(
                ranked,
                ranking_key=args.ranking_key,
                user_input=variant["user_input"],
                query_constraints=query_constraints,
                constraint_profile=profile,
            )
        target_id = target_item_id(variant["target"])
        target_stage = canonical_stage(variant["target"].get("script_stage"))
        target_purposes = set(variant["target"].get("creative_purpose", []))
        top_results = ranked[: args.top_k]
        rows.append(
            {
                "case_id": variant["case_id"],
                "source_case_id": variant["source_case_id"],
                "variant_type": variant["variant_type"],
                "user_input": variant["user_input"],
                "target_item_id": target_id,
                "target_stage": target_stage,
                "target_purposes": list(target_purposes),
                "target_rank": rank_of(ranked, target_id),
                "target_score": score_of(ranked, target_id),
                "stage_hit_at_1": bool(top_results and result_stage(top_results[0]) == target_stage),
                "stage_hit_at_3": any(result_stage(result) == target_stage for result in top_results[:3]),
                "purpose_hit_at_3": purpose_hit_at(top_results, target_purposes, 3),
                "top1_top2_margin": round(top1_top2_margin({"top_results": top_results}), 6),
                "confidence": confidence_bucket(top1_top2_margin({"top_results": top_results})),
                "top_results": top_results,
            }
        )
    return {
        "method": "mock_paraphrase_stress",
        "split": args.split,
        "case_type": args.case_type,
        "source_case_count": len(source_cases),
        "variant_count": len(rows),
        "top_k": args.top_k,
        "ranking_key": args.ranking_key,
        "summary": summarize_paraphrase_rows(rows),
        "by_variant_type": {
            variant_type: summarize_paraphrase_rows([row for row in rows if row["variant_type"] == variant_type])
            for variant_type in sorted({row["variant_type"] for row in rows})
        },
        "by_source_case": summarize_paraphrase_by_source_case(rows),
        "cases": rows,
    }


def evaluate_cases(
    *,
    index_path: Path,
    cache_path: Path,
    model: str,
    dimension: int,
    embedding_batch_size: int,
    inputs_path: Path,
    limit: int,
    split: str,
    top_k: int,
    constraint_profile: dict[str, Any] | None,
    constraint_profile_path: Path,
    constraints_enabled: bool,
    include_cases: bool = True,
) -> dict[str, Any]:
    index = read_index(index_path)
    cache = EmbeddingCache(
        cache_path=cache_path,
        model=model,
        dimension=dimension,
        batch_size=embedding_batch_size,
    )
    cases = read_cases(inputs_path, limit, split=split)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    result = evaluate_loaded_cases(
        index=index,
        cache=cache,
        cases=cases,
        top_k=top_k,
        constraint_profile=constraint_profile,
        constraints_enabled=constraints_enabled,
        include_cases=include_cases,
    )
    result["split"] = split
    result["constraint_profile"] = str(constraint_profile_path)
    return result


def evaluate_loaded_cases(
    *,
    index: dict[str, Any],
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
    top_k: int,
    constraint_profile: dict[str, Any] | None,
    constraints_enabled: bool,
    include_cases: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    case_results = []
    for case in cases:
        query_channels = build_query_channels(case["user_input"])
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = search_index(
            index,
            query_channels,
            cache,
            top_k=max(top_k, len(index["items"])),
            user_input=case["user_input"],
            constraint_profile=constraint_profile,
            constraints_enabled=constraints_enabled,
        )
        target_id = target_item_id(case["target"])
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_rank = rank_of(ranked, target_id)
        expected_prefer_rank = rank_of(ranked, expected_prefer_id) if expected_prefer_id else None
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id) if expected_prefer_id else None
        case_results.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "user_input": case["user_input"],
                "query_constraints": query_constraints,
                "target_item_id": target_id,
                "target_rank": target_rank,
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": expected_prefer_rank,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": ranked[:top_k],
            }
        )
    elapsed = round(time.perf_counter() - started_at, 3)
    result = {
        "method": "mock_multi_channel_embedding_retrieval",
        "case_count": len(case_results),
        "top_k": top_k,
        "constraints_enabled": constraints_enabled,
        "elapsed_seconds": elapsed,
        "seconds_per_case": round(elapsed / max(1, len(case_results)), 6),
        "metrics": build_metrics(case_results),
    }
    if include_cases:
        result["cases"] = case_results
    return result


def read_cases(path: Path, limit: int, *, split: str = "all") -> list[dict[str, Any]]:
    cases = split_cases(json.loads(path.read_text(encoding="utf-8"))["cases"], split)
    if limit > 0:
        return cases[:limit]
    return cases


def split_cases(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if split == "all":
        return list(cases)
    return [case for case in cases if case_split(case["case_id"]) == split]


def case_split(case_id: str) -> str:
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket <= 3:
        return "dev"
    if bucket <= 7:
        return "test"
    return "hidden"


def filter_cases_by_fixture(
    cases: list[dict[str, Any]],
    fixture_id: str,
    *,
    include: bool = True,
) -> list[dict[str, Any]]:
    return [case for case in cases if (case_fixture_id(case) == fixture_id) is include]


def case_fixture_id(case: dict[str, Any]) -> str:
    return str(case["target"]["fixture_id"])


def precompute_embedding_rankings(
    index: dict[str, Any],
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        query_channels = build_query_channels(case["user_input"])
        ranked = search_index(
            index,
            query_channels,
            cache,
            top_k=len(index["items"]),
            constraints_enabled=False,
        )
        rows.append({"case": case, "embedding_ranked": ranked})
    return rows


def tune_profile_from_precomputed(
    precomputed: list[dict[str, Any]],
    *,
    base_profile: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    candidates = []
    for desired_stage_bonus in TUNING_GRID["desired_stage_bonus"]:
        for forbidden_stage_penalty in TUNING_GRID["forbidden_stage_penalty"]:
            for negative_constraint_penalty in TUNING_GRID["negative_constraint_penalty"]:
                profile = profile_with_weights(
                    desired_stage_bonus=desired_stage_bonus,
                    forbidden_stage_penalty=forbidden_stage_penalty,
                    negative_constraint_penalty=negative_constraint_penalty,
                    base_profile=base_profile,
                )
                report = evaluate_precomputed_cases(
                    precomputed=precomputed,
                    top_k=top_k,
                    constraint_profile=profile,
                    constraints_enabled=True,
                    include_cases=False,
                )
                candidates.append(
                    {
                        "weights": profile["weights"],
                        "metrics": report["metrics"],
                        "selection_score": tuning_selection_key(report["metrics"]),
                    }
                )
    return {"best": select_best_tuning_result(candidates), "candidates": candidates}


def filter_precomputed_by_fixture(
    rows: list[dict[str, Any]],
    fixture_id: str,
    *,
    include: bool = True,
) -> list[dict[str, Any]]:
    return [row for row in rows if (case_fixture_id(row["case"]) == fixture_id) is include]


def evaluate_precomputed_cases(
    *,
    precomputed: list[dict[str, Any]],
    top_k: int,
    constraint_profile: dict[str, Any],
    constraints_enabled: bool = True,
    include_cases: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    case_results = []
    for row in precomputed:
        case = row["case"]
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = (
            rerank_with_constraints(row["embedding_ranked"], query_constraints, constraint_profile)
            if constraints_enabled
            else row["embedding_ranked"]
        )
        target_id = target_item_id(case["target"])
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_rank = rank_of(ranked, target_id)
        expected_prefer_rank = rank_of(ranked, expected_prefer_id) if expected_prefer_id else None
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id) if expected_prefer_id else None
        case_results.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "query_constraints": query_constraints,
                "target_item_id": target_id,
                "target_rank": target_rank,
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": expected_prefer_rank,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": ranked[:top_k],
            }
        )
    elapsed = round(time.perf_counter() - started_at, 3)
    result = {
        "method": "mock_constraint_tuning_retrieval",
        "case_count": len(case_results),
        "top_k": top_k,
        "constraints_enabled": constraints_enabled,
        "elapsed_seconds": elapsed,
        "seconds_per_case": round(elapsed / max(1, len(case_results)), 6),
        "metrics": build_metrics(case_results),
    }
    if include_cases:
        result["cases"] = case_results
    return result


def rank_precomputed_cases(
    precomputed: list[dict[str, Any]],
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for row in precomputed:
        case = row["case"]
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = rank_items_for_key(
            row["embedding_ranked"],
            ranking_key=ranking_key,
            user_input=case["user_input"],
            query_constraints=query_constraints,
            constraint_profile=constraint_profile,
        )
        target_id = target_item_id(case["target"])
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id) if expected_prefer_id else None
        rows.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "query_constraints": query_constraints,
                "target_item_id": target_id,
                "target_rank": rank_of(ranked, target_id),
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": rank_of(ranked, expected_prefer_id) if expected_prefer_id else None,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": ranked[:top_k],
                "all_results": ranked,
            }
        )
    return rows


def rank_items_for_key(
    embedding_ranked: list[dict[str, Any]],
    *,
    ranking_key: str,
    user_input: str = "",
    query_constraints: dict[str, Any],
    constraint_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    lexical_scores = lexical_scores_for_ranked(user_input, embedding_ranked)
    semantic_scores = [float(row["embedding_score"]) for row in embedding_ranked]
    rrf_scores = reciprocal_rank_fusion(
        [
            ranked_indices_from_scores(semantic_scores),
            ranked_indices_from_scores(lexical_scores),
        ],
        item_count=len(embedding_ranked),
        k=DEFAULT_RRF_K,
    )
    ranked = []
    for index, row in enumerate(embedding_ranked):
        constraint_score, constraint_hits = score_constraints(
            query_constraints,
            row.get("metadata", {}),
            constraint_profile,
        )
        style_score, style_hits = score_mock_style_constraints(user_input, row)
        full_constraint_score = constraint_score + style_score
        full_constraint_hits = {**constraint_hits}
        full_constraint_hits.update(style_hits)
        if ranking_key == "final_score":
            score = float(row["embedding_score"]) + full_constraint_score
        elif ranking_key in {"embedding_only", "semantic_only"}:
            score = float(row["embedding_score"])
        elif ranking_key == "lexical_only":
            score = lexical_scores[index]
        elif ranking_key == "hybrid_rrf":
            score = rrf_scores[index] * 100
        elif ranking_key in {"hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}:
            score = (rrf_scores[index] * 100) + full_constraint_score
        elif ranking_key == "constraints_only":
            score = full_constraint_score
        elif ranking_key.endswith("_only"):
            channel = ranking_key.removesuffix("_only")
            score = float(row.get("channel_scores", {}).get(channel, 0.0))
        else:
            raise ValueError(f"Unknown ranking key: {ranking_key}")
        if ranking_key in CONSTRAINT_RANKING_KEYS and has_forbidden_stage_hit(query_constraints, row.get("metadata", {})):
            score -= HARD_FORBIDDEN_STAGE_VETO
        ranked.append(
            {
                **row,
                "score": round(score, 6),
                "final_score": round(score, 6),
                "ranking_key": ranking_key,
                "lexical_score": round(lexical_scores[index], 6),
                "rrf_score": round(rrf_scores[index] * 100, 6),
                "constraint_score": round(full_constraint_score, 6),
                "constraint_hits": full_constraint_hits,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def summarize_ranking_key_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact_rows = [
        {key: value for key, value in row.items() if key != "all_results"}
        for row in rows
    ]
    return {
        "metrics": build_metrics(compact_rows),
        "case_count": len(compact_rows),
    }


def lexical_scores_for_ranked(user_input: str, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    plan = build_query_plan(user_input)
    query_text = " ".join(
        part
        for part in (
            plan.positive_query,
            " ".join(plan.desired_stage),
            " ".join(plan.positive_purposes),
            " ".join(plan.positive_style),
            " ".join(plan.visual_hints),
        )
        if part
    )
    docs = [tokenize(row.get("lexical_text") or mock_item_lexical_text(row)) for row in rows]
    return bm25_scores(tokenize(query_text), docs)


def score_mock_style_constraints(user_input: str, row: dict[str, Any]) -> tuple[float, dict[str, list[str]]]:
    plan = build_query_plan(user_input)
    text = (row.get("lexical_text") or mock_item_lexical_text(row)).lower()
    score = 0.0
    hits: dict[str, list[str]] = {}
    negative_hits = [style for style in plan.negative_style if style in text or style_alias_hit(style, text)]
    if negative_hits:
        score -= STYLE_NEGATIVE_PENALTY * len(negative_hits)
        hits["negative_style"] = negative_hits
    positive_hits = [style for style in plan.positive_style if style in text or style_alias_hit(style, text)]
    if positive_hits:
        score += STYLE_POSITIVE_BONUS * len(positive_hits)
        hits["positive_style"] = positive_hits
    return round(score, 6), hits


def style_alias_hit(style: str, text: str) -> bool:
    aliases = {
        "big_company_office": ("大厂", "互联网大厂", "泛泛办公", "空泛办公", "generic office"),
        "ad_like": ("广告", "宣传片腔", "硬广", "口号", "slogan"),
        "tech_showoff": ("炫技", "技术炫耀", "功能说明", "产品说明", "纯科技", "冷冰冰"),
        "human_warmth": ("有人味", "人味", "人的温度", "human"),
        "documentary": ("纪录片", "纪实", "观察", "documentary"),
        "real_location": ("真实现场", "真实场景", "现场感"),
    }
    return any(alias.lower() in text for alias in aliases.get(style, ()))


def target_negative_style_hits(case: dict[str, Any], index_items: dict[str, dict[str, Any]]) -> list[str]:
    item = index_items.get(target_item_id(case["target"]))
    if item is None:
        return []
    row = {
        "metadata": item.get("metadata", {}),
        "lexical_text": index_item_lexical_text(item),
    }
    _score, hits = score_mock_style_constraints(case["user_input"], row)
    return list(hits.get("negative_style", []))


def has_forbidden_stage_hit(query_constraints: dict[str, Any], metadata: dict[str, Any]) -> bool:
    forbidden = {canonical_stage(stage) for stage in query_constraints.get("forbidden_stage", [])}
    return bool(forbidden and canonical_stage(metadata.get("script_stage", "")) in forbidden)


def mock_item_lexical_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    parts = [
        metadata.get("script_stage", ""),
        " ".join(metadata.get("creative_purpose", [])),
        metadata.get("script_use_sentence", ""),
        metadata.get("industry", ""),
        metadata.get("style", ""),
        row.get("lexical_text", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def build_pairwise_report(
    precomputed: list[dict[str, Any]],
    ranked_by_key: dict[str, list[dict[str, Any]]],
    *,
    index: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    pairs = build_pairwise_pairs(precomputed, index=index, top_k=top_k)
    report: dict[str, Any] = {}
    by_case_and_key = {
        key: {row["case_id"]: row for row in rows}
        for key, rows in ranked_by_key.items()
    }
    for pair_type, type_pairs in pairs.items():
        report[pair_type] = {}
        for ranking_key in RANKING_KEYS:
            key_rows = by_case_and_key[ranking_key]
            correct = 0
            scored = 0
            skipped = 0
            margins: list[float] = []
            for pair in type_pairs:
                row = key_rows.get(pair["case_id"])
                if row is None:
                    skipped += 1
                    continue
                better_score = score_of(row["all_results"], pair["better_item_id"])
                worse_score = score_of(row["all_results"], pair["worse_item_id"])
                if better_score is None or worse_score is None:
                    skipped += 1
                    continue
                margin = round(better_score - worse_score, 6)
                margins.append(margin)
                scored += 1
                if margin > 0:
                    correct += 1
            report[pair_type][ranking_key] = {
                "count": scored,
                "correct": correct,
                "skipped_pairs": skipped,
                "accuracy": round(correct / scored, 6) if scored else 0.0,
                "mean_margin": round(mean(margins), 6) if margins else 0.0,
            }
    return report


def build_pairwise_pairs(
    precomputed: list[dict[str, Any]],
    *,
    index: dict[str, Any],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    index_items = {item["item_id"]: item for item in index["items"]}
    pairs: dict[str, list[dict[str, Any]]] = {
        "positive_vs_random": [],
        "positive_vs_wrong_stage": [],
        "expected_prefer_vs_forbidden": [],
    }
    for row in precomputed:
        case = row["case"]
        target_id = target_item_id(case["target"])
        if case["case_type"] in {"simple_positive", "hard_positive"}:
            random_negative = find_random_negative_item_id(index_items, case)
            if random_negative:
                pairs["positive_vs_random"].append(
                    {"case_id": case["case_id"], "better_item_id": target_id, "worse_item_id": random_negative}
                )
            wrong_stage = find_visually_similar_wrong_stage_item_id(row["embedding_ranked"], case, top_k=top_k)
            if wrong_stage:
                pairs["positive_vs_wrong_stage"].append(
                    {"case_id": case["case_id"], "better_item_id": target_id, "worse_item_id": wrong_stage}
                )
        elif case["case_type"] == "hard_negative" and case.get("expected_prefer"):
            pairs["expected_prefer_vs_forbidden"].append(
                {
                    "case_id": case["case_id"],
                    "better_item_id": target_item_id(case["expected_prefer"]),
                    "worse_item_id": target_id,
                }
            )
    return pairs


def find_random_negative_item_id(index_items: dict[str, dict[str, Any]], case: dict[str, Any]) -> str | None:
    target = case["target"]
    for item_id, item in sorted(index_items.items()):
        metadata = item["metadata"]
        if metadata.get("fixture_id") == target.get("fixture_id"):
            continue
        if canonical_stage(metadata.get("script_stage")) == canonical_stage(target.get("script_stage")):
            continue
        return item_id
    return None


def find_visually_similar_wrong_stage_item_id(
    embedding_ranked: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    top_k: int,
) -> str | None:
    target_id = target_item_id(case["target"])
    target_stage = canonical_stage(case["target"].get("script_stage"))
    for row in embedding_ranked[:top_k]:
        if row["item_id"] == target_id:
            continue
        if canonical_stage(row.get("metadata", {}).get("script_stage")) != target_stage:
            return row["item_id"]
    return None


def pair_metric(pairwise: dict[str, Any], pair_type: str, ranking_key: str, metric_name: str) -> float:
    return float(pairwise.get(pair_type, {}).get(ranking_key, {}).get(metric_name, 0.0))


def build_margin_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [case_margin(row) for row in rows]
    buckets = {"low": 0, "medium": 0, "high": 0}
    for margin in margins:
        buckets[confidence_bucket(margin)] += 1
    count = len(margins)
    expected_prefer_margins = [
        row["expected_prefer_margin"]
        for row in rows
        if row.get("expected_prefer_margin") is not None
    ]
    top1_top2_margins = [top1_top2_margin(row) for row in rows if len(row.get("top_results", [])) > 1]
    return {
        "count": count,
        "low_confidence_rate": round(buckets["low"] / count, 6) if count else 0.0,
        "medium_confidence_rate": round(buckets["medium"] / count, 6) if count else 0.0,
        "high_confidence_rate": round(buckets["high"] / count, 6) if count else 0.0,
        "mean_margin": round(mean(margins), 6) if margins else 0.0,
        "mean_top1_top2_margin": round(mean(top1_top2_margins), 6) if top1_top2_margins else 0.0,
        "mean_expected_prefer_minus_forbidden_margin": (
            round(mean(expected_prefer_margins), 6) if expected_prefer_margins else 0.0
        ),
        "cases": [
            {
                "case_id": row["case_id"],
                "case_type": row["case_type"],
                "margin": round(case_margin(row), 6),
                "confidence": confidence_bucket(case_margin(row)),
                "top1_item_id": row.get("top_results", [{}])[0].get("item_id") if row.get("top_results") else None,
            }
            for row in rows
        ],
    }


def case_margin(row: dict[str, Any]) -> float:
    if row.get("expected_prefer_margin") is not None:
        return float(row["expected_prefer_margin"])
    return top1_top2_margin(row)


def top1_top2_margin(row: dict[str, Any]) -> float:
    top_results = row.get("top_results", [])
    if len(top_results) < 2:
        return 0.0
    return float(top_results[0]["score"]) - float(top_results[1]["score"])


def confidence_bucket(margin: float) -> str:
    if margin < 0.02:
        return "low"
    if margin < 0.05:
        return "medium"
    return "high"


def select_worst_cases(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=case_margin)
    worst = []
    for row in sorted_rows[:limit]:
        worst.append(
            {
                "case_id": row["case_id"],
                "case_type": row["case_type"],
                "margin": round(case_margin(row), 6),
                "confidence": confidence_bucket(case_margin(row)),
                "target_rank": row.get("target_rank"),
                "expected_prefer_rank": row.get("expected_prefer_rank"),
                "top_results": [
                    {
                        "item_id": result["item_id"],
                        "score": result["score"],
                        "script_stage": result.get("metadata", {}).get("script_stage"),
                    }
                    for result in row.get("top_results", [])[:3]
                ],
            }
        )
    return worst


def run_llm_sample_judge(
    *,
    ranked_by_key: dict[str, list[dict[str, Any]]],
    sample_size: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    candidates = select_llm_sample_candidates(ranked_by_key, sample_size=sample_size)
    results = []
    errors = []
    for row in candidates:
        try:
            results.append(
                judge_top_results_with_llm(
                    row,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
            )
        except Exception as exc:
            errors.append({"case_id": row["case_id"], "error": str(exc)})
    scored_items = [
        item
        for result in results
        for item in result.get("judgements", [])
        if isinstance(item.get("score"), (int, float))
    ]
    relevant = [item for item in scored_items if float(item["score"]) >= 2.0]
    return {
        "sample_size": sample_size,
        "attempted": len(candidates),
        "judged": len(results),
        "precision_at_3": round(len(relevant) / len(scored_items), 6) if scored_items else 0.0,
        "average_score": round(mean([float(item["score"]) for item in scored_items]), 6) if scored_items else 0.0,
        "results": results,
        "llm_errors": errors,
    }


def select_llm_sample_candidates(
    ranked_by_key: dict[str, list[dict[str, Any]]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    final_rows = ranked_by_key["final_score"]
    embedding_rows = {row["case_id"]: row for row in ranked_by_key["embedding_only"]}

    def priority(row: dict[str, Any]) -> tuple[int, float, str]:
        low_confidence = confidence_bucket(case_margin(row)) == "low"
        embedding_row = embedding_rows.get(row["case_id"], {})
        final_top = row.get("top_results", [{}])[0].get("item_id") if row.get("top_results") else None
        embedding_top = (
            embedding_row.get("top_results", [{}])[0].get("item_id")
            if embedding_row.get("top_results")
            else None
        )
        changed = final_top != embedding_top
        hard_negative = row["case_type"] == "hard_negative"
        return (
            0 if low_confidence else 1 if changed else 2 if hard_negative else 3,
            case_margin(row),
            row["case_id"],
        )

    return sorted(final_rows, key=priority)[:sample_size]


def judge_top_results_with_llm(
    row: dict[str, Any],
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    from sceneweaver.llm.client import VisionLLMClient

    payload = {
        "task": "你是企业宣传片导演，请判断每个检索结果是否适合用户需求。重点看脚本阶段、创作目的、误用风险。只输出 JSON。",
        "score_rule": {
            "0": "完全不适合",
            "1": "画面相似但脚本用途不对",
            "2": "基本适合",
            "3": "非常适合",
        },
        "output_schema": {
            "judgements": [
                {
                    "item_id": "string",
                    "score": "0..3 integer",
                    "reason": "short Chinese sentence",
                }
            ]
        },
        "user_input": row["user_input"],
        "query_constraints": row.get("query_constraints", {}),
        "results": [
            {
                "item_id": result["item_id"],
                "script_stage": result.get("metadata", {}).get("script_stage"),
                "creative_purpose": result.get("metadata", {}).get("creative_purpose"),
                "script_use_sentence": result.get("metadata", {}).get("script_use_sentence"),
                "score": result.get("score"),
            }
            for result in row.get("top_results", [])[:3]
        ],
    }
    result = VisionLLMClient().analyze_text_json(
        system_prompt="你是严格的检索质量评审。只输出 JSON。",
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=600,
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    raw_judgements = result.get("judgements", [])
    if not isinstance(raw_judgements, list):
        raise ValueError("LLM response must contain a judgements list")
    judgements = []
    top_item_ids = {item["item_id"] for item in row.get("top_results", [])[:3]}
    for item in raw_judgements:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id", ""))
        if item_id not in top_item_ids:
            continue
        judgements.append(
            {
                "item_id": item_id,
                "score": int(item.get("score", 0)),
                "reason": str(item.get("reason", "")),
            }
        )
    return {
        "case_id": row["case_id"],
        "case_type": row["case_type"],
        "margin": round(case_margin(row), 6),
        "judgements": judgements,
    }


def build_paraphrase_variants(case: dict[str, Any]) -> list[dict[str, Any]]:
    target = case["target"]
    stage = stage_word(target.get("script_stage", ""))
    purposes = purpose_text(target.get("creative_purpose", []))
    tags = tag_hint(case)
    summary = case.get("target_summary", "")
    industry = target.get("industry", "").replace("_", " ")
    variants = [
        (
            "explicit",
            f"我要一个{stage}段落，核心就是{summary}",
        ),
        (
            "fuzzy",
            f"想让这一段先把观众带进情绪和问题里，不急着讲产品，重点是{purposes}。",
        ),
        (
            "style",
            f"别太像广告，也不要宣传片腔，要更像纪录片观察，但仍然服务于{stage}和{purposes}。",
        ),
        (
            "negative",
            f"不要炫技，不要互联网大厂味，也不要只摆概念；我需要它完成{stage}，重点是{purposes}。",
        ),
        (
            "constraint_first",
            f"先排除那种产品说明和技术炫耀，我真正要的是{stage}，让观众感到{purposes}。",
        ),
        (
            "human_value",
            f"这段要有人味，不要只展示系统；如果出现技术，也只是帮助观众理解{purposes}。",
        ),
        (
            "director_brief",
            f"按导演阐述来讲，这里需要承担{stage}的叙事功能，画面要克制，服务于{purposes}。",
        ),
        (
            "mixed",
            mixed_paraphrase(stage, purposes, tags),
        ),
    ]
    return [
        {
            "case_id": f"paraphrase__{variant_type}__{case['case_id']}",
            "source_case_id": case["case_id"],
            "variant_type": variant_type,
            "user_input": text,
            "target": target,
            "target_summary": summary,
        }
        for variant_type, text in variants
    ]


def build_style_negative_case(case: dict[str, Any]) -> dict[str, Any]:
    target = case["target"]
    stage = stage_word(target.get("script_stage", ""))
    purposes = purpose_text(target.get("creative_purpose", []))
    return {
        **case,
        "case_id": f"{case['case_id']}__style_negative",
        "case_type": "style_negative",
        "difficulty": "hard",
        "expected_relation": "should_match",
        "variant_type": "style_negative",
        "source_case_id": case["case_id"],
        "user_input": (
            f"需要一个{stage}段落，重点是{purposes}，要有人味、像纪录片、真实现场。"
            "不要大厂味，不要广告感，不要炫技，也不要技术炫耀。"
            f"参考语义：{case.get('target_summary', '')}"
        ),
    }


def mixed_paraphrase(stage: str, purposes: str, tags: str) -> str:
    if "技术展示" in stage or "技术入场" in stage:
        return f"真正要的是{stage}：{purposes}。画面可以借用{tags}，但不要做成冷冰冰的功能说明。"
    return f"真正要的是{stage}：{purposes}。画面可以借用{tags}，但这些只是辅助，不要让它变成技术展示。"


def stage_word(stage: str) -> str:
    return STAGE_WORDS.get(stage, stage.replace("_", " ") or "某个段落")


def purpose_text(purposes: list[str]) -> str:
    values = [purpose_word(purpose) for purpose in purposes[:3]]
    return "、".join(values) if values else "明确的创作目的"


def purpose_word(purpose: str) -> str:
    overrides = {
        "opening": "开场建立",
        "setup": "需求铺垫",
        "character_intro": "人物出场",
        "team_work": "团队协作",
        "technology_showcase": "技术能力展示",
        "value_expression": "价值表达",
        "scale_reveal": "规模揭示",
        "outcome": "结果落地",
        "ending": "结尾收束",
        "build_empathy": "建立共情",
        "show_responsibility": "表现责任感",
        "humanize_technology": "让技术更有人味",
    }
    return PURPOSE_WORDS.get(purpose) or STAGE_WORDS.get(purpose) or overrides.get(purpose) or purpose.replace("_", " ")


def tag_hint(case: dict[str, Any]) -> str:
    tags = str(case.get("target_tags_text", "")).split()
    if tags:
        return "、".join(tags[:3])
    return "人物、场景、动作、屏幕"


def purpose_hit_at(results: list[dict[str, Any]], target_purposes: set[str], k: int) -> bool:
    if not target_purposes:
        return False
    for result in results[:k]:
        purposes = set(result.get("metadata", {}).get("creative_purpose", []))
        if target_purposes & purposes:
            return True
    return False


def summarize_paraphrase_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "target_recall_at_1": 0.0,
            "target_recall_at_3": 0.0,
            "target_recall_at_10": 0.0,
            "stage_hit_at_1": 0.0,
            "stage_hit_at_3": 0.0,
            "purpose_hit_at_3": 0.0,
            "low_confidence_rate": 0.0,
            "mean_top1_top2_margin": 0.0,
        }
    return {
        "count": len(rows),
        "target_recall_at_1": round(sum(1 for row in rows if row["target_rank"] == 1) / len(rows), 6),
        "target_recall_at_3": round(sum(1 for row in rows if row["target_rank"] and row["target_rank"] <= 3) / len(rows), 6),
        "target_recall_at_10": round(sum(1 for row in rows if row["target_rank"] and row["target_rank"] <= 10) / len(rows), 6),
        "stage_hit_at_1": round(sum(1 for row in rows if row["stage_hit_at_1"]) / len(rows), 6),
        "stage_hit_at_3": round(sum(1 for row in rows if row["stage_hit_at_3"]) / len(rows), 6),
        "purpose_hit_at_3": round(sum(1 for row in rows if row["purpose_hit_at_3"]) / len(rows), 6),
        "low_confidence_rate": round(sum(1 for row in rows if row["confidence"] == "low") / len(rows), 6),
        "mean_top1_top2_margin": round(mean([float(row["top1_top2_margin"]) for row in rows]), 6),
    }


def style_negative_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "target_recall_at_1": 0.0,
            "target_recall_at_3": 0.0,
            "target_recall_at_10": 0.0,
            "style_negative_violation_at_1": 0.0,
            "style_negative_violation_at_3": 0.0,
            "style_negative_violation_at_10": 0.0,
            "low_confidence_rate": 0.0,
        }
    return {
        "count": len(rows),
        "target_recall_at_1": recall_at(rows, 1),
        "target_recall_at_3": recall_at(rows, 3),
        "target_recall_at_10": recall_at(rows, 10),
        "style_negative_violation_at_1": style_violation_at(rows, 1),
        "style_negative_violation_at_3": style_violation_at(rows, 3),
        "style_negative_violation_at_10": style_violation_at(rows, 10),
        "low_confidence_rate": round(sum(1 for row in rows if confidence_bucket(case_margin(row)) == "low") / len(rows), 6),
    }


def style_violation_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    violations = 0
    for row in rows:
        if any(result.get("constraint_hits", {}).get("negative_style") for result in row.get("top_results", [])[:k]):
            violations += 1
    return round(violations / len(rows), 6)


def summarize_paraphrase_by_source_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = sorted({row["source_case_id"] for row in rows})
    return [
        {
            "source_case_id": source_id,
            **summarize_paraphrase_rows([row for row in rows if row["source_case_id"] == source_id]),
        }
        for source_id in source_ids
    ]


def rerank_with_constraints(
    embedding_ranked: list[dict[str, Any]],
    query_constraints: dict[str, Any],
    constraint_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    reranked = []
    for row in embedding_ranked:
        constraint_score, constraint_hits = score_constraints(
            query_constraints,
            row.get("metadata", {}),
            constraint_profile,
        )
        final_score = float(row["embedding_score"]) + constraint_score
        if has_forbidden_stage_hit(query_constraints, row.get("metadata", {})):
            final_score -= HARD_FORBIDDEN_STAGE_VETO
        reranked.append(
            {
                **row,
                "score": round(final_score, 6),
                "final_score": round(final_score, 6),
                "constraint_score": round(constraint_score, 6),
                "constraint_hits": constraint_hits,
            }
        )
    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked


def possible_overfit(
    dev_baseline_metrics: dict[str, Any],
    dev_tuned_metrics: dict[str, Any],
    test_baseline_metrics: dict[str, Any],
    test_tuned_metrics: dict[str, Any],
) -> bool:
    return (
        tuning_selection_key(dev_tuned_metrics) > tuning_selection_key(dev_baseline_metrics)
        and tuning_selection_key(test_tuned_metrics) < tuning_selection_key(test_baseline_metrics)
    )


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
    hard_negative_rates = [
        metric_value(row["test_metrics"], "hard_negative", "hard_negative_expected_prefer_margin_positive_rate")
        for row in fixture_reports
    ]
    simple_recall_at_3 = [
        metric_value(row["test_metrics"], "simple_positive", "recall_at_3")
        for row in fixture_reports
    ]
    hard_positive_recall_at_10 = [
        metric_value(row["test_metrics"], "hard_positive", "recall_at_10")
        for row in fixture_reports
    ]
    worst_index = min(range(len(fixture_reports)), key=lambda index: hard_negative_rates[index])
    return {
        "fixture_count": len(fixture_reports),
        "mean_hard_negative_margin_positive_rate": round(mean(hard_negative_rates), 6),
        "min_hard_negative_margin_positive_rate": round(min(hard_negative_rates), 6),
        "mean_simple_positive_recall_at_3": round(mean(simple_recall_at_3), 6),
        "mean_hard_positive_recall_at_10": round(mean(hard_positive_recall_at_10), 6),
        "worst_fixture_id": fixture_reports[worst_index]["fixture_id"],
        "possible_overfit": min(hard_negative_rates) < 0.5,
    }


def metric_value(metrics: dict[str, Any], case_type: str, metric_name: str) -> float:
    return float(metrics.get("by_case_type", {}).get(case_type, {}).get(metric_name, 0.0))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def search_index(
    index: dict[str, Any],
    query_channels: list[dict[str, Any]],
    cache: EmbeddingCache,
    *,
    top_k: int,
    user_input: str = "",
    constraint_profile: dict[str, Any] | None = None,
    constraints_enabled: bool = True,
) -> list[dict[str, Any]]:
    query_vectors = {
        channel["channel"]: cache.require_embedding(channel["text"])
        for channel in query_channels
        if channel.get("enabled", True)
    }
    query_constraints = (
        parse_query_constraints(user_input, constraint_profile)
        if constraints_enabled and user_input
        else {"desired_stage": [], "forbidden_stage": [], "negative_constraints": [], "visual_hints": []}
    )
    scored = []
    for item in index["items"]:
        embedding_score, channel_scores = score_item(item, query_channels, query_vectors)
        constraint_score, constraint_hits = (
            score_constraints(query_constraints, item.get("metadata", {}), constraint_profile)
            if constraints_enabled
            else (0.0, {})
        )
        final_score = embedding_score + constraint_score
        if constraints_enabled and has_forbidden_stage_hit(query_constraints, item.get("metadata", {})):
            final_score -= HARD_FORBIDDEN_STAGE_VETO
        scored.append(
            {
                "item_id": item["item_id"],
                "score": round(final_score, 6),
                "final_score": round(final_score, 6),
                "embedding_score": round(embedding_score, 6),
                "constraint_score": round(constraint_score, 6),
                "constraint_hits": constraint_hits,
                "channel_scores": channel_scores,
                "metadata": item["metadata"],
                "lexical_text": index_item_lexical_text(item),
            }
        )
    scored.sort(key=lambda row: row["final_score"], reverse=True)
    return scored[:top_k]


def score_item(
    item: dict[str, Any],
    query_channels: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    item_channels = {channel["channel"]: channel for channel in item["channels"]}
    total = 0.0
    channel_scores: dict[str, float] = {}
    for query_channel in query_channels:
        if not query_channel.get("enabled", True):
            continue
        target_channel = target_channel_for_query(query_channel["channel"])
        item_channel = item_channels.get(target_channel)
        query_vector = query_vectors.get(query_channel["channel"])
        if item_channel is None or query_vector is None:
            continue
        weight = float(query_channel.get("weight", DEFAULT_CHANNEL_WEIGHTS.get(target_channel, 0.0)))
        similarity = cosine(query_vector, item_channel["embedding"])
        contribution = weight * similarity
        channel_scores[target_channel] = round(contribution, 6)
        total += contribution
    return total, channel_scores


def index_item_lexical_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    channel_text = " ".join(
        channel.get("text", "")
        for channel in item.get("channels", [])
        if isinstance(channel, dict)
    )
    parts = [
        metadata.get("script_stage", ""),
        " ".join(metadata.get("creative_purpose", [])),
        metadata.get("script_use_sentence", ""),
        metadata.get("industry", ""),
        metadata.get("style", ""),
        channel_text,
    ]
    return " ".join(str(part) for part in parts if part)


def build_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in case_results:
        by_type.setdefault(row["case_type"], []).append(row)
    return {
        "overall": metric_group(case_results),
        "by_case_type": {case_type: metric_group(rows) for case_type, rows in by_type.items()},
    }


def metric_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if row["expected_relation"] == "should_match"]
    negative_rows = [row for row in rows if row["expected_relation"] == "should_not_match"]
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


def recall_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row["target_rank"] is not None and row["target_rank"] <= k)
    return round(hits / len(rows), 6)


def mean_reciprocal_rank(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = sum(1 / row["target_rank"] for row in rows if row["target_rank"])
    return round(total / len(rows), 6)


def margin_positive_rate(rows: list[dict[str, Any]]) -> float:
    margins = [row["expected_prefer_margin"] for row in rows if row["expected_prefer_margin"] is not None]
    if not margins:
        return 0.0
    return round(sum(1 for margin in margins if margin > 0) / len(margins), 6)


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
    return canonical_stage(result.get("metadata", {}).get("script_stage", ""))


def tuning_selection_key(metrics: dict[str, Any]) -> list[float]:
    by_type = metrics.get("by_case_type", {})
    hard_negative = by_type.get("hard_negative", {})
    hard_positive = by_type.get("hard_positive", {})
    simple_positive = by_type.get("simple_positive", {})
    return [
        float(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", 0.0)),
        float(hard_positive.get("recall_at_10", 0.0)),
        float(simple_positive.get("recall_at_3", 0.0)),
    ]


def select_best_tuning_result(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No tuning candidates were produced.")
    return max(candidates, key=lambda row: row["selection_score"])


def rank_of(rows: list[dict[str, Any]], item_id: str | None) -> int | None:
    if item_id is None:
        return None
    for index, row in enumerate(rows, start=1):
        if row["item_id"] == item_id:
            return index
    return None


def score_of(rows: list[dict[str, Any]], item_id: str | None) -> float | None:
    if item_id is None:
        return None
    for row in rows:
        if row["item_id"] == item_id:
            return row["score"]
    return None


def cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def read_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_embedding_setup_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "dashscope package is required" in message
        or "DASHSCOPE_API_KEY" in message
        or "embedding missing for text key=" in message
    )


def embedding_setup_error_payload(exc: RuntimeError) -> dict[str, Any]:
    return {
        "error": "embedding_setup_unavailable",
        "message": str(exc),
        "hint": (
            "Activate the environment with dashscope and API credentials before uncached embedding runs, "
            "for example: conda activate video_expert_analyzer and set DASHSCOPE_API_KEY."
        ),
    }


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if not is_embedding_setup_error(exc):
            raise
        print(json.dumps(embedding_setup_error_payload(exc), ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2) from None
