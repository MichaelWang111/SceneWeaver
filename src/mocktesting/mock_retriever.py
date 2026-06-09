from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_embedding_index.json"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_retrieval_report.json"
DEFAULT_SEARCH_OUTPUT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_search_result.json"
DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "eval_inputs" / "review_generated_inputs.json"
DEFAULT_TUNING_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_constraint_tuning_report.json"

TUNING_GRID = {
    "desired_stage_bonus": [0.06, 0.10, 0.12, 0.16],
    "forbidden_stage_penalty": [0.10, 0.15, 0.18, 0.22, 0.28],
    "negative_constraint_penalty": [0.00, 0.05, 0.08, 0.12],
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
    eval_parser.add_argument("--limit", type=int, default=450)
    eval_parser.add_argument("--top-k", type=int, default=10)
    eval_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)

    tune_parser = subparsers.add_parser("tune-constraints")
    add_common_paths(tune_parser)
    tune_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    tune_parser.add_argument("--limit", type=int, default=60)
    tune_parser.add_argument("--top-k", type=int, default=10)
    tune_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    tune_parser.add_argument("--output", type=Path, default=DEFAULT_TUNING_REPORT_PATH)

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
    cases = read_cases(args.inputs, args.limit)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)

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
                    top_k=args.top_k,
                    constraint_profile=profile,
                    include_cases=False,
                )
                candidates.append(
                    {
                        "weights": profile["weights"],
                        "metrics": report["metrics"],
                        "selection_score": tuning_selection_key(report["metrics"]),
                    }
                )

    best = select_best_tuning_result(candidates)
    best_profile = profile_with_weights(
        desired_stage_bonus=best["weights"]["desired_stage_bonus"],
        forbidden_stage_penalty=best["weights"]["forbidden_stage_penalty"],
        negative_constraint_penalty=best["weights"]["negative_constraint_penalty"],
        base_profile=base_profile,
    )
    write_constraint_profile(args.constraint_profile, best_profile)
    return {
        "method": "mock_constraint_tuning",
        "limit": args.limit,
        "top_k": args.top_k,
        "candidate_count": len(candidates),
        "profile_output": str(args.constraint_profile),
        "best": {
            "weights": best["weights"],
            "selection_score": best["selection_score"],
            "metrics": best["metrics"],
        },
        "candidates": candidates,
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
    cases = read_cases(inputs_path, limit)
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


def read_cases(path: Path, limit: int) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    if limit > 0:
        return cases[:limit]
    return cases


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


def evaluate_precomputed_cases(
    *,
    precomputed: list[dict[str, Any]],
    top_k: int,
    constraint_profile: dict[str, Any],
    include_cases: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    case_results = []
    for row in precomputed:
        case = row["case"]
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = rerank_with_constraints(row["embedding_ranked"], query_constraints, constraint_profile)
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
        "constraints_enabled": True,
        "elapsed_seconds": elapsed,
        "seconds_per_case": round(elapsed / max(1, len(case_results)), 6),
        "metrics": build_metrics(case_results),
    }
    if include_cases:
        result["cases"] = case_results
    return result


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


if __name__ == "__main__":
    main()
