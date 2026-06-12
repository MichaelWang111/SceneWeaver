from __future__ import annotations

from collections import Counter
import math
from typing import Any


def graded_metrics(rows: list[dict[str, Any]], qrels: list[dict[str, Any]], *, top_k: int) -> dict[str, float]:
    qrel_map = {(row["query_id"], row["item_id"]): int(row["grade"]) for row in qrels}
    by_query: dict[str, list[int]] = {}
    for qrel in qrels:
        by_query.setdefault(qrel["query_id"], []).append(int(qrel["grade"]))
    return {
        "nDCG@3": round(mean([ndcg_at(row, qrel_map, by_query, 3) for row in rows]), 6),
        "nDCG@10": round(mean([ndcg_at(row, qrel_map, by_query, min(10, top_k)) for row in rows]), 6),
        "ERR@10": round(mean([err_at(row, qrel_map, min(10, top_k)) for row in rows]), 6),
        "MRR@10": round(mean([mrr_at(row, qrel_map, min(10, top_k)) for row in rows]), 6),
        "Judged@10": round(mean([judged_at(row, qrel_map, min(10, top_k)) for row in rows]), 6),
        "Unjudged@10": round(mean([unjudged_at(row, qrel_map, min(10, top_k)) for row in rows]), 6),
        "Recall@10": round(mean([graded_recall_at(row, qrel_map, by_query, min(10, top_k)) for row in rows]), 6),
        "case_count": len(rows),
        "qrels_count": len(qrels),
    }


def ndcg_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], by_query: dict[str, list[int]], k: int) -> float:
    query_id = row["case_id"]
    grades = [qrels.get((query_id, result["item_id"]), 0) for result in row.get("top_results", [])[:k]]
    dcg = discounted_gain(grades)
    ideal = discounted_gain(sorted(by_query.get(query_id, []), reverse=True)[:k])
    return 0.0 if ideal == 0 else dcg / ideal


def discounted_gain(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def err_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int, *, max_grade: int = 3) -> float:
    query_id = row["case_id"]
    carry = 1.0
    total = 0.0
    max_gain = float(2**max_grade)
    for rank, result in enumerate(row.get("top_results", [])[:k], start=1):
        grade = qrels.get((query_id, result["item_id"]), 0)
        relevance_probability = (2**grade - 1) / max_gain
        total += carry * relevance_probability / rank
        carry *= 1 - relevance_probability
    return total


def mrr_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int) -> float:
    query_id = row["case_id"]
    for rank, result in enumerate(row.get("top_results", [])[:k], start=1):
        if qrels.get((query_id, result["item_id"]), 0) > 0:
            return 1 / rank
    return 0.0


def judged_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int) -> float:
    query_id = row["case_id"]
    results = row.get("top_results", [])[:k]
    if not results:
        return 0.0
    return sum(1 for result in results if (query_id, result["item_id"]) in qrels) / len(results)


def unjudged_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int) -> float:
    results = row.get("top_results", [])[:k]
    if not results:
        return 0.0
    return 1.0 - judged_at(row, qrels, k)


def graded_recall_at(
    row: dict[str, Any],
    qrels: dict[tuple[str, str], int],
    by_query: dict[str, list[int]],
    k: int,
) -> float:
    query_id = row["case_id"]
    relevant_total = sum(1 for grade in by_query.get(query_id, []) if grade >= 2)
    if relevant_total == 0:
        return 0.0
    retrieved = sum(1 for result in row.get("top_results", [])[:k] if qrels.get((query_id, result["item_id"]), 0) >= 2)
    return retrieved / relevant_total


def qrels_judged_coverage(rows: list[dict[str, Any]], qrels: list[dict[str, Any]], *, top_k: int) -> float:
    qrel_map = {(row["query_id"], row["item_id"]): int(row["grade"]) for row in qrels}
    return mean([judged_at(row, qrel_map, top_k) for row in rows])


def recall_bound_rows(
    rows_by_key: dict[str, list[dict[str, Any]]],
    *,
    baseline_key: str,
    candidate_depth: int,
    top_k: int,
) -> list[dict[str, Any]]:
    baseline_rows = rows_by_key.get(baseline_key, [])
    rows_by_case = {key: {row["case_id"]: row for row in rows} for key, rows in rows_by_key.items()}
    result = []
    for baseline_row in baseline_rows:
        case_id = baseline_row["case_id"]
        ranks = {key: rows.get(case_id, {}).get("target_rank") for key, rows in rows_by_case.items()}
        known_ranks = [rank for rank in ranks.values() if rank is not None]
        oracle_best_rank = min(known_ranks) if known_ranks else None
        candidate_hit = bool(oracle_best_rank is not None and oracle_best_rank <= candidate_depth)
        baseline_rank = ranks.get(baseline_key)
        result.append(
            {
                "case_id": case_id,
                "user_input": baseline_row.get("user_input", ""),
                "target_item_id": baseline_row.get("target_item_id"),
                "baseline_rank": baseline_rank,
                "oracle_best_rank": oracle_best_rank,
                "candidate_depth_hit": candidate_hit,
                "rank_by_workflow": ranks,
                "failure_type": recall_bound_failure_type(
                    baseline_rank=baseline_rank,
                    oracle_best_rank=oracle_best_rank,
                    candidate_depth=candidate_depth,
                    top_k=top_k,
                ),
                "top1_by_baseline": baseline_row.get("top_results", [{}])[0].get("item_id")
                if baseline_row.get("top_results")
                else None,
            }
        )
    return result


def recall_bound_failure_type(
    *,
    baseline_rank: int | None,
    oracle_best_rank: int | None,
    candidate_depth: int,
    top_k: int,
) -> str:
    if baseline_rank is not None and baseline_rank <= top_k:
        return "success"
    if oracle_best_rank is None or oracle_best_rank > candidate_depth:
        return "candidate_recall_failure"
    if oracle_best_rank <= top_k:
        return "workflow_selection_failure"
    return "fusion_ranking_failure"


def recall_bound_summary(rows: list[dict[str, Any]], *, top_k: int, candidate_depth: int) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        f"baseline_recall_at_{top_k}": round(
            sum(1 for row in rows if row.get("baseline_rank") is not None and row["baseline_rank"] <= top_k)
            / max(1, len(rows)),
            6,
        ),
        f"oracle_recall_at_{top_k}": round(
            sum(1 for row in rows if row.get("oracle_best_rank") is not None and row["oracle_best_rank"] <= top_k)
            / max(1, len(rows)),
            6,
        ),
        f"oracle_recall_at_{candidate_depth}": round(
            sum(1 for row in rows if row.get("candidate_depth_hit")) / max(1, len(rows)),
            6,
        ),
        "failure_type_counts": dict(sorted(Counter(row["failure_type"] for row in rows).items())),
        "mean_oracle_best_rank": round(
            mean([float(row["oracle_best_rank"]) for row in rows if row.get("oracle_best_rank") is not None]),
            6,
        ),
    }


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


__all__ = [
    "discounted_gain",
    "err_at",
    "graded_metrics",
    "graded_recall_at",
    "judged_at",
    "mrr_at",
    "ndcg_at",
    "qrels_judged_coverage",
    "recall_bound_rows",
    "recall_bound_summary",
    "unjudged_at",
]
