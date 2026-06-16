from __future__ import annotations

from pathlib import Path
from collections import Counter
import hashlib
import math
import time
from typing import Any

import numpy as np

from retrieval_lab.artifacts import data_sha256, read_json, write_json
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retrieval_lab.experiments.runs import cases_from_run_rows, run_artifact_summary
from retrieval_lab.indexes import index_items_from_cases, lexical_tokens, target_item_id
from retrieval_lab.planners import DEFAULT_PLANNER_CACHE_PATH, plan_many
from retrieval_lab.ranking.workflow import WORKFLOW_RANKING_KEYS, rerank_row_by_workflow
from retrieval_lab.ranking.workflow import style_guardrail_action, style_risk_evidence, style_risk_score


DEFAULT_RETRIEVAL_RUN_OUTPUT = Path(".tmp") / "retrieval_lab" / "retrieval_run_latest.json"
DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT = Path(".tmp") / "retrieval_lab" / "retrieval_legacy_comparison_latest.json"
HARD_FORBIDDEN_STAGE_VETO = 1000.0
RRF_K = 60


def retrieval_run(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    split: str = "test",
    limit: int = 0,
    planner: str = "multi_query",
    planner_cache: Path | None = DEFAULT_PLANNER_CACHE_PATH,
    top_k: int = 10,
    candidate_depth: int = 100,
    run_name: str = "",
    ranking_key: str = "hybrid_rrf_constraints_signature",
) -> dict[str, Any]:
    cases = read_cases(dataset_path, split=split, limit=limit)
    return retrieval_run_from_cases(
        cases,
        dataset=str(dataset_path),
        split=split,
        limit=limit,
        planner=planner,
        planner_cache=planner_cache,
        top_k=top_k,
        candidate_depth=candidate_depth,
        run_name=run_name,
        ranking_key=ranking_key,
    )


def retrieval_run_from_cases(
    cases: list[dict[str, Any]],
    *,
    dataset: str,
    split: str = "test",
    limit: int = 0,
    planner: str = "multi_query",
    planner_cache: Path | None = DEFAULT_PLANNER_CACHE_PATH,
    top_k: int = 10,
    candidate_depth: int = 100,
    run_name: str = "",
    ranking_key: str = "hybrid_rrf_constraints_signature",
    planner_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    items = index_items_from_cases(cases)
    prepared_index = prepare_retrieval_index(items)
    plans_report = plan_many(
        [str(case.get("user_input", "")) for case in cases],
        planner=planner,
        cache_path=planner_cache,
        config={"command": "retrieval_run", "split": split, "limit": limit, **(planner_config or {})},
    )
    rows = []
    for case, plan in zip(cases, plans_report["plans"], strict=False):
        rows.append(
            retrieve_case(
                case,
                plan=plan,
                items=items,
                prepared_index=prepared_index,
                top_k=top_k,
                candidate_depth=candidate_depth,
                ranking_key=ranking_key,
            )
        )
    run_rows = {run_name or f"{planner}::{ranking_key}": rows}
    summary = {
        **run_artifact_summary(run_rows, cases_from_run_rows(run_rows)),
        "dataset": dataset,
        "split": split,
        "limit": limit,
        "planner": planner,
        "top_k": top_k,
        "candidate_depth": candidate_depth,
        "ranking_key": ranking_key,
        "index_item_count": len(items),
        "retrieval_runtime": "native_in_memory_bm25_hash_dense_rrf",
        "dense_source": "local_hash_vector_fallback",
        "planner_negative_leak_rate": plans_report["summary"].get("negative_leak_rate", 0.0),
        "target_recall_at_10": target_recall(rows, 10),
        "stage_hit_at_3": stage_hit(rows, 3),
        "purpose_hit_at_3": purpose_hit(rows, 3),
        "style_violation_at_3": style_violation(rows, 3),
        "mean_top1_top2_margin": mean_top_margin(rows),
        "low_confidence_rate": low_confidence_rate(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    artifact = {
        "method": "retrieval_lab_native_retrieval_run",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_config": {
            "workflow": "native_retrieval_runtime",
            "ranking_key": ranking_key,
            "query_planner": planner,
            "top_k": top_k,
            "candidate_depth": candidate_depth,
            "constraints_enabled": True,
            "llm_enabled": False,
            "parameters": {"split": split, "limit": limit},
        },
        "planner_summary": plans_report["summary"],
        "run_rows": run_rows,
        "cases": cases_from_run_rows(run_rows),
        "summary": summary,
        "fingerprint": data_sha256({"run_rows": run_rows, "summary": summary}),
    }
    return artifact


def retrieve_case(
    case: dict[str, Any],
    *,
    plan: dict[str, Any],
    items: list[dict[str, Any]],
    prepared_index: dict[str, Any] | None = None,
    top_k: int,
    candidate_depth: int,
    ranking_key: str,
) -> dict[str, Any]:
    scored = score_items(items, plan=plan, prepared_index=prepared_index)
    ranked = sorted(scored, key=lambda row: (-float(row["final_score"]), row["item_id"]))
    all_results = ranked[:candidate_depth]
    top_results = ranked[:top_k]
    expected = expected_target(case)
    expected_id = target_item_id(expected) if expected else ""
    target_rank = next((index for index, item in enumerate(ranked, start=1) if item["item_id"] == expected_id), None)
    row = {
        "case_id": case.get("case_id", ""),
        "variant_type": case.get("case_type", "default"),
        "fuzzy_set_type": case.get("fuzzy_set_type", ""),
        "user_input": case.get("user_input", ""),
        "query_plan": plan,
        "query_constraints": {
            "desired_stage": plan.get("desired_stage", []),
            "forbidden_stage": plan.get("forbidden_stage", []),
            "negative_style": plan.get("negative_style", []),
        },
        "target_item_id": expected_id,
        "target_stage": expected.get("script_stage", "") if expected else "",
        "target_purposes": expected.get("creative_purpose", []) if expected else [],
        "target_rank": target_rank,
        "planner_confidence": plan.get("confidence", 0.0),
        "ranking_key": ranking_key,
        "top_results": top_results,
        "all_results": all_results,
    }
    if ranking_key in WORKFLOW_RANKING_KEYS:
        return rerank_row_by_workflow(row, ranking_key=ranking_key, top_k=top_k)
    return row


def retrieve_case_compact_fast(
    case: dict[str, Any],
    *,
    plan: dict[str, Any],
    items: list[dict[str, Any]],
    prepared_index: dict[str, Any],
    top_k: int,
    ranking_key: str,
) -> dict[str, Any]:
    signals = fast_score_arrays(items, plan=plan, prepared_index=prepared_index)
    final_scores = signals["final_score"]
    item_ids = prepared_index.get("item_ids", [item["item_id"] for item in items])
    top_indices = top_indices_for_scores(final_scores, item_ids, top_k)
    top_results = [compact_result_for_index(items[index], signals, index) for index in top_indices]
    expected = expected_target(case)
    expected_id = target_item_id(expected) if expected else ""
    target_rank = rank_for_item(expected_id, final_scores, item_ids)
    return {
        "case_id": case.get("case_id", ""),
        "variant_type": case.get("case_type", "default"),
        "fuzzy_set_type": case.get("fuzzy_set_type", ""),
        "user_input": case.get("user_input", ""),
        "query_plan": plan,
        "target_item_id": expected_id,
        "target_stage": expected.get("script_stage", "") if expected else "",
        "target_purposes": expected.get("creative_purpose", []) if expected else [],
        "target_rank": target_rank,
        "planner_confidence": plan.get("confidence", 0.0),
        "ranking_key": ranking_key,
        "top_results": top_results,
    }


def prepare_retrieval_index(items: list[dict[str, Any]]) -> dict[str, Any]:
    doc_freq: Counter[str] = Counter()
    doc_lengths = []
    vectors = {}
    token_sets = []
    visual_token_sets = []
    token_to_indices: dict[str, list[int]] = {}
    item_ids = []
    for index, item in enumerate(items):
        tokens = item.get("tokens", []) or lexical_tokens(item.get("text", ""))
        item_ids.append(item["item_id"])
        token_sets.append(set(tokens))
        doc_lengths.append(len(tokens))
        doc_freq.update(set(tokens))
        vectors[item["item_id"]] = hashed_vector(tokens)
        for token in set(tokens):
            token_to_indices.setdefault(token, []).append(index)
        metadata = item.get("metadata", {})
        visual_tokens = lexical_tokens(" ".join(str(value) for value in metadata.values()))
        visual_token_sets.append(set(visual_tokens))
    vector_matrix = np.asarray([vectors[item_id] for item_id in item_ids], dtype=np.float32) if item_ids else np.zeros((0, 64), dtype=np.float32)
    doc_lengths_np = np.asarray(doc_lengths, dtype=np.float32) if doc_lengths else np.zeros((0,), dtype=np.float32)
    return {
        "version": "prepared_retrieval_index_v2",
        "doc_count": len(items),
        "doc_freq": dict(doc_freq),
        "avg_doc_len": sum(doc_lengths) / max(1, len(doc_lengths)),
        "vectors": vectors,
        "vector_matrix": vector_matrix,
        "doc_lengths_np": doc_lengths_np,
        "token_sets": token_sets,
        "visual_token_sets": visual_token_sets,
        "token_to_indices": {token: np.asarray(indices, dtype=np.int32) for token, indices in token_to_indices.items()},
        "item_ids": item_ids,
    }


def score_items(
    items: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    prepared_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prepared = prepared_index or prepare_retrieval_index(items)
    if prepared.get("version") == "prepared_retrieval_index_v2":
        return fast_score_items(items, plan=plan, prepared_index=prepared)
    base_rows = [score_item(item, plan=plan, prepared_index=prepared) for item in items]
    return rows_with_rrf_scores(base_rows)


def rows_with_rrf_scores(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lexical_ranks = score_ranks(base_rows, "lexical_score")
    dense_ranks = score_ranks(base_rows, "embedding_score")
    signature_ranks = score_ranks(base_rows, "signature_score")
    for row in base_rows:
        item_id = row["item_id"]
        rrf = rrf_score([lexical_ranks.get(item_id), dense_ranks.get(item_id), signature_ranks.get(item_id)])
        row["rrf_score"] = round(rrf, 6)
        row["final_score"] = round(
            rrf * 10
            + float(row.get("constraint_score", 0.0))
            + float(row.get("purpose_score", 0.0))
            + float(row.get("signature_score", 0.0)) * 0.35
            + float(row.get("style_score", 0.0)),
            6,
        )
        row["score"] = row["final_score"]
        row["channel_scores"] = {
            "script_use": round(float(row.get("lexical_score", 0.0)), 6),
            "combined": round(float(row.get("embedding_score", 0.0)), 6),
            "experience": round(float(row.get("signature_score", 0.0)), 6),
            "visual_tags": round(float(row.get("visual_score", 0.0)), 6),
        }
    return base_rows


def fast_score_items(
    items: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    prepared_index: dict[str, Any],
) -> list[dict[str, Any]]:
    query_texts = [str(plan.get("positive_query", ""))]
    query_texts.extend(str(row.get("text", "")) for row in plan.get("rewrites", []) if isinstance(row, dict))
    if plan.get("hyde_text"):
        query_texts.append(str(plan["hyde_text"]))
    query_tokens = lexical_tokens(" ".join(query_texts))
    query_token_set = set(query_tokens)
    count = len(items)
    lexical_scores = bm25_scores_for_query(query_tokens, prepared_index, count)
    dense_scores = dense_scores_for_query(query_tokens, prepared_index, count)
    signature_tokens = set(lexical_tokens(signature_text(plan)))
    signature_scores = np.zeros(count, dtype=np.float32)
    visual_scores = np.zeros(count, dtype=np.float32)
    token_sets = prepared_index.get("token_sets", [])
    visual_token_sets = prepared_index.get("visual_token_sets", [])
    for index in range(count):
        item_tokens = token_sets[index] if index < len(token_sets) else set(items[index].get("tokens", []))
        signature_scores[index] = overlap_from_sets(signature_tokens, item_tokens)
        visual_tokens = visual_token_sets[index] if index < len(visual_token_sets) else item_tokens
        visual_scores[index] = overlap_from_sets(query_token_set, visual_tokens)

    base_rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        metadata = item.get("metadata", {})
        constraint, hits = constraint_score(metadata, plan)
        purpose = purpose_score(metadata, plan)
        style, style_hits = style_score(metadata, plan)
        hits.update(style_hits)
        lexical = float(lexical_scores[index])
        dense = float(dense_scores[index])
        signature = float(signature_scores[index])
        visual = float(visual_scores[index])
        final_score = lexical + dense + constraint + purpose + signature * 0.25 + style
        base_rows.append(
            result_with_style_diagnostics(
                {
                    "item_id": item["item_id"],
                    "score": round(final_score, 6),
                    "final_score": round(final_score, 6),
                    "lexical_score": round(lexical, 6),
                    "embedding_score": round(dense, 6),
                    "semantic_score": round(dense, 6),
                    "constraint_score": round(constraint, 6),
                    "purpose_score": round(purpose, 6),
                    "signature_score": round(signature, 6),
                    "style_score": round(style, 6),
                    "visual_score": round(visual, 6),
                    "metadata": metadata,
                    "constraint_hits": hits,
                    "explanation": explain_score(lexical, constraint, purpose, signature, dense, style, hits),
                }
            )
        )
    return rows_with_rrf_scores(base_rows)


def fast_score_arrays(
    items: list[dict[str, Any]],
    *,
    plan: dict[str, Any],
    prepared_index: dict[str, Any],
) -> dict[str, Any]:
    query_texts = [str(plan.get("positive_query", ""))]
    query_texts.extend(str(row.get("text", "")) for row in plan.get("rewrites", []) if isinstance(row, dict))
    if plan.get("hyde_text"):
        query_texts.append(str(plan["hyde_text"]))
    query_tokens = lexical_tokens(" ".join(query_texts))
    count = len(items)
    lexical_scores = bm25_scores_for_query(query_tokens, prepared_index, count)
    dense_scores = dense_scores_for_query(query_tokens, prepared_index, count)
    signature_tokens = set(lexical_tokens(signature_text(plan)))
    token_sets = prepared_index.get("token_sets", [])
    signature_scores = np.zeros(count, dtype=np.float32)
    constraint_scores = np.zeros(count, dtype=np.float32)
    purpose_scores = np.zeros(count, dtype=np.float32)
    style_scores = np.zeros(count, dtype=np.float32)
    hits_by_index: list[dict[str, list[str]]] = []
    for index, item in enumerate(items):
        metadata = item.get("metadata", {})
        item_tokens = token_sets[index] if index < len(token_sets) else set(item.get("tokens", []))
        signature_scores[index] = overlap_from_sets(signature_tokens, item_tokens)
        constraint, hits = constraint_score(metadata, plan)
        style, style_hits = style_score(metadata, plan)
        hits.update(style_hits)
        constraint_scores[index] = float(constraint)
        purpose_scores[index] = float(purpose_score(metadata, plan))
        style_scores[index] = float(style)
        hits_by_index.append(hits)
    lexical_ranks = rank_array(lexical_scores, prepared_index.get("item_ids", []))
    dense_ranks = rank_array(dense_scores, prepared_index.get("item_ids", []))
    signature_ranks = rank_array(signature_scores, prepared_index.get("item_ids", []))
    rrf_scores = (1.0 / (RRF_K + lexical_ranks)) + (1.0 / (RRF_K + dense_ranks)) + (1.0 / (RRF_K + signature_ranks))
    final_scores = (
        rrf_scores * 10
        + constraint_scores
        + purpose_scores
        + signature_scores * 0.35
        + style_scores
    )
    return {
        "lexical_score": lexical_scores,
        "embedding_score": dense_scores,
        "constraint_score": constraint_scores,
        "purpose_score": purpose_scores,
        "signature_score": signature_scores,
        "style_score": style_scores,
        "rrf_score": rrf_scores,
        "final_score": final_scores,
        "hits_by_index": hits_by_index,
    }


def rank_array(scores: np.ndarray, item_ids: list[str]) -> np.ndarray:
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), item_ids[index] if index < len(item_ids) else str(index)))
    ranks = np.empty(len(scores), dtype=np.float32)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def top_indices_for_scores(scores: np.ndarray, item_ids: list[str], top_k: int) -> list[int]:
    if len(scores) == 0:
        return []
    k = min(max(1, top_k), len(scores))
    if k >= len(scores):
        candidates = range(len(scores))
    else:
        candidates = np.argpartition(-scores, k - 1)[:k]
    return sorted(candidates, key=lambda index: (-float(scores[index]), item_ids[index] if index < len(item_ids) else str(index)))[:k]


def compact_result_for_index(item: dict[str, Any], signals: dict[str, Any], index: int) -> dict[str, Any]:
    final_score = float(signals["final_score"][index])
    return result_with_style_diagnostics(
        {
        "item_id": item["item_id"],
        "score": round(final_score, 6),
        "final_score": round(final_score, 6),
        "lexical_score": round(float(signals["lexical_score"][index]), 6),
        "embedding_score": round(float(signals["embedding_score"][index]), 6),
        "semantic_score": round(float(signals["embedding_score"][index]), 6),
        "constraint_score": round(float(signals["constraint_score"][index]), 6),
        "purpose_score": round(float(signals["purpose_score"][index]), 6),
        "signature_score": round(float(signals["signature_score"][index]), 6),
        "style_score": round(float(signals["style_score"][index]), 6),
        "metadata": item.get("metadata", {}),
        "constraint_hits": signals["hits_by_index"][index],
        }
    )


def result_with_style_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "style_risk_score": style_risk_score(result),
        "style_guardrail_action": style_guardrail_action(result),
        "risk_evidence": style_risk_evidence(result),
    }


def rank_for_item(item_id: str, scores: np.ndarray, item_ids: list[str]) -> int | None:
    if not item_id or item_id not in item_ids:
        return None
    index = item_ids.index(item_id)
    score = float(scores[index])
    better = int(np.count_nonzero(scores > score))
    tied_before = sum(1 for other_id, other_score in zip(item_ids, scores, strict=False) if float(other_score) == score and other_id < item_id)
    return better + tied_before + 1


def bm25_scores_for_query(query_tokens: list[str], index: dict[str, Any], count: int) -> np.ndarray:
    scores = np.zeros(count, dtype=np.float32)
    if not query_tokens or count <= 0:
        return scores
    doc_freq = index.get("doc_freq", {})
    doc_count = max(1, int(index.get("doc_count", 1)))
    avg_len = max(1e-9, float(index.get("avg_doc_len", 1.0)))
    doc_lengths = index.get("doc_lengths_np")
    if not isinstance(doc_lengths, np.ndarray) or doc_lengths.shape[0] != count:
        doc_lengths = np.ones(count, dtype=np.float32)
    postings = index.get("token_to_indices", {})
    k1 = 1.2
    b = 0.75
    for token in set(query_tokens):
        indices = postings.get(token)
        if indices is None or len(indices) == 0:
            continue
        df = int(doc_freq.get(token, 0))
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        selected_lengths = doc_lengths[indices]
        denom = 1.0 + k1 * (1 - b + b * selected_lengths / avg_len)
        scores[indices] += idf * (k1 + 1) / np.maximum(1e-9, denom)
    return scores


def dense_scores_for_query(query_tokens: list[str], index: dict[str, Any], count: int) -> np.ndarray:
    matrix = index.get("vector_matrix")
    if not query_tokens or count <= 0 or not isinstance(matrix, np.ndarray) or matrix.shape[0] != count:
        return np.zeros(count, dtype=np.float32)
    query_vector = np.asarray(hashed_vector(query_tokens), dtype=np.float32)
    return matrix @ query_vector


def overlap_from_sets(query: set[str], item: set[str]) -> float:
    if not query or not item:
        return 0.0
    return len(query & item) / math.sqrt(max(1, len(query)) * max(1, len(item)))


def score_item(item: dict[str, Any], *, plan: dict[str, Any], prepared_index: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = item.get("metadata", {})
    query_texts = [str(plan.get("positive_query", ""))]
    query_texts.extend(str(row.get("text", "")) for row in plan.get("rewrites", []) if isinstance(row, dict))
    if plan.get("hyde_text"):
        query_texts.append(str(plan["hyde_text"]))
    query_tokens = lexical_tokens(" ".join(query_texts))
    item_tokens = item.get("tokens", []) or lexical_tokens(item.get("text", ""))
    prepared = prepared_index or prepare_retrieval_index([item])
    lexical = bm25_score(query_tokens, item_tokens, prepared)
    dense = cosine_hash_score(query_tokens, item_tokens, prepared, item.get("item_id", ""))
    constraint, hits = constraint_score(metadata, plan)
    purpose = purpose_score(metadata, plan)
    signature = lexical_overlap_score(lexical_tokens(signature_text(plan)), item_tokens)
    style, style_hits = style_score(metadata, plan)
    hits.update(style_hits)
    visual = lexical_overlap_score(lexical_tokens(" ".join(str(value) for value in metadata.values())), item_tokens)
    final_score = lexical + dense + constraint + purpose + signature * 0.25 + style
    result = result_with_style_diagnostics(
        {
            "item_id": item["item_id"],
            "score": round(final_score, 6),
            "final_score": round(final_score, 6),
            "lexical_score": round(lexical, 6),
            "embedding_score": round(dense, 6),
            "semantic_score": round(dense, 6),
            "constraint_score": round(constraint, 6),
            "purpose_score": round(purpose, 6),
            "signature_score": round(signature, 6),
            "style_score": round(style, 6),
            "visual_score": round(visual, 6),
            "metadata": metadata,
            "constraint_hits": hits,
            "explanation": explain_score(lexical, constraint, purpose, signature, dense, style, hits),
        }
    )
    return result


def bm25_score(query_tokens: list[str], item_tokens: list[str], index: dict[str, Any]) -> float:
    if not query_tokens or not item_tokens:
        return 0.0
    doc_freq = index.get("doc_freq", {})
    doc_count = max(1, int(index.get("doc_count", 1)))
    avg_len = max(1e-9, float(index.get("avg_doc_len", 1.0)))
    freqs = Counter(item_tokens)
    score = 0.0
    k1 = 1.2
    b = 0.75
    doc_len = len(item_tokens)
    for token in set(query_tokens):
        tf = freqs.get(token, 0)
        if tf <= 0:
            continue
        df = int(doc_freq.get(token, 0))
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * doc_len / avg_len)
        score += idf * (tf * (k1 + 1)) / max(1e-9, denom)
    return score


def hashed_vector(tokens: list[str], dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(str(token).encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_hash_score(query_tokens: list[str], item_tokens: list[str], index: dict[str, Any], item_id: str) -> float:
    if not query_tokens or not item_tokens:
        return 0.0
    query_vector = hashed_vector(query_tokens)
    item_vector = (index.get("vectors", {}) or {}).get(item_id) or hashed_vector(item_tokens)
    return sum(left * right for left, right in zip(query_vector, item_vector, strict=False))


def lexical_overlap_score(query_tokens: list[str], item_tokens: list[str]) -> float:
    if not query_tokens or not item_tokens:
        return 0.0
    query = set(query_tokens)
    item = set(item_tokens)
    overlap = len(query & item)
    return overlap / math.sqrt(max(1, len(query)) * max(1, len(item)))


def constraint_score(metadata: dict[str, Any], plan: dict[str, Any]) -> tuple[float, dict[str, list[str]]]:
    score = 0.0
    hits: dict[str, list[str]] = {}
    stage = str(metadata.get("script_stage", ""))
    if stage in set(plan.get("forbidden_stage", []) or []):
        hits["forbidden_stage"] = [stage]
        return -HARD_FORBIDDEN_STAGE_VETO, hits
    if stage in set(plan.get("desired_stage", []) or []):
        score += 3.0
        hits["desired_stage"] = [stage]
    return score, hits


def style_score(metadata: dict[str, Any], plan: dict[str, Any]) -> tuple[float, dict[str, list[str]]]:
    score = 0.0
    hits: dict[str, list[str]] = {}
    item_traits = set(str(value) for value in metadata.get("style_traits", []) or [])
    item_risks = set(str(value) for value in metadata.get("style_risks", []) or [])
    positive = set(str(value) for value in plan.get("positive_style", []) or [])
    negative = set(str(value) for value in plan.get("negative_style", []) or [])
    positive_hits = sorted(item_traits & positive)
    negative_hits = sorted(item_risks & negative)
    if positive_hits:
        hits["positive_style"] = positive_hits
        score += 0.35 * len(positive_hits)
    if negative_hits:
        hits["negative_style"] = negative_hits
        score -= 1.5 * len(negative_hits)
    return score, hits


def purpose_score(metadata: dict[str, Any], plan: dict[str, Any]) -> float:
    item_purposes = set(str(value) for value in metadata.get("creative_purpose", []) or [])
    query_purposes = set(str(value) for value in plan.get("positive_purposes", []) or [])
    return 0.75 * len(item_purposes & query_purposes)


def signature_text(plan: dict[str, Any]) -> str:
    signature = plan.get("scene_signature", {}) if isinstance(plan.get("scene_signature"), dict) else {}
    parts = [str(signature.get("raw_positive_query", ""))]
    for key in ("people", "place", "actions", "objects", "emotion_function", "narrative_position", "camera_experience"):
        value = signature.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def explain_score(
    lexical: float,
    constraint: float,
    purpose: float,
    signature: float,
    dense: float,
    style: float,
    hits: dict[str, list[str]],
) -> str:
    parts = [
        f"lexical={lexical:.3f}",
        f"dense={dense:.3f}",
        f"constraint={constraint:.3f}",
        f"purpose={purpose:.3f}",
        f"signature={signature:.3f}",
        f"style={style:.3f}",
    ]
    if hits:
        parts.append(f"hits={hits}")
    return "; ".join(parts)


def expected_target(case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_prefer")
    if isinstance(expected, dict):
        return expected
    target = case.get("target")
    return target if isinstance(target, dict) else {}


def target_recall(rows: list[dict[str, Any]], k: int) -> float:
    hits = sum(1 for row in rows if target_in_top_k(row, k))
    return round(hits / max(1, len(rows)), 6)


def stage_hit(rows: list[dict[str, Any]], k: int) -> float:
    hits = 0
    for row in rows:
        target_stage = str(row.get("target_stage", ""))
        if any(result.get("metadata", {}).get("script_stage") == target_stage for result in row.get("top_results", [])[:k]):
            hits += 1
    return round(hits / max(1, len(rows)), 6)


def purpose_hit(rows: list[dict[str, Any]], k: int) -> float:
    hits = 0
    for row in rows:
        target_purposes = set(row.get("target_purposes", []) or [])
        if not target_purposes:
            continue
        for result in row.get("top_results", [])[:k]:
            purposes = set(result.get("metadata", {}).get("creative_purpose", []) or [])
            if target_purposes & purposes:
                hits += 1
                break
    return round(hits / max(1, len(rows)), 6)


def style_violation(rows: list[dict[str, Any]], k: int) -> float:
    violations = 0
    for row in rows:
        if any(result.get("constraint_hits", {}).get("negative_style") for result in row.get("top_results", [])[:k]):
            violations += 1
    return round(violations / max(1, len(rows)), 6)


def target_in_top_k(row: dict[str, Any], k: int) -> bool:
    target_id = row.get("target_item_id")
    return bool(target_id and any(result.get("item_id") == target_id for result in row.get("top_results", [])[:k]))


def score_ranks(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    ranked = sorted(rows, key=lambda row: (-float(row.get(field, 0.0) or 0.0), row["item_id"]))
    return {row["item_id"]: index for index, row in enumerate(ranked, start=1)}


def rrf_score(ranks: list[int | None], *, k: int = RRF_K) -> float:
    return sum(1.0 / (k + rank) for rank in ranks if rank is not None)


def mean_top_margin(rows: list[dict[str, Any]]) -> float:
    margins = []
    for row in rows:
        top = row.get("top_results", [])
        if len(top) >= 2:
            margins.append(float(top[0].get("score", 0.0)) - float(top[1].get("score", 0.0)))
    return round(sum(margins) / max(1, len(margins)), 6)


def low_confidence_rate(rows: list[dict[str, Any]]) -> float:
    low = 0
    for row in rows:
        top = row.get("top_results", [])
        margin = 0.0
        if len(top) >= 2:
            margin = float(top[0].get("score", 0.0)) - float(top[1].get("score", 0.0))
        if margin < 0.05 or float(row.get("planner_confidence", 1.0) or 1.0) < 0.6:
            low += 1
    return round(low / max(1, len(rows)), 6)


def write_retrieval_run(path: Path, artifact: dict[str, Any]) -> None:
    write_json(path, artifact)


def compare_run_artifacts(native_path: Path, legacy_path: Path) -> dict[str, Any]:
    native = read_json(native_path)
    legacy = read_json(legacy_path)
    native_summary = normalize_metric_names(summary_like(native))
    legacy_summary = normalize_metric_names(summary_like(legacy))
    keys = sorted(set(native_summary) | set(legacy_summary))
    delta = {}
    for key in keys:
        left = native_summary.get(key)
        right = legacy_summary.get(key)
        if isinstance(left, int | float) and isinstance(right, int | float):
            delta[key] = round(float(left) - float(right), 6)
    return {
        "method": "retrieval_lab_compare_legacy_retrieval",
        "summary": {
            "native": str(native_path),
            "legacy": str(legacy_path),
            "numeric_delta_count": len(delta),
        },
        "native_summary": native_summary,
        "legacy_summary": legacy_summary,
        "delta": delta,
    }


def summary_like(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    for key in ("summary", "overall", "graded_metrics"):
        value = report.get(key)
        if isinstance(value, dict):
            return value
    metrics = report.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("overall"), dict):
        return metrics["overall"]
    return {}


def normalize_metric_names(summary: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(summary)
    aliases = {
        "target_recall_at_1": "recall_at_1",
        "target_recall_at_3": "recall_at_3",
        "target_recall_at_10": "recall_at_10",
        "stage_hit_at_1": "desired_stage_hit_at_1",
        "stage_hit_at_3": "desired_stage_hit_at_3",
    }
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    return normalized


__all__ = [
    "DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT",
    "DEFAULT_RETRIEVAL_RUN_OUTPUT",
    "compare_run_artifacts",
    "prepare_retrieval_index",
    "retrieval_run",
    "retrieval_run_from_cases",
    "score_item",
    "score_items",
    "write_retrieval_run",
]
