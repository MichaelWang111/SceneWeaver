from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


STAGE_CANONICAL_ALIASES = {
    "technology_entrance": "technology_showcase",
}


def canonical_stage(stage: str | None) -> str:
    value = str(stage or "")
    return STAGE_CANONICAL_ALIASES.get(value, value)


def validate_qrel(row: dict[str, Any]) -> None:
    if not row.get("query_id") or not row.get("item_id"):
        raise ValueError("qrel requires query_id and item_id")
    grade = int(row.get("grade", -1))
    if grade < 0 or grade > 3:
        raise ValueError(f"qrel grade must be 0..3, got {grade}")


def load_qrels(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validate_qrel(row)
        rows.append(row)
    return rows


def write_qrels(path: Path, qrels: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in qrels:
        validate_qrel(row)
        key = (str(row["query_id"]), str(row["item_id"]))
        if key not in seen or qrel_row_priority(row) > qrel_row_priority(seen[key]):
            seen[key] = row
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in seen.values()),
        encoding="utf-8",
    )


def qrel_vote_judge_type(vote: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    judge_type = str(vote.get("judge_type", "")).lower().strip()
    if judge_type in {"human", "manual"}:
        return "human"
    if judge_type in {"llm", "model"}:
        return "llm"
    if judge_type == "bootstrap":
        return "bootstrap"
    source = str((row or {}).get("source", "")).lower()
    if source.startswith("manual"):
        return "human"
    if source.startswith("llm"):
        return "llm"
    return "bootstrap"


def qrel_has_judge_type(row: dict[str, Any], judge_type: str) -> bool:
    votes = [vote for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    if any(qrel_vote_judge_type(vote, row) == judge_type for vote in votes):
        return True
    source = str(row.get("source", "")).lower()
    return (judge_type == "human" and source.startswith("manual")) or (
        judge_type == "llm" and source.startswith("llm")
    )


def qrel_is_bootstrap_only(row: dict[str, Any]) -> bool:
    return not qrel_has_judge_type(row, "human") and not qrel_has_judge_type(row, "llm")


def qrel_has_vote_conflict(row: dict[str, Any]) -> bool:
    votes = [int(vote.get("grade", row.get("grade", 0))) for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    return len(set(votes)) > 1


def qrel_needs_adjudication(row: dict[str, Any]) -> bool:
    return bool(row.get("needs_adjudication")) or qrel_has_vote_conflict(row) or qrel_confidence(row) < 0.6


def qrel_confidence(row: dict[str, Any]) -> float:
    source = str(row.get("source", ""))
    grade = int(row.get("grade", 0))
    pooled_count = len(row.get("pooled_from", []))
    vote_rows = [vote for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    votes = [int(vote.get("grade", grade)) for vote in vote_rows]
    human_votes = [vote for vote in vote_rows if qrel_vote_judge_type(vote, row) == "human"]
    llm_votes = [vote for vote in vote_rows if qrel_vote_judge_type(vote, row) == "llm"]
    if source.startswith("manual"):
        return round(max([float(vote.get("confidence", 0.95)) for vote in human_votes] or [0.95]), 6)
    if source.startswith("llm"):
        return round(max([float(vote.get("confidence", 0.85)) for vote in llm_votes] or [0.85]), 6)
    if human_votes:
        return round(max(float(vote.get("confidence", 0.95)) for vote in human_votes), 6)
    if llm_votes:
        return round(max(float(vote.get("confidence", 0.85)) for vote in llm_votes), 6)
    if grade == 3 and "target item" in str(row.get("reason", "")):
        return 0.95
    if votes:
        if len(votes) == 1 and grade <= 1:
            return 0.5
        agreement = max(Counter(votes).values()) / len(votes)
        base = 0.45 + 0.35 * agreement + min(0.15, 0.03 * len(votes))
        if len(set(votes)) > 1:
            base -= 0.15
        return round(min(0.9, max(0.3, base)), 6)
    if pooled_count >= 3:
        return 0.75
    if pooled_count >= 2:
        return 0.65
    return 0.5


def qrels_trust_level(qrels: list[dict[str, Any]]) -> str:
    if not qrels:
        return "low"
    reviewed = sum(1 for row in qrels if not qrel_is_bootstrap_only(row))
    reviewed_rate = reviewed / len(qrels)
    needs_rate = sum(1 for row in qrels if qrel_needs_adjudication(row)) / len(qrels)
    conflict_rate = sum(1 for row in qrels if qrel_has_vote_conflict(row)) / len(qrels)
    if reviewed == 0:
        return "low"
    if reviewed_rate >= 0.5 and needs_rate <= 0.1 and conflict_rate <= 0.05:
        return "high"
    return "medium"


def qrels_audit_summary(qrels: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = [qrel_confidence(row) for row in qrels]
    pooled_counts = [len(row.get("pooled_from", [])) for row in qrels]
    conflict_count = sum(1 for row in qrels if qrel_has_vote_conflict(row))
    needs_adjudication_count = sum(1 for row in qrels if qrel_needs_adjudication(row))
    manual_count = sum(1 for row in qrels if qrel_has_judge_type(row, "human"))
    llm_count = sum(1 for row in qrels if qrel_has_judge_type(row, "llm"))
    bootstrap_only_count = sum(1 for row in qrels if qrel_is_bootstrap_only(row))
    source_counts = Counter(str(row.get("source", "unknown")) for row in qrels)
    return {
        "qrels_count": len(qrels),
        "grade_counts": dict(sorted(Counter(int(row.get("grade", 0)) for row in qrels).items())),
        "source_counts": dict(sorted(source_counts.items())),
        "mean_confidence": round(mean(confidences), 6),
        "high_confidence_rate": round(sum(1 for value in confidences if value >= 0.75) / max(1, len(confidences)), 6),
        "low_confidence_count": sum(1 for value in confidences if value < 0.6),
        "conflict_count": conflict_count,
        "conflict_rate": round(conflict_count / max(1, len(qrels)), 6),
        "vote_conflict_rate": round(conflict_count / max(1, len(qrels)), 6),
        "manual_count": manual_count,
        "llm_count": llm_count,
        "bootstrap_only_count": bootstrap_only_count,
        "needs_adjudication_count": needs_adjudication_count,
        "mean_pooled_from_count": round(mean([float(value) for value in pooled_counts]), 6),
        "manual_or_llm_count": manual_count + llm_count,
        "qrels_trust_level": qrels_trust_level(qrels),
    }


def bootstrap_qrels_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qrels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        query_id = row["case_id"]
        target_id = row.get("target_item_id")
        if target_id:
            qrels[(query_id, target_id)] = {
                "query_id": query_id,
                "item_id": target_id,
                "grade": 3,
                "reason": "target item from generated eval case",
                "source": "bootstrap",
            }
        for result in row.get("top_results", []):
            item_id = result.get("item_id")
            if not item_id:
                continue
            grade, reason = bootstrap_grade(row, result)
            key = (query_id, item_id)
            current = qrels.get(key)
            if current is None or grade > int(current["grade"]):
                qrels[key] = {
                    "query_id": query_id,
                    "item_id": item_id,
                    "grade": grade,
                    "reason": reason,
                    "source": "bootstrap",
                }
    return list(qrels.values())


def pooled_qrels_from_run_rows(run_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qrels: dict[tuple[str, str], dict[str, Any]] = {}
    for run_name, rows in run_rows.items():
        for row in rows:
            query_id = row["case_id"]
            target_id = row.get("target_item_id")
            if target_id:
                merge_qrel(
                    qrels,
                    query_id=query_id,
                    item_id=target_id,
                    grade=3,
                    reason="target item from generated eval case",
                    source="pooled_bootstrap",
                    run_name=run_name,
                    rank=row.get("target_rank"),
                )
            for rank, result in enumerate(row.get("top_results", []), start=1):
                item_id = result.get("item_id")
                if not item_id:
                    continue
                grade, reason = bootstrap_grade(row, result)
                merge_qrel(
                    qrels,
                    query_id=query_id,
                    item_id=item_id,
                    grade=grade,
                    reason=reason,
                    source="pooled_bootstrap",
                    run_name=run_name,
                    rank=rank,
                )
    return sorted(qrels.values(), key=lambda row: (row["query_id"], -int(row["grade"]), row["item_id"]))


def merge_qrel(
    qrels: dict[tuple[str, str], dict[str, Any]],
    *,
    query_id: str,
    item_id: str,
    grade: int,
    reason: str,
    source: str,
    run_name: str,
    rank: int | None,
) -> None:
    key = (query_id, item_id)
    pooled_from = {"run": run_name}
    if rank is not None:
        pooled_from["rank"] = int(rank)
    existing = qrels.get(key)
    vote = {
        "run": run_name,
        "rank": rank,
        "grade": int(grade),
        "reason": reason,
        "judge_type": "bootstrap",
        "judge_id": run_name,
        "judge_version": "pooled_v1",
    }
    if existing is None:
        qrels[key] = {
            "query_id": query_id,
            "item_id": item_id,
            "grade": int(grade),
            "reason": reason,
            "source": source,
            "pooled_from": [pooled_from],
            "grade_votes": [vote],
        }
        qrels[key]["confidence"] = qrel_confidence(qrels[key])
        return
    existing.setdefault("pooled_from", []).append(pooled_from)
    existing.setdefault("grade_votes", []).append(vote)
    if int(grade) > int(existing["grade"]):
        existing["grade"] = int(grade)
        existing["reason"] = reason
        existing["source"] = source
    existing["confidence"] = qrel_confidence(existing)


def pooled_qrels_summary(
    qrels: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    run_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_query: dict[str, set[str]] = {}
    for qrel in qrels:
        by_query.setdefault(qrel["query_id"], set()).add(qrel["item_id"])
    pool_sizes = [len(by_query.get(case["case_id"], set())) for case in cases]
    return {
        "qrels_count": len(qrels),
        "query_count": len(cases),
        "run_count": len(run_rows),
        "avg_pool_size": round(mean([float(size) for size in pool_sizes]), 6),
        "max_pool_size": max(pool_sizes) if pool_sizes else 0,
        "grade_counts": dict(sorted(Counter(int(row["grade"]) for row in qrels).items())),
    }


def bootstrap_grade(row: dict[str, Any], result: dict[str, Any]) -> tuple[int, str]:
    if result.get("constraint_hits", {}).get("negative_style"):
        return 0, "candidate violates negative style constraint"
    if result.get("item_id") == row.get("target_item_id"):
        return 3, "exact generated target"
    result_stage_value = result_stage(result)
    stage_match = result_stage_value == row.get("target_stage")
    target_purposes = set(row.get("target_purposes", []))
    result_purposes = set(result.get("metadata", {}).get("creative_purpose", []))
    purpose_match = bool(target_purposes & result_purposes)
    if stage_match and purpose_match:
        return 2, "same stage and overlapping creative purpose"
    if stage_match or purpose_match:
        return 1, "partial stage or purpose match"
    return 0, "no generated relevance signal"


def result_stage(result: dict[str, Any]) -> str:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    return canonical_stage(metadata.get("script_stage", ""))


def qrel_relevance_vote(
    *,
    query_id: str,
    item_id: str,
    grade: int,
    reason: str,
    judge_type: str,
    judge_id: str,
    judge_version: str,
    confidence: float | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    vote = {
        "query_id": str(query_id),
        "item_id": str(item_id),
        "grade": int(grade),
        "reason": str(reason),
        "judge_type": "human" if judge_type == "manual" else str(judge_type),
        "judge_id": str(judge_id),
        "judge_version": str(judge_version),
    }
    if confidence is not None:
        vote["confidence"] = float(confidence)
    if created_at:
        vote["created_at"] = str(created_at)
    return vote


def merge_adjudicated_qrels(existing_qrels: list[dict[str, Any]], votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {(str(row["query_id"]), str(row["item_id"])): dict(row) for row in existing_qrels}
    for vote in votes:
        validate_qrel(vote)
        key = (str(vote["query_id"]), str(vote["item_id"]))
        row = merged.setdefault(
            key,
            {
                "query_id": vote["query_id"],
                "item_id": vote["item_id"],
                "grade": int(vote["grade"]),
                "reason": vote.get("reason", ""),
                "source": f"{qrel_vote_judge_type(vote)}_adjudicated",
                "pooled_from": [],
                "grade_votes": [],
            },
        )
        row.setdefault("grade_votes", []).append(dict(vote))
        recompute_qrel_from_votes(row)
    return sorted(merged.values(), key=lambda row: (row["query_id"], -int(row["grade"]), row["item_id"]))


def recompute_qrel_from_votes(row: dict[str, Any]) -> None:
    votes = [vote for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    selected_vote = selected_qrel_vote(votes, row)
    if selected_vote is not None:
        row["grade"] = int(selected_vote.get("grade", row.get("grade", 0)))
        row["reason"] = str(selected_vote.get("reason", row.get("reason", "")))
        selected_type = qrel_vote_judge_type(selected_vote, row)
        if selected_type == "human":
            row["source"] = "manual_adjudicated"
        elif selected_type == "llm":
            row["source"] = "llm_adjudicated"
        else:
            row.setdefault("source", "pooled_bootstrap")
    row["needs_adjudication"] = qrel_has_vote_conflict(row)
    row["confidence"] = qrel_confidence(row)


def selected_qrel_vote(votes: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    for judge_type in ("human", "llm", "bootstrap"):
        typed = [vote for vote in votes if qrel_vote_judge_type(vote, row) == judge_type]
        if not typed:
            continue
        counts = Counter(int(vote.get("grade", 0)) for vote in typed)
        selected_grade = max(counts.items(), key=lambda pair: (pair[1], pair[0]))[0]
        selected = [vote for vote in typed if int(vote.get("grade", 0)) == selected_grade]
        return max(selected, key=lambda vote: float(vote.get("confidence", 0.0)))
    return None


def qrel_row_priority(row: dict[str, Any]) -> tuple[int, float, int]:
    if qrel_has_judge_type(row, "human"):
        tier = 3
    elif qrel_has_judge_type(row, "llm"):
        tier = 2
    else:
        tier = 1
    return (tier, qrel_confidence(row), int(row.get("grade", 0)))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def active_qrels_samples(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    existing_qrels: list[dict[str, Any]],
    sample_size: int,
    include_judged: bool,
) -> list[dict[str, Any]]:
    reviewed = {
        (row["query_id"], row["item_id"])
        for row in existing_qrels
        if not qrel_is_bootstrap_only(row) and not qrel_needs_adjudication(row)
    }
    existing_by_key = {(row["query_id"], row["item_id"]): row for row in existing_qrels}
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    ranks: dict[tuple[str, str], list[int]] = {}
    for run_name, rows in run_rows.items():
        for row in rows:
            query_id = row["case_id"]
            query_bucket = by_query.setdefault(query_id, {})
            for rank, result in enumerate(row.get("top_results", []), start=1):
                item_id = result["item_id"]
                candidate = query_bucket.setdefault(
                    item_id,
                    {
                        "query_id": query_id,
                        "item_id": item_id,
                        "user_input": row.get("user_input", ""),
                        "target_item_id": row.get("target_item_id"),
                        "metadata": result.get("metadata", {}),
                        "constraint_hits": result.get("constraint_hits", {}),
                        "scores": {},
                        "pooled_from": [],
                    },
                )
                candidate["scores"][run_name] = result.get("score")
                candidate["pooled_from"].append({"run": run_name, "rank": rank})
                ranks.setdefault((query_id, item_id), []).append(rank)
    samples = []
    for query_id, items in by_query.items():
        target_missed = all(
            row.get("target_rank") is None or row.get("target_rank", 999999) > len(row.get("top_results", []))
            for rows in run_rows.values()
            for row in rows
            if row.get("case_id") == query_id
        )
        query_top_candidates = top_candidates_for_active_review(items, ranks, limit=5)
        for item_id, candidate in items.items():
            if not include_judged and (query_id, item_id) in reviewed:
                continue
            candidate_ranks = ranks.get((query_id, item_id), [])
            reasons = active_sample_reasons(candidate, candidate_ranks, target_missed=target_missed)
            existing_qrel = existing_by_key.get((query_id, item_id))
            if existing_qrel is not None and qrel_confidence(existing_qrel) < 0.6:
                reasons.append("low_confidence_qrel")
            if existing_qrel is not None and qrel_needs_adjudication(existing_qrel):
                reasons.append("needs_adjudication")
            if not reasons:
                continue
            target_stage = target_stage_from_target_id(candidate.get("target_item_id"), run_rows, query_id)
            target_purposes = target_purposes_from_rows(run_rows, query_id)
            suggested_grade, reason = bootstrap_grade(
                {
                    "case_id": query_id,
                    "target_item_id": candidate.get("target_item_id"),
                    "target_stage": target_stage,
                    "target_purposes": target_purposes,
                },
                candidate,
            )
            samples.append(
                {
                    **candidate,
                    "query": candidate.get("user_input", ""),
                    "target": {
                        "item_id": candidate.get("target_item_id"),
                        "stage": target_stage,
                        "purposes": target_purposes,
                    },
                    "candidate_summary": reranker_candidate_summary(candidate, rank=min(candidate_ranks or [0]) or None),
                    "top_candidates": query_top_candidates,
                    "suggested_grade": suggested_grade,
                    "suggested_reason": reason,
                    "suggested_granularity": suggested_adjudication_granularity(reasons),
                    "low_confidence_reasons": low_confidence_reasons_for_active_sample(reasons, candidate_ranks),
                    "workflow_disagreement": "workflow_rank_disagreement" in reasons,
                    "existing_qrel": compact_existing_qrel(existing_qrel),
                    "adjudication_schema": {
                        "query_id": query_id,
                        "item_id": item_id,
                        "grade": "0..3",
                        "reason": "short reason",
                        "judge_type": "human|llm",
                        "confidence": "0..1 optional",
                    },
                    "reasons": reasons,
                    "priority": active_sample_priority(reasons, candidate_ranks),
                    "judged": (query_id, item_id) in reviewed,
                }
            )
    samples.sort(key=lambda row: (-float(row["priority"]), row["query_id"], row["item_id"]))
    return samples[:sample_size]


def active_sample_reasons(candidate: dict[str, Any], ranks: list[int], *, target_missed: bool) -> list[str]:
    reasons = []
    if target_missed:
        reasons.append("target_miss_query")
    if len(ranks) >= 2 and max(ranks) - min(ranks) >= 8:
        reasons.append("workflow_rank_disagreement")
    if len(candidate.get("scores", {})) == 1:
        reasons.append("single_workflow_only")
    if candidate.get("constraint_hits", {}).get("negative_style"):
        reasons.append("style_risk_candidate")
    if min(ranks or [999999]) <= 3:
        reasons.append("top3_candidate")
    return reasons


def active_sample_priority(reasons: list[str], ranks: list[int]) -> float:
    weights = {
        "target_miss_query": 3.0,
        "workflow_rank_disagreement": 2.0,
        "style_risk_candidate": 2.0,
        "needs_adjudication": 2.5,
        "low_confidence_qrel": 2.0,
        "top3_candidate": 1.5,
        "single_workflow_only": 0.75,
    }
    return round(sum(weights.get(reason, 0.5) for reason in reasons) + 1 / max(1, min(ranks or [100])), 6)


def top_candidates_for_active_review(
    items: dict[str, dict[str, Any]],
    ranks: dict[tuple[str, str], list[int]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for item_id, candidate in items.items():
        query_id = str(candidate.get("query_id", ""))
        item_ranks = ranks.get((query_id, item_id), [])
        candidates.append(
            {
                **reranker_candidate_summary(candidate, rank=min(item_ranks or [999999])),
                "pooled_from": candidate.get("pooled_from", [])[:5],
            }
        )
    candidates.sort(key=lambda row: (int(row.get("rank") or 999999), str(row.get("item_id", ""))))
    return candidates[:limit]


def compact_existing_qrel(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "grade": int(row.get("grade", 0)),
        "reason": row.get("reason", ""),
        "source": row.get("source", ""),
        "confidence": qrel_confidence(row),
        "needs_adjudication": qrel_needs_adjudication(row),
        "vote_count": len(row.get("grade_votes", [])),
    }


def suggested_adjudication_granularity(reasons: list[str]) -> str:
    if "style_risk_candidate" in reasons:
        return "constraint_level"
    if "target_miss_query" in reasons or "workflow_rank_disagreement" in reasons:
        return "scene_or_purpose_level"
    return "scene_level"


def low_confidence_reasons_for_active_sample(reasons: list[str], ranks: list[int]) -> list[str]:
    low_reasons = []
    if "single_workflow_only" in reasons:
        low_reasons.append("candidate only appears in one workflow")
    if "workflow_rank_disagreement" in reasons:
        low_reasons.append("candidate rank differs strongly across workflows")
    if "target_miss_query" in reasons:
        low_reasons.append("target is missing from the sampled pool")
    if "low_confidence_qrel" in reasons:
        low_reasons.append("existing qrel confidence is low")
    if "needs_adjudication" in reasons:
        low_reasons.append("existing qrel has conflicting votes or needs adjudication")
    if ranks and min(ranks) <= 3 and max(ranks) >= 10:
        low_reasons.append("candidate is top ranked by one workflow but weak in another")
    return low_reasons


def target_stage_from_target_id(target_id: str | None, run_rows: dict[str, list[dict[str, Any]]], query_id: str) -> str:
    for rows in run_rows.values():
        for row in rows:
            if row.get("case_id") == query_id:
                return row.get("target_stage", "")
    return ""


def target_purposes_from_rows(run_rows: dict[str, list[dict[str, Any]]], query_id: str) -> list[str]:
    for rows in run_rows.values():
        for row in rows:
            if row.get("case_id") == query_id:
                return list(row.get("target_purposes", []))
    return []


def reranker_candidate_summary(result: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    summary = {
        "item_id": result.get("item_id"),
        "rank": rank,
        "score": result.get("score"),
        "script_stage": metadata.get("script_stage"),
        "creative_purpose": metadata.get("creative_purpose", []),
        "script_usecase": metadata.get("script_usecase")
        or {
            "best_usage": metadata.get("script_usecase_best_usage"),
            "risk": metadata.get("script_usecase_risk"),
            "sentence": metadata.get("script_use_sentence"),
        },
        "scene_signature": result.get("scene_signature") or metadata.get("scene_signature") or {},
        "style_traits": metadata.get("style_traits", []),
        "style_risks": metadata.get("style_risks", []),
        "constraint_hits": result.get("constraint_hits", {}),
    }
    return {key: value for key, value in summary.items() if value not in (None, {}, [])}


__all__ = [
    "active_qrels_samples",
    "bootstrap_grade",
    "bootstrap_qrels_from_rows",
    "canonical_stage",
    "load_qrels",
    "merge_adjudicated_qrels",
    "pooled_qrels_from_run_rows",
    "pooled_qrels_summary",
    "qrel_confidence",
    "qrel_has_vote_conflict",
    "qrel_needs_adjudication",
    "qrel_relevance_vote",
    "qrel_vote_judge_type",
    "qrels_audit_summary",
    "qrels_trust_level",
    "validate_qrel",
    "write_qrels",
]
