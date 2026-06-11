from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import mocktesting.mock_retriever as mock_retriever
from mocktesting.constraint_layer import (
    parse_query_constraints,
    profile_with_weights,
    score_constraints,
)
from mocktesting.embedding_cache import EmbeddingCache, build_matrix_cache, cache_key
from mocktesting.embedding_text_builder import (
    build_item_channels,
    build_query_channels,
    load_review_items,
)
from mocktesting.eval_input_generator import generate_eval_inputs
from mocktesting.mock_retriever import build_metrics, search_index, select_best_tuning_result
from mocktesting.mock_retriever import (
    build_paraphrase_variants,
    build_fuzzy_understanding_variants,
    bootstrap_qrels_from_rows,
    compact_case_rows,
    graded_metrics,
    build_margin_report,
    build_pairwise_pairs,
    case_split,
    active_qrels_samples,
    attach_variant_metadata,
    capability_bar_svg,
    capability_delta,
    capability_report_markdown,
    compute_capability_scores,
    extract_capability_raw_metrics,
    llm_rerank_payload,
    merge_adjudicated_qrels,
    pooled_qrels_from_run_rows,
    qrel_confidence,
    qrels_trust_level,
    qrels_audit_summary,
    query_scene_signature,
    recall_bound_rows,
    recall_bound_summary,
    record_capability_cycle_command,
    rerank_row_by_llm,
    rerank_gate_decision,
    reranker_candidate_summary,
    rerank_row_by_qrels,
    retrieval_flywheel_guide,
    generate_capability_report_command,
    precompute_nonsemantic_case_signals,
    filter_cases_by_fixture,
    filter_precomputed_by_fixture,
    infer_scene_signature,
    prepare_mock_index,
    precompute_fast_case_signals,
    possible_overfit,
    rank_fast_cases,
    rank_items_for_key,
    run_llm_sample_judge,
    select_llm_sample_candidates,
    score_query_fast,
    signature_similarity,
    signature_tokens,
    split_cases,
    summarize_fuzzy_rows,
    summarize_paraphrase_rows,
    summarize_leave_one_fixture,
    validate_qrel,
)
from mocktesting.query_planner import (
    ExperimentalQueryPlan,
    build_query_channels_for_plan,
    plan_queries,
    planner_constraints,
)


def test_item_channels_include_required_texts_without_json_noise():
    item = load_review_items()[0]

    channels = build_item_channels(item)
    by_name = {channel["channel"]: channel for channel in channels}

    assert {"script_use", "visual_tags", "experience", "combined"} <= set(by_name)
    assert "脚本用途" in by_name["script_use"]["text"]
    assert "画面元素" in by_name["visual_tags"]["text"]
    assert "导演经验" in by_name["experience"]["text"]
    for channel in channels:
        assert "{" not in channel["text"]
        assert "}" not in channel["text"]
        assert channel["enabled"] is True


def test_query_channels_keep_negative_constraints_and_avoid_snake_case():
    dataset = generate_eval_inputs(limit_fixtures=1)
    hard_positive = dataset["cases"][1]
    hard_negative = dataset["cases"][2]

    assert "_" not in hard_positive["user_input"]
    query_channels = build_query_channels(hard_negative["user_input"])
    text = "\n".join(channel["text"] for channel in query_channels)

    assert "不要" in text
    assert "开场" in text
    assert {"query_script_use", "query_visual", "query_experience", "query_combined"} <= {
        channel["channel"] for channel in query_channels
    }


def test_rule_planner_matches_query_plan_and_keeps_negatives_out_of_positive_channels():
    result = plan_queries(["不要大厂味，也不要广告感，要有人味"], planner="rule")
    plan = result.plans[0]
    channels = build_query_channels_for_plan(plan)
    channel_text = "\n".join(channel["text"] for channel in channels)

    assert "要有人味" in plan.positive_query
    assert plan.negative_style == ["big_company_office", "ad_like"]
    assert planner_constraints(plan)["negative_constraints"] == ["大厂味", "广告感"]
    assert "大厂味" not in plan.positive_query
    assert "广告感" not in channel_text
    assert result.stats["negative_leak_rate"] == 0.0


def test_multi_query_planner_generates_weighted_positive_rewrites_without_negative_leak():
    result = plan_queries(["不要炫技，先让观众进入现场，要真实现场"], planner="multi_query")
    plan = result.plans[0]
    channels = build_query_channels_for_plan(plan)
    channel_text = "\n".join(channel["text"] for channel in channels)

    assert plan.planner == "multi_query"
    assert plan.rewrites
    assert plan.positive_style == ["real_location"]
    assert plan.negative_style == ["tech_showoff"]
    assert "炫技" not in plan.positive_query
    assert "炫技" not in channel_text
    assert all(channel["weight"] < 0.5 for channel in channels)


def test_llm_planner_validates_output_caches_and_falls_back_when_sample_size_is_zero(tmp_path):
    calls = {"count": 0}

    class FakeClient:
        def analyze_text_json(self, **_kwargs):
            calls["count"] += 1
            return {
                "positive_query": "不要大厂味，要有人味",
                "desired_stage": ["setup", "unknown_stage"],
                "positive_style": ["human_warmth", "bad_style"],
                "negative_style": ["big_company_office", "bad_style"],
                "rewrites": ["不要广告感，要真实现场"],
                "confidence": 0.9,
            }

    cache_path = tmp_path / "planner.jsonl"
    first = plan_queries(
        ["不要大厂味，要有人味"],
        planner="llm_multi_query",
        cache_path=cache_path,
        llm_sample_size=1,
        llm_client=FakeClient(),
    )
    second = plan_queries(
        ["不要大厂味，要有人味"],
        planner="llm_multi_query",
        cache_path=cache_path,
        llm_sample_size=0,
        llm_client=FakeClient(),
    )
    fallback = plan_queries(
        ["不要大厂味，要有人味"],
        planner="llm_structured",
        cache_path=tmp_path / "empty.jsonl",
        llm_sample_size=0,
        llm_client=FakeClient(),
    )

    assert calls["count"] == 1
    assert first.plans[0].desired_stage == ["setup"]
    assert first.plans[0].positive_style == ["human_warmth"]
    assert first.plans[0].negative_style == ["big_company_office"]
    assert "大厂味" not in first.plans[0].positive_query
    assert "广告感" not in " ".join(first.plans[0].rewrites)
    assert second.stats["cache_hits"] == 1
    assert fallback.stats["fallback_count"] == 1
    assert fallback.plans[0].planner_metadata["fallback_to"] == "rule"


def test_embedding_cache_deduplicates_and_keys_include_model_dimension(tmp_path):
    calls: list[list[str]] = []

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(index), 1.0] for index, _text in enumerate(texts)]

    cache_path = tmp_path / "cache.jsonl"
    cache = EmbeddingCache(cache_path=cache_path, model="m1", dimension=2, batch_size=25, embedder=fake_embedder)

    stats = cache.embed_texts(["alpha", "alpha", "beta"])
    assert stats == {"requested": 3, "missing": 2, "written": 2}
    assert calls == [["alpha", "beta"]]
    assert cache.batch_size == 10
    assert cache.embed_texts(["alpha", "beta"]) == {"requested": 2, "missing": 0, "written": 0}
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 2
    assert cache_key("alpha", model="m1", dimension=2) != cache_key("alpha", model="m1", dimension=3)


def test_embedding_cache_lazy_loads_only_requested_rows(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    eager = EmbeddingCache(cache_path=cache_path, model="m1", dimension=2, batch_size=10, embedder=lambda texts: [[1.0, 0.0] for _ in texts])
    eager.embed_texts(["alpha", "beta", "gamma"])

    lazy = EmbeddingCache(cache_path=cache_path, model="m1", dimension=2, batch_size=10, load_all=False)

    assert lazy.cache_report()["resident_rows"] == 0
    assert lazy.require_embedding("beta") == [1.0, 0.0]
    assert lazy.cache_report()["resident_rows"] == 1
    assert lazy.cache_report()["lazy_scan_count"] == 1


def test_embedding_cache_matrix_sidecar_avoids_jsonl_scan(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    matrix_path = tmp_path / "cache.npz"
    eager = EmbeddingCache(
        cache_path=cache_path,
        model="m1",
        dimension=2,
        batch_size=10,
        embedder=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    eager.embed_texts(["alpha", "beta", "gamma"])
    report = build_matrix_cache(cache_path, matrix_path=matrix_path, dtype="float16")

    lazy = EmbeddingCache(
        cache_path=cache_path,
        matrix_path=matrix_path,
        model="m1",
        dimension=2,
        batch_size=10,
        load_all=False,
        prefer_matrix=True,
    )

    assert report["rows"] == 3
    assert lazy.require_embedding("beta") == [1.0, 0.0]
    assert lazy.cache_report()["matrix_status"] == "loaded"
    assert lazy.cache_report()["matrix_rows"] == 3
    assert lazy.cache_report()["resident_rows"] == 0
    assert lazy.cache_report()["lazy_scan_count"] == 0


def test_graded_qrels_and_metrics_handle_bootstrap_relevance():
    rows = [
        {
            "case_id": "case_1",
            "target_item_id": "target",
            "target_stage": "setup",
            "target_purposes": ["build_reality"],
            "top_results": [
                {
                    "item_id": "target",
                    "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"]},
                    "constraint_hits": {},
                },
                {
                    "item_id": "partial",
                    "metadata": {"script_stage": "setup", "creative_purpose": ["other"]},
                    "constraint_hits": {},
                },
                {
                    "item_id": "bad",
                    "metadata": {"script_stage": "opening", "creative_purpose": []},
                    "constraint_hits": {"negative_style": ["ad_like"]},
                },
            ],
        }
    ]

    qrels = bootstrap_qrels_from_rows(rows)
    metrics = graded_metrics(rows, qrels, top_k=3)

    assert any(row["grade"] == 3 and row["item_id"] == "target" for row in qrels)
    assert any(row["grade"] == 0 and row["item_id"] == "bad" for row in qrels)
    assert metrics["nDCG@3"] > 0.9
    assert metrics["MRR@10"] == 1.0
    assert metrics["Judged@10"] == 1.0


def test_graded_qrels_reject_invalid_grade():
    with pytest.raises(ValueError, match="grade must be 0..3"):
        validate_qrel({"query_id": "q1", "item_id": "i1", "grade": 4})


def test_pooled_qrels_merge_runs_and_keep_best_grade():
    rows_a = [
        {
            "case_id": "q1",
            "target_item_id": "target",
            "target_stage": "setup",
            "target_purposes": ["build_reality"],
            "top_results": [
                {"item_id": "weak", "metadata": {"script_stage": "setup", "creative_purpose": []}, "constraint_hits": {}},
                {"item_id": "target", "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"]}, "constraint_hits": {}},
            ],
        }
    ]
    rows_b = [
        {
            "case_id": "q1",
            "target_item_id": "target",
            "target_stage": "setup",
            "target_purposes": ["build_reality"],
            "top_results": [
                {"item_id": "weak", "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"]}, "constraint_hits": {}},
            ],
        }
    ]

    qrels = pooled_qrels_from_run_rows({"rule::a": rows_a, "rule::b": rows_b})
    weak = next(row for row in qrels if row["item_id"] == "weak")
    target = next(row for row in qrels if row["item_id"] == "target")

    assert weak["grade"] == 2
    assert len(weak["pooled_from"]) == 2
    assert target["grade"] == 3


def test_qrels_audit_reports_confidence_and_vote_conflicts():
    qrels = [
        {
            "query_id": "q1",
            "item_id": "target",
            "grade": 3,
            "reason": "target item from generated eval case",
            "source": "pooled_bootstrap",
            "pooled_from": [{"run": "a"}],
            "grade_votes": [{"run": "a", "grade": 3}, {"run": "b", "grade": 2}],
        },
        {
            "query_id": "q1",
            "item_id": "weak",
            "grade": 1,
            "reason": "single weak signal",
            "source": "pooled_bootstrap",
            "pooled_from": [{"run": "a"}],
            "grade_votes": [{"run": "a", "grade": 1}],
        },
    ]

    summary = qrels_audit_summary(qrels)

    assert qrel_confidence(qrels[0]) == 0.95
    assert summary["qrels_count"] == 2
    assert summary["conflict_count"] == 1
    assert summary["low_confidence_count"] == 1


def test_merge_adjudicated_qrels_preserves_votes_and_marks_conflicts():
    existing = [
        {
            "query_id": "q1",
            "item_id": "candidate",
            "grade": 1,
            "reason": "bootstrap partial",
            "source": "pooled_bootstrap",
            "grade_votes": [{"grade": 1, "judge_type": "bootstrap", "reason": "partial"}],
        }
    ]
    votes = [
        {
            "query_id": "q1",
            "item_id": "candidate",
            "grade": 3,
            "reason": "human says this is ideal",
            "judge_type": "human",
            "judge_id": "reviewer",
            "judge_version": "v1",
            "confidence": 0.97,
        }
    ]

    merged = merge_adjudicated_qrels(existing, votes)
    row = merged[0]
    summary = qrels_audit_summary(merged)

    assert row["grade"] == 3
    assert row["source"] == "manual_adjudicated"
    assert row["needs_adjudication"] is True
    assert len(row["grade_votes"]) == 2
    assert qrel_confidence(row) == 0.97
    assert qrels_trust_level(merged) == "medium"
    assert summary["manual_count"] == 1
    assert summary["bootstrap_only_count"] == 0


def test_capability_metrics_scores_deltas_and_svg_are_stable():
    reports = [
        {
            "path": "fuzzy.json",
            "method": "mock_fuzzy_multirelevance_evaluation",
            "summary": {
                "nDCG@10": 0.6,
                "MRR@10": 0.8,
                "scene_level_recall_at_10": 0.5,
                "stage_level_hit_at_3": 0.9,
                "purpose_level_hit_at_3": 0.85,
                "style_violation_at_3": 0.06,
            },
            "graded_metrics": {},
            "elapsed_seconds": 1.0,
        },
        {
            "path": "audit.json",
            "method": "mock_qrels_audit",
            "summary": {
                "qrels_trust_level": "low",
                "qrels_count": 100,
                "manual_or_llm_count": 0,
                "bootstrap_only_count": 100,
                "needs_adjudication_count": 20,
                "vote_conflict_rate": 0.1,
            },
            "graded_metrics": {},
            "elapsed_seconds": 0.2,
        },
        {
            "path": "rerank.json",
            "method": "mock_rerank_upper_bound_comparison",
            "summary": {
                "rerank_opportunity_nDCG@10": 0.2,
                "oracle_rerank_nDCG@10": 0.74,
                "baseline_nDCG@10": 0.43,
            },
            "graded_metrics": {},
            "elapsed_seconds": 0.5,
        },
    ]

    raw = extract_capability_raw_metrics(reports, missing_reports=[])
    capabilities = compute_capability_scores(raw)
    previous = {"retrieval_quality": {"score": capabilities["retrieval_quality"]["score"] - 5}}
    delta = capability_delta(capabilities, previous)
    svg = capability_bar_svg({"capabilities": capabilities})

    assert raw["ndcg_at_10"] == 0.6
    assert raw["qrels_trust_level"] == "low"
    assert capabilities["retrieval_quality"]["score"] > 60
    assert capabilities["rerank_potential"]["kind"] == "opportunity"
    assert delta["retrieval_quality"]["score_delta"] == 5
    assert "<svg" in svg
    assert "Capability Scores" in svg


def test_capability_cycle_commands_write_registry_markdown_and_guide(tmp_path):
    fuzzy = tmp_path / "fuzzy.json"
    audit = tmp_path / "audit.json"
    rerank = tmp_path / "rerank.json"
    fuzzy.write_text(
        json.dumps(
            {
                "method": "mock_fuzzy_multirelevance_evaluation",
                "summary": {
                    "nDCG@10": 0.62,
                    "MRR@10": 0.87,
                    "scene_level_recall_at_10": 0.59,
                    "stage_level_hit_at_3": 0.91,
                    "purpose_level_hit_at_3": 0.91,
                    "style_violation_at_3": 0.06,
                },
                "elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "method": "mock_qrels_audit",
                "summary": {
                    "qrels_trust_level": "low",
                    "qrels_count": 1244,
                    "manual_or_llm_count": 0,
                    "manual_count": 0,
                    "llm_count": 0,
                    "bootstrap_only_count": 1244,
                    "needs_adjudication_count": 310,
                    "vote_conflict_rate": 0.135,
                },
                "elapsed_seconds": 0.1,
            }
        ),
        encoding="utf-8",
    )
    rerank.write_text(
        json.dumps(
            {
                "method": "mock_rerank_upper_bound_comparison",
                "summary": {
                    "rerank_opportunity_nDCG@10": 0.31,
                    "oracle_rerank_nDCG@10": 0.74,
                    "baseline_nDCG@10": 0.44,
                },
                "elapsed_seconds": 0.8,
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "cycles.jsonl"
    origin = record_capability_cycle_command(
        SimpleNamespace(
            cycle_id="origin",
            label="Origin",
            reports=[audit, fuzzy, rerank],
            registry=registry,
            output=tmp_path / "latest.json",
            as_origin=True,
        )
    )
    second = record_capability_cycle_command(
        SimpleNamespace(
            cycle_id="cycle_002",
            label="Second",
            reports=[audit, fuzzy, rerank],
            registry=registry,
            output=tmp_path / "latest2.json",
            as_origin=False,
        )
    )
    report = generate_capability_report_command(
        SimpleNamespace(
            registry=registry,
            output=tmp_path / "capability_report.md",
            chart_dir=tmp_path / "charts",
        )
    )
    markdown = (tmp_path / "capability_report.md").read_text(encoding="utf-8")
    guide = retrieval_flywheel_guide()

    assert origin["summary"]["is_origin"] is True
    assert origin["delta_vs_previous"]["retrieval_quality"]["previous_score"] is None
    assert second["summary"]["previous_cycle_id"] == "origin"
    assert report["summary"]["cycle_count"] == 2
    assert (tmp_path / "charts" / "capability_bar_latest.svg").exists()
    assert "Retrieval Capability Report" in markdown
    assert "record-capability-cycle" in json.dumps(guide, ensure_ascii=False)
    assert "generate-capability-report" in json.dumps(guide, ensure_ascii=False)


def test_recall_bound_attributes_candidate_and_fusion_failures():
    rows_by_key = {
        "baseline": [
            {"case_id": "q1", "target_item_id": "t1", "target_rank": 25, "top_results": [{"item_id": "a"}]},
            {"case_id": "q2", "target_item_id": "t2", "target_rank": None, "top_results": [{"item_id": "b"}]},
        ],
        "strong": [
            {"case_id": "q1", "target_item_id": "t1", "target_rank": 5, "top_results": [{"item_id": "t1"}]},
            {"case_id": "q2", "target_item_id": "t2", "target_rank": 200, "top_results": [{"item_id": "c"}]},
        ],
    }

    rows = recall_bound_rows(rows_by_key, baseline_key="baseline", candidate_depth=100, top_k=10)
    summary = recall_bound_summary(rows, top_k=10, candidate_depth=100)

    assert rows[0]["failure_type"] == "workflow_selection_failure"
    assert rows[1]["failure_type"] == "candidate_recall_failure"
    assert summary["oracle_recall_at_10"] == 0.5
    assert summary["oracle_recall_at_100"] == 0.5


def test_active_qrels_sampler_prioritizes_unjudged_disagreements():
    run_rows = {
        "rule::a": [
            {
                "case_id": "q1",
                "user_input": "query",
                "target_item_id": "target",
                "target_rank": 20,
                "target_stage": "setup",
                "target_purposes": ["build_reality"],
                "top_results": [
                    {"item_id": "candidate", "score": 1.0, "metadata": {"script_stage": "setup", "creative_purpose": []}, "constraint_hits": {}},
                ],
            }
        ],
        "rule::b": [
            {
                "case_id": "q1",
                "user_input": "query",
                "target_item_id": "target",
                "target_rank": 20,
                "target_stage": "setup",
                "target_purposes": ["build_reality"],
                "top_results": [
                    {"item_id": f"filler_{index}", "score": 1.0 - index * 0.01, "metadata": {}, "constraint_hits": {}}
                    for index in range(8)
                ]
                + [
                    {"item_id": "candidate", "score": 0.1, "metadata": {"script_stage": "setup", "creative_purpose": []}, "constraint_hits": {}}
                ],
            }
        ],
    }

    samples = active_qrels_samples(run_rows, existing_qrels=[], sample_size=5, include_judged=False)
    candidate = next(row for row in samples if row["item_id"] == "candidate")

    assert "target_miss_query" in candidate["reasons"]
    assert "workflow_rank_disagreement" in candidate["reasons"]
    assert candidate["suggested_grade"] == 1
    assert candidate["query"] == "query"
    assert candidate["target"]["item_id"] == "target"
    assert candidate["workflow_disagreement"] is True
    assert candidate["suggested_granularity"] == "scene_or_purpose_level"
    assert candidate["candidate_summary"]["item_id"] == "candidate"
    assert candidate["top_candidates"]
    assert "grade" in candidate["adjudication_schema"]


def test_qrels_oracle_rerank_promotes_judged_relevant_candidate():
    row = {
        "case_id": "q1",
        "target_item_id": "target",
        "target_stage": "setup",
        "target_purposes": ["build_reality"],
        "target_rank": 3,
        "top_results": [
            {"item_id": "bad", "score": 3.0, "metadata": {"script_stage": "ending", "creative_purpose": []}, "constraint_hits": {}},
            {"item_id": "target", "score": 1.0, "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"]}, "constraint_hits": {}},
        ],
    }
    qrels = [
        {"query_id": "q1", "item_id": "bad", "grade": 0, "reason": "bad", "source": "test"},
        {"query_id": "q1", "item_id": "target", "grade": 3, "reason": "target", "source": "test"},
    ]

    reranked = rerank_row_by_qrels(row, qrels, rerank_depth=2, top_k=2)

    assert reranked["top_results"][0]["item_id"] == "target"
    assert reranked["target_rank"] == 1


def test_llm_rerank_uses_ranked_ids_without_real_api(monkeypatch):
    class FakeClient:
        def analyze_text_json(self, **_kwargs):
            return {"ranked_item_ids": ["target", "bad"], "veto_item_ids": [], "reason": "target fits"}

    import sceneweaver.llm.client as client_module

    monkeypatch.setattr(client_module, "VisionLLMClient", lambda: FakeClient())
    row = {
        "case_id": "q1",
        "user_input": "query",
        "target_item_id": "target",
        "target_stage": "setup",
        "target_purposes": ["build_reality"],
        "top_results": [
            {"item_id": "bad", "score": 3.0, "metadata": {"script_stage": "ending", "creative_purpose": []}, "constraint_hits": {}},
            {"item_id": "target", "score": 1.0, "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"]}, "constraint_hits": {}},
        ],
    }

    reranked = rerank_row_by_llm(row, rerank_depth=2, top_k=2, timeout_seconds=1, retries=0)

    assert reranked["top_results"][0]["item_id"] == "target"
    assert reranked["target_rank"] == 1
    assert reranked["llm_rerank_reason"] == "target fits"


def test_reranker_payload_uses_compact_candidate_summary():
    row = {
        "case_id": "q1",
        "user_input": "query",
        "target_stage": "setup",
        "target_purposes": ["build_reality"],
        "top_results": [
            {
                "item_id": "candidate",
                "score": 1.2,
                "lexical_text": "this long debug field must not be sent",
                "metadata": {
                    "script_stage": "setup",
                    "creative_purpose": ["build_reality"],
                    "script_use_sentence": "use for entering the scene",
                    "style_risks": ["ad_like"],
                    "lexical_text": "metadata debug text must not be sent",
                },
                "constraint_hits": {"desired_stage": ["setup"]},
            }
        ],
    }

    payload = llm_rerank_payload(row, rerank_depth=1)
    summary = reranker_candidate_summary(row["top_results"][0], rank=1)

    assert payload["candidates"][0]["item_id"] == "candidate"
    assert payload["candidates"][0]["script_stage"] == "setup"
    assert "lexical_text" not in json.dumps(payload, ensure_ascii=False)
    assert summary["constraint_hits"] == {"desired_stage": ["setup"]}


def test_attach_variant_metadata_preserves_fuzzy_grouping():
    rows = [{"case_id": "case_1", "top_results": []}]
    variants = [{"case_id": "case_1", "variant_type": "fuzzy_style", "expected_granularity": "purpose_level", "source_case_id": "src"}]

    enriched = attach_variant_metadata(rows, variants)

    assert enriched[0]["variant_type"] == "fuzzy_style"
    assert enriched[0]["expected_granularity"] == "purpose_level"
    assert enriched[0]["source_case_id"] == "src"


def test_scene_signature_fallback_and_similarity_are_nonempty():
    metadata = {
        "script_stage": "setup",
        "creative_purpose": ["build_reality"],
        "script_use_sentence": "doctor checks device records in real hospital location",
    }
    card_signature = infer_scene_signature(metadata, "doctor device hospital close shot")
    plan = ExperimentalQueryPlan(
        planner="rule",
        original_text="need real hospital doctor setup",
        positive_query="real hospital doctor checks device",
        desired_stage=["setup"],
        positive_purposes=["build_reality"],
    )
    query_signature = query_scene_signature(plan)

    assert all(card_signature[field] for field in ("people", "place", "objects", "narrative_position"))
    assert query_signature["narrative_position"] == ["setup"]
    assert signature_similarity(signature_tokens(query_signature), signature_tokens(card_signature)) > 0


def test_signature_ranking_key_returns_signature_score():
    index = {
        "items": [
            {
                "item_id": "fixture::scene_001::target",
                "metadata": {
                    "script_stage": "setup",
                    "creative_purpose": ["build_reality"],
                    "script_use_sentence": "doctor checks device records in hospital",
                },
                "channels": [{"channel": "script_use", "embedding": [1.0, 0.0]}],
            },
            {
                "item_id": "distractor",
                "metadata": {
                    "script_stage": "ending",
                    "creative_purpose": ["express_value"],
                    "script_use_sentence": "executive slogan brand film",
                },
                "channels": [{"channel": "script_use", "embedding": [1.0, 0.0]}],
            },
        ]
    }

    class FakeCache:
        def require_embedding(self, text: str):
            return [1.0, 0.0]

    cases = [
        {
            "case_id": "case_1",
            "case_type": "simple_positive",
            "expected_relation": "should_match",
            "user_input": "need real hospital doctor setup",
            "target": {
                "fixture_id": "fixture",
                "scene_id": "scene_001",
                "retrieval_id": "target",
                "script_stage": "setup",
                "creative_purpose": ["build_reality"],
            },
        }
    ]
    prepared = prepare_mock_index(index)
    plan = ExperimentalQueryPlan(
        planner="rule",
        original_text="need real hospital doctor setup",
        positive_query="real hospital doctor checks device",
        desired_stage=["setup"],
        positive_purposes=["build_reality"],
    )
    signals = precompute_fast_case_signals(
        prepared,
        FakeCache(),
        cases,
        constraint_profile=profile_with_weights(
            desired_stage_bonus=0.12,
            forbidden_stage_penalty=0.18,
            negative_constraint_penalty=0.08,
        ),
        query_plans=[plan],
    )
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key="signature_only",
        constraint_profile=profile_with_weights(
            desired_stage_bonus=0.12,
            forbidden_stage_penalty=0.18,
            negative_constraint_penalty=0.08,
        ),
        top_k=2,
    )

    assert rows[0]["top_results"][0]["item_id"] == "fixture::scene_001::target"
    assert rows[0]["target_rank"] == 1
    assert rows[0]["top_results"][0]["signature_score"] > rows[0]["top_results"][1]["signature_score"]


def test_nonsemantic_fallback_can_rank_with_lexical_constraints_signature():
    index = {
        "items": [
            {
                "item_id": "fixture::scene_001::target",
                "metadata": {
                    "script_stage": "setup",
                    "creative_purpose": ["build_reality"],
                    "script_use_sentence": "doctor checks device records in hospital",
                },
                "channels": [{"channel": "script_use", "text": "doctor hospital real setup", "embedding": [1.0, 0.0]}],
            },
            {
                "item_id": "fixture::scene_002::bad",
                "metadata": {
                    "script_stage": "technology_showcase",
                    "creative_purpose": ["show_technology"],
                    "script_use_sentence": "product pitch slogan cold tech demo",
                },
                "channels": [{"channel": "script_use", "text": "product pitch slogan cold tech", "embedding": [0.0, 1.0]}],
            },
        ]
    }
    case = {
        "case_id": "case_1",
        "case_type": "simple_positive",
        "expected_relation": "should_match",
        "user_input": "need real hospital doctor setup without product pitch",
        "target": {
            "fixture_id": "fixture",
            "scene_id": "scene_001",
            "retrieval_id": "target",
            "script_stage": "setup",
            "creative_purpose": ["build_reality"],
        },
    }
    plan = ExperimentalQueryPlan(
        planner="rule",
        original_text=case["user_input"],
        positive_query="real hospital doctor setup",
        desired_stage=["setup"],
        forbidden_stage=["technology_showcase"],
        positive_purposes=["build_reality"],
        negative_style=["product_pitch"],
    )
    profile = profile_with_weights(desired_stage_bonus=0.12, forbidden_stage_penalty=0.18, negative_constraint_penalty=0.08)
    prepared = prepare_mock_index(index)
    signals = precompute_nonsemantic_case_signals(prepared, [case], query_plans=[plan])
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key="lexical_constraints_signature",
        constraint_profile=profile,
        top_k=2,
    )

    assert rows[0]["top_results"][0]["item_id"] == "fixture::scene_001::target"
    assert rows[0]["target_rank"] == 1
    assert "signature" in signals[0].computed


def test_rerank_gate_selects_low_margin_and_style_risk():
    row = {
        "case_id": "case_1",
        "user_input": "query",
        "target_rank": 8,
        "planner_confidence": 0.5,
        "query_constraints": {"negative_constraints": ["ad_like"]},
        "top_results": [
            {"score": 1.0, "item_id": "bad", "constraint_hits": {"negative_style": ["ad_like"]}},
            {"score": 0.96, "item_id": "target", "constraint_hits": {}},
        ],
    }

    decision = rerank_gate_decision(row)

    assert decision["should_rerank"] is True
    assert {"low_margin", "low_planner_confidence", "has_negative_constraints", "style_risk", "target_near_miss"} <= set(
        decision["gate_reasons"]
    )


def test_search_index_uses_weighted_channel_scores():
    index = {
        "items": [
            {
                "item_id": "target",
                "metadata": {"scene_id": "scene_001"},
                "channels": [
                    {"channel": "script_use", "embedding": [1.0, 0.0]},
                    {"channel": "experience", "embedding": [1.0, 0.0]},
                    {"channel": "visual_tags", "embedding": [0.0, 1.0]},
                    {"channel": "combined", "embedding": [1.0, 0.0]},
                ],
            },
            {
                "item_id": "distractor",
                "metadata": {"scene_id": "scene_002"},
                "channels": [
                    {"channel": "script_use", "embedding": [0.0, 1.0]},
                    {"channel": "experience", "embedding": [0.0, 1.0]},
                    {"channel": "visual_tags", "embedding": [1.0, 0.0]},
                    {"channel": "combined", "embedding": [0.0, 1.0]},
                ],
            },
        ]
    }

    class FakeCache:
        def require_embedding(self, text: str):
            return [1.0, 0.0]

    query_channels = [
        {"channel": "query_script_use", "target_channel": "script_use", "text": "q", "weight": 0.5, "enabled": True},
        {"channel": "query_experience", "target_channel": "experience", "text": "q", "weight": 0.25, "enabled": True},
        {"channel": "query_visual", "target_channel": "visual_tags", "text": "q", "weight": 0.15, "enabled": True},
        {"channel": "query_combined", "target_channel": "combined", "text": "q", "weight": 0.10, "enabled": True},
    ]

    results = search_index(index, query_channels, FakeCache(), top_k=2)

    assert results[0]["item_id"] == "target"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["embedding_score"] == results[0]["final_score"]


def test_fast_search_matches_weighted_channel_scores():
    index = {
        "items": [
            {
                "item_id": "target",
                "metadata": {"scene_id": "scene_001"},
                "channels": [
                    {"channel": "script_use", "embedding": [1.0, 0.0]},
                    {"channel": "experience", "embedding": [1.0, 0.0]},
                    {"channel": "visual_tags", "embedding": [0.0, 1.0]},
                    {"channel": "combined", "embedding": [1.0, 0.0]},
                ],
            },
            {
                "item_id": "distractor",
                "metadata": {"scene_id": "scene_002"},
                "channels": [
                    {"channel": "script_use", "embedding": [0.0, 1.0]},
                    {"channel": "experience", "embedding": [0.0, 1.0]},
                    {"channel": "visual_tags", "embedding": [1.0, 0.0]},
                    {"channel": "combined", "embedding": [0.0, 1.0]},
                ],
            },
        ]
    }

    class FakeCache:
        def require_embedding(self, text: str):
            return [1.0, 0.0]

    cases = [
        {
            "case_id": "case_001",
            "case_type": "simple_positive",
            "expected_relation": "should_match",
            "user_input": "query",
            "target": {"fixture_id": "fixture", "scene_id": "scene_001", "retrieval_id": "target"},
        }
    ]
    profile = profile_with_weights(desired_stage_bonus=0.12, forbidden_stage_penalty=0.18, negative_constraint_penalty=0.08)
    prepared = prepare_mock_index(index)
    signals = precompute_fast_case_signals(prepared, FakeCache(), cases, constraint_profile=profile)
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key="semantic_only",
        constraint_profile=profile,
        top_k=2,
    )

    assert rows[0]["top_results"][0]["item_id"] == "target"
    assert rows[0]["top_results"][0]["score"] > rows[0]["top_results"][1]["score"]
    assert signals[0].lexical_scores is None


def test_fast_score_accumulates_duplicate_target_channels_from_multi_query():
    index = {
        "items": [
            {
                "item_id": "target",
                "metadata": {"scene_id": "scene_001"},
                "channels": [{"channel": "script_use", "embedding": [1.0, 0.0]}],
            }
        ]
    }

    class FakeCache:
        def require_embedding(self, text: str):
            return [1.0, 0.0]

    plan = ExperimentalQueryPlan(
        planner="multi_query",
        original_text="query",
        positive_query="query",
        rewrites=["rewrite"],
    )
    prepared = prepare_mock_index(index)
    query_channels = [
        channel
        for channel in build_query_channels_for_plan(plan)
        if channel["target_channel"] == "script_use"
    ]
    semantic_scores, channel_scores = score_query_fast(prepared, FakeCache(), query_channels)

    assert len(query_channels) == 2
    assert semantic_scores.tolist() == [0.5]
    assert channel_scores["script_use"].tolist() == [0.5]


def test_prepared_bm25_matches_existing_scorer():
    index = {
        "items": [
            {"item_id": "a", "metadata": {"script_stage": "setup"}, "channels": []},
            {"item_id": "b", "metadata": {"script_stage": "opening"}, "channels": []},
        ]
    }
    prepared = prepare_mock_index(index)
    query_terms = ["setup"]

    assert prepared.bm25_scores(query_terms).tolist() == mock_retriever.bm25_scores(
        query_terms,
        [mock_retriever.tokenize(text) for text in prepared.lexical_texts],
    )


def test_fast_hybrid_ranking_matches_existing_ranking_top_result():
    index = {
        "items": [
            {
                "item_id": "target",
                "metadata": {
                    "script_stage": "setup",
                    "creative_purpose": ["establish_context"],
                    "script_use_sentence": "setup human documentary",
                },
                "channels": [
                    {"channel": "script_use", "embedding": [1.0, 0.0]},
                    {"channel": "experience", "embedding": [1.0, 0.0]},
                    {"channel": "visual_tags", "embedding": [0.0, 1.0]},
                    {"channel": "combined", "embedding": [1.0, 0.0]},
                ],
            },
            {
                "item_id": "distractor",
                "metadata": {
                    "script_stage": "opening",
                    "creative_purpose": ["show_scale"],
                    "script_use_sentence": "opening office slogan",
                },
                "channels": [
                    {"channel": "script_use", "embedding": [0.0, 1.0]},
                    {"channel": "experience", "embedding": [0.0, 1.0]},
                    {"channel": "visual_tags", "embedding": [1.0, 0.0]},
                    {"channel": "combined", "embedding": [0.0, 1.0]},
                ],
            },
        ]
    }

    class FakeCache:
        def require_embedding(self, text: str):
            return [1.0, 0.0]

    cases = [
        {
            "case_id": "case_001",
            "case_type": "simple_positive",
            "expected_relation": "should_match",
            "user_input": "要 setup human documentary，不要开场",
            "target": {"fixture_id": "fixture", "scene_id": "scene_001", "retrieval_id": "target"},
        }
    ]
    profile = profile_with_weights(desired_stage_bonus=0.12, forbidden_stage_penalty=0.18, negative_constraint_penalty=0.08)

    old_precomputed = mock_retriever.precompute_embedding_rankings(index, FakeCache(), cases)
    old_rows = mock_retriever.rank_precomputed_cases(
        old_precomputed,
        ranking_key="hybrid_rrf_constraints",
        constraint_profile=profile,
        top_k=2,
    )
    prepared = prepare_mock_index(index)
    signals = precompute_fast_case_signals(prepared, FakeCache(), cases, constraint_profile=profile)
    fast_rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key="hybrid_rrf_constraints",
        constraint_profile=profile,
        top_k=2,
    )

    assert fast_rows[0]["top_results"][0]["item_id"] == old_rows[0]["top_results"][0]["item_id"]


def test_compact_case_rows_hide_debug_text_by_default():
    rows = [
        {
            "case_id": "case_001",
            "all_results": [],
            "top_results": [
                {"item_id": "item_001", "score": 1.0, "lexical_text": "long text"},
            ],
        }
    ]

    compact = compact_case_rows(rows, include_debug_text=False)
    debug = compact_case_rows(rows, include_debug_text=True)

    assert "all_results" not in compact[0]
    assert "lexical_text" not in compact[0]["top_results"][0]
    assert debug[0]["top_results"][0]["lexical_text"] == "long text"


def test_search_index_can_disable_constraints_without_changing_embedding_order():
    index = {
        "items": [
            {
                "item_id": "opening",
                "metadata": {"script_stage": "opening"},
                "channels": [{"channel": "script_use", "embedding": [1.0, 0.0]}],
            },
            {
                "item_id": "setup",
                "metadata": {"script_stage": "setup"},
                "channels": [{"channel": "script_use", "embedding": [0.8, 0.2]}],
            },
        ]
    }

    class FakeCache:
        def require_embedding(self, text: str):
            return [1.0, 0.0]

    query_channels = [
        {"channel": "query_script_use", "target_channel": "script_use", "text": "q", "weight": 1.0, "enabled": True}
    ]

    embedding_only = search_index(index, query_channels, FakeCache(), top_k=2, constraints_enabled=False)
    constrained = search_index(
        index,
        query_channels,
        FakeCache(),
        top_k=2,
        user_input="不要做成开场，我真正要的是铺垫",
        constraints_enabled=True,
    )

    assert [row["item_id"] for row in embedding_only] == ["opening", "setup"]
    assert embedding_only[0]["constraint_score"] == 0.0
    assert constrained[0]["item_id"] == "setup"
    assert constrained[1]["constraint_hits"]["forbidden_stage"] == ["opening"]


def test_parse_query_constraints_stage_desire_and_forbidden():
    constraints = parse_query_constraints("不要做成开场，我真正要的是铺垫")

    assert constraints["forbidden_stage"] == ["opening"]
    assert constraints["desired_stage"] == ["setup"]


def test_parse_query_constraints_keeps_non_stage_negative_text():
    constraints = parse_query_constraints("避免技术展示，要更像价值表达，不要互联网大厂味")

    assert constraints["forbidden_stage"] == ["technology_showcase"]
    assert constraints["desired_stage"] == ["value_expression"]
    assert any("互联网大厂味" in item for item in constraints["negative_constraints"])


def test_parse_query_constraints_canonicalizes_stage_aliases():
    constraints = parse_query_constraints("避免技术入场，我真正要的是技术展示")

    assert constraints["forbidden_stage"] == ["technology_showcase"]
    assert constraints["desired_stage"] == ["technology_showcase"]


def test_score_constraints_applies_bonus_penalty_and_penalty_wins():
    profile = profile_with_weights(
        desired_stage_bonus=0.12,
        forbidden_stage_penalty=0.18,
        negative_constraint_penalty=0.08,
    )

    bonus_score, bonus_hits = score_constraints({"desired_stage": ["setup"]}, {"script_stage": "setup"}, profile)
    penalty_score, penalty_hits = score_constraints({"forbidden_stage": ["setup"]}, {"script_stage": "setup"}, profile)
    conflict_score, conflict_hits = score_constraints(
        {"desired_stage": ["setup"], "forbidden_stage": ["setup"]},
        {"script_stage": "setup"},
        profile,
    )

    assert bonus_score == 0.12
    assert bonus_hits == {"desired_stage": ["setup"]}
    assert penalty_score == -0.18
    assert penalty_hits == {"forbidden_stage": ["setup"]}
    assert conflict_score == -0.18
    assert conflict_hits == {"forbidden_stage": ["setup"]}


def test_score_constraints_canonicalizes_item_stage_alias():
    profile = profile_with_weights(
        desired_stage_bonus=0.12,
        forbidden_stage_penalty=0.18,
        negative_constraint_penalty=0.08,
    )

    score, hits = score_constraints(
        {"desired_stage": ["technology_showcase"]},
        {"script_stage": "technology_entrance"},
        profile,
    )

    assert score == 0.12
    assert hits == {"desired_stage": ["technology_showcase"]}


def test_metrics_include_recall_mrr_and_hard_negative_margin():
    rows = [
        {"case_type": "simple_positive", "expected_relation": "should_match", "target_rank": 1},
        {"case_type": "hard_positive", "expected_relation": "should_match", "target_rank": 4},
        {
            "case_type": "hard_negative",
            "expected_relation": "should_not_match",
            "target_rank": 2,
            "expected_prefer_margin": 0.3,
        },
    ]

    metrics = build_metrics(rows)

    assert metrics["overall"]["recall_at_1"] == 0.5
    assert metrics["overall"]["recall_at_3"] == 0.5
    assert metrics["overall"]["recall_at_10"] == 1.0
    assert metrics["overall"]["mrr"] == 0.625
    assert metrics["overall"]["hard_negative_expected_prefer_margin_positive_rate"] == 1.0
    assert metrics["overall"]["forbidden_stage_violation_at_1"] == 0.0


def test_select_best_tuning_result_prioritizes_hard_negative_margin():
    weak = {
        "weights": {"forbidden_stage_penalty": 0.1},
        "selection_score": [0.4, 1.0, 1.0],
    }
    strong = {
        "weights": {"forbidden_stage_penalty": 0.28},
        "selection_score": [0.7, 0.9, 0.95],
    }

    assert select_best_tuning_result([weak, strong]) == strong


def test_case_splits_are_stable_disjoint_and_cover_all_cases():
    cases = [
        {"case_id": f"case_{index:03d}", "case_type": ["simple_positive", "hard_positive", "hard_negative"][index % 3]}
        for index in range(300)
    ]
    dev = split_cases(cases, "dev")
    test = split_cases(cases, "test")
    hidden = split_cases(cases, "hidden")

    assert case_split("case_001") == case_split("case_001")
    assert {case["case_id"] for case in dev}.isdisjoint({case["case_id"] for case in test})
    assert {case["case_id"] for case in dev}.isdisjoint({case["case_id"] for case in hidden})
    assert {case["case_id"] for case in test}.isdisjoint({case["case_id"] for case in hidden})
    assert len(dev) + len(test) + len(hidden) == len(split_cases(cases, "all"))
    for case_type in ("simple_positive", "hard_positive", "hard_negative"):
        assert any(case["case_type"] == case_type for case in dev)
        assert any(case["case_type"] == case_type for case in test)
        assert any(case["case_type"] == case_type for case in hidden)


def test_constraint_violation_metrics_check_top_results():
    rows = [
        {
            "case_type": "hard_negative",
            "expected_relation": "should_not_match",
            "target_rank": 2,
            "expected_prefer_margin": -0.1,
            "query_constraints": {"forbidden_stage": ["opening"], "desired_stage": ["setup"]},
            "top_results": [
                {"metadata": {"script_stage": "opening"}},
                {"metadata": {"script_stage": "setup"}},
            ],
        },
        {
            "case_type": "hard_negative",
            "expected_relation": "should_not_match",
            "target_rank": 2,
            "expected_prefer_margin": 0.1,
            "query_constraints": {"forbidden_stage": ["opening"], "desired_stage": ["setup"]},
            "top_results": [
                {"metadata": {"script_stage": "value_expression"}},
                {"metadata": {"script_stage": "setup"}},
            ],
        },
    ]

    metrics = build_metrics(rows)["overall"]

    assert metrics["forbidden_stage_violation_at_1"] == 0.5
    assert metrics["forbidden_stage_violation_at_3"] == 0.5
    assert metrics["desired_stage_hit_at_1"] == 0.0
    assert metrics["desired_stage_hit_at_3"] == 1.0


def test_possible_overfit_uses_dev_gain_and_test_drop():
    dev_before = {"by_case_type": {"hard_negative": {"hard_negative_expected_prefer_margin_positive_rate": 0.4}}}
    dev_after = {"by_case_type": {"hard_negative": {"hard_negative_expected_prefer_margin_positive_rate": 0.8}}}
    test_before = {"by_case_type": {"hard_negative": {"hard_negative_expected_prefer_margin_positive_rate": 0.7}}}
    test_after = {"by_case_type": {"hard_negative": {"hard_negative_expected_prefer_margin_positive_rate": 0.6}}}

    assert possible_overfit(dev_before, dev_after, test_before, test_after) is True


def test_fixture_filters_keep_train_and_test_separate():
    cases = [
        {"case_id": "a", "target": {"fixture_id": "fixture_a"}},
        {"case_id": "b", "target": {"fixture_id": "fixture_b"}},
    ]
    precomputed = [{"case": case, "embedding_ranked": []} for case in cases]

    assert [case["case_id"] for case in filter_cases_by_fixture(cases, "fixture_a", include=True)] == ["a"]
    assert [case["case_id"] for case in filter_cases_by_fixture(cases, "fixture_a", include=False)] == ["b"]
    assert [row["case"]["case_id"] for row in filter_precomputed_by_fixture(precomputed, "fixture_a", include=True)] == ["a"]
    assert [row["case"]["case_id"] for row in filter_precomputed_by_fixture(precomputed, "fixture_a", include=False)] == ["b"]


def test_leave_one_fixture_summary_finds_worst_fixture():
    report = summarize_leave_one_fixture(
        [
            {
                "fixture_id": "good",
                "test_metrics": {
                    "by_case_type": {
                        "hard_negative": {"hard_negative_expected_prefer_margin_positive_rate": 0.8},
                        "simple_positive": {"recall_at_3": 1.0},
                        "hard_positive": {"recall_at_10": 1.0},
                    }
                },
            },
            {
                "fixture_id": "bad",
                "test_metrics": {
                    "by_case_type": {
                        "hard_negative": {"hard_negative_expected_prefer_margin_positive_rate": 0.4},
                        "simple_positive": {"recall_at_3": 0.9},
                        "hard_positive": {"recall_at_10": 0.8},
                    }
                },
            },
        ]
    )

    assert report["fixture_count"] == 2
    assert report["min_hard_negative_margin_positive_rate"] == 0.4
    assert report["worst_fixture_id"] == "bad"
    assert report["possible_overfit"] is True


def test_rank_items_for_key_uses_single_channels_and_constraints_only():
    rows = [
        {
            "item_id": "script",
            "embedding_score": 0.4,
            "channel_scores": {"script_use": 0.4, "visual_tags": 0.1},
            "metadata": {"script_stage": "setup"},
        },
        {
            "item_id": "visual",
            "embedding_score": 0.5,
            "channel_scores": {"script_use": 0.1, "visual_tags": 0.5},
            "metadata": {"script_stage": "opening"},
        },
    ]
    profile = profile_with_weights(
        desired_stage_bonus=0.12,
        forbidden_stage_penalty=0.18,
        negative_constraint_penalty=0.08,
    )
    constraints = {"desired_stage": ["setup"], "forbidden_stage": ["opening"]}

    script_ranked = rank_items_for_key(rows, ranking_key="script_use_only", query_constraints=constraints, constraint_profile=profile)
    visual_ranked = rank_items_for_key(rows, ranking_key="visual_tags_only", query_constraints=constraints, constraint_profile=profile)
    constraints_ranked = rank_items_for_key(rows, ranking_key="constraints_only", query_constraints=constraints, constraint_profile=profile)
    final_ranked = rank_items_for_key(rows, ranking_key="final_score", query_constraints=constraints, constraint_profile=profile)

    assert script_ranked[0]["item_id"] == "script"
    assert visual_ranked[0]["item_id"] == "visual"
    assert constraints_ranked[0]["item_id"] == "script"
    assert final_ranked[0]["item_id"] == "script"
    assert final_ranked[1]["score"] < final_ranked[1]["embedding_score"]


def test_pairwise_pairs_include_expected_prefer_and_skip_missing_wrong_stage():
    index = {
        "items": [
            {"item_id": "fixture_a::scene_1::ret", "metadata": {"fixture_id": "fixture_a", "script_stage": "opening"}},
            {"item_id": "fixture_b::scene_2::ret", "metadata": {"fixture_id": "fixture_b", "script_stage": "setup"}},
        ]
    }
    precomputed = [
        {
            "case": {
                "case_id": "positive",
                "case_type": "simple_positive",
                "target": {"fixture_id": "fixture_a", "scene_id": "scene_1", "retrieval_id": "ret", "script_stage": "opening"},
            },
            "embedding_ranked": [
                {"item_id": "fixture_a::scene_1::ret", "metadata": {"script_stage": "opening"}},
                {"item_id": "fixture_b::scene_2::ret", "metadata": {"script_stage": "setup"}},
            ],
        },
        {
            "case": {
                "case_id": "negative",
                "case_type": "hard_negative",
                "target": {"fixture_id": "fixture_a", "scene_id": "scene_1", "retrieval_id": "ret", "script_stage": "opening"},
                "expected_prefer": {"fixture_id": "fixture_b", "scene_id": "scene_2", "retrieval_id": "ret"},
            },
            "embedding_ranked": [],
        },
    ]

    pairs = build_pairwise_pairs(precomputed, index=index, top_k=2)

    assert pairs["positive_vs_random"][0]["better_item_id"] == "fixture_a::scene_1::ret"
    assert pairs["positive_vs_wrong_stage"][0]["worse_item_id"] == "fixture_b::scene_2::ret"
    assert pairs["expected_prefer_vs_forbidden"][0] == {
        "case_id": "negative",
        "better_item_id": "fixture_b::scene_2::ret",
        "worse_item_id": "fixture_a::scene_1::ret",
    }


def test_margin_report_buckets_confidence_levels():
    rows = [
        {"case_id": "low", "case_type": "hard_positive", "top_results": [{"item_id": "a", "score": 1.0}, {"item_id": "b", "score": 0.99}]},
        {"case_id": "medium", "case_type": "hard_positive", "top_results": [{"item_id": "a", "score": 1.0}, {"item_id": "b", "score": 0.97}]},
        {"case_id": "high", "case_type": "hard_negative", "expected_prefer_margin": 0.08, "top_results": [{"item_id": "a", "score": 1.0}]},
    ]

    report = build_margin_report(rows)

    assert report["low_confidence_rate"] == 0.333333
    assert report["medium_confidence_rate"] == 0.333333
    assert report["high_confidence_rate"] == 0.333333
    assert report["cases"][0]["confidence"] == "low"


def test_llm_sample_candidates_prioritize_low_confidence_and_can_use_fake_judge(monkeypatch):
    final_rows = [
        {"case_id": "medium", "case_type": "simple_positive", "user_input": "q", "top_results": [{"item_id": "a", "score": 1.0}, {"item_id": "b", "score": 0.96}]},
        {"case_id": "low", "case_type": "hard_negative", "user_input": "q", "expected_prefer_margin": 0.01, "top_results": [{"item_id": "c", "score": 1.0}]},
    ]
    embedding_rows = [
        {"case_id": "medium", "top_results": [{"item_id": "a", "score": 1.0}]},
        {"case_id": "low", "top_results": [{"item_id": "c", "score": 1.0}]},
    ]
    ranked_by_key = {"final_score": final_rows, "embedding_only": embedding_rows}

    assert select_llm_sample_candidates(ranked_by_key, sample_size=1)[0]["case_id"] == "low"

    def fake_judge(row, *, timeout_seconds, retries):
        return {"case_id": row["case_id"], "judgements": [{"item_id": "x", "score": 3}]}

    monkeypatch.setattr(mock_retriever, "judge_top_results_with_llm", fake_judge)
    report = run_llm_sample_judge(ranked_by_key=ranked_by_key, sample_size=2, timeout_seconds=1, retries=0)

    assert report["attempted"] == 2
    assert report["judged"] == 2
    assert report["precision_at_3"] == 1.0


def test_compare_query_understanding_command_runs_with_fake_embedding_cache(tmp_path, monkeypatch):
    index_path = tmp_path / "index.json"
    inputs_path = tmp_path / "inputs.json"
    index_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "item_id": "fixture::scene_001::ret",
                        "metadata": {
                            "fixture_id": "fixture",
                            "script_stage": "setup",
                            "creative_purpose": ["build_reality"],
                            "script_use_sentence": "setup real location",
                        },
                        "channels": [
                            {"channel": "script_use", "text": "setup", "embedding": [1.0, 0.0]},
                            {"channel": "experience", "text": "reality", "embedding": [1.0, 0.0]},
                            {"channel": "visual_tags", "text": "doctor", "embedding": [0.0, 1.0]},
                            {"channel": "combined", "text": "setup reality", "embedding": [1.0, 0.0]},
                        ],
                    },
                    {
                        "item_id": "fixture::scene_002::ret",
                        "metadata": {
                            "fixture_id": "fixture",
                            "script_stage": "technology_showcase",
                            "creative_purpose": ["show_technology"],
                            "script_use_sentence": "tech showoff",
                        },
                        "channels": [
                            {"channel": "script_use", "text": "tech", "embedding": [0.0, 1.0]},
                            {"channel": "experience", "text": "tech", "embedding": [0.0, 1.0]},
                            {"channel": "visual_tags", "text": "screen", "embedding": [1.0, 0.0]},
                            {"channel": "combined", "text": "tech", "embedding": [0.0, 1.0]},
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    inputs_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_001",
                        "case_type": "simple_positive",
                        "expected_relation": "should_match",
                        "user_input": "要铺垫，建立真实感，不要炫技",
                        "target": {
                            "fixture_id": "fixture",
                            "scene_id": "scene_001",
                            "retrieval_id": "ret",
                            "script_stage": "setup",
                            "creative_purpose": ["build_reality"],
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class FakeCache:
        def __init__(self, **_kwargs):
            self.last_requested = 0
            pass

        def embed_texts(self, texts, dry_run=False):
            self.last_requested = len(texts)
            return {"requested": len(texts), "missing": 0, "written": 0}

        def require_embedding(self, text):
            return [1.0, 0.0]

        def cache_report(self):
            return {
                "load_mode": "lazy",
                "resident_rows": 0,
                "last_embed_requested": self.last_requested,
                "last_embed_missing": 0,
                "last_embed_written": 0,
            }

    monkeypatch.setattr(mock_retriever, "EmbeddingCache", FakeCache)
    args = SimpleNamespace(
        index=index_path,
        cache=tmp_path / "cache.jsonl",
        model="fake",
        dimension=2,
        embedding_batch_size=10,
        inputs=inputs_path,
        limit=0,
        split="all",
        top_k=2,
        ranking_key="hybrid_rrf_constraints",
        query_planners="rule,multi_query",
        constraint_profile=tmp_path / "profile.json",
        planner_cache=tmp_path / "planner.jsonl",
        llm_planner_sample_size=0,
        planner_timeout_seconds=1.0,
        planner_retries=0,
        require_llm_planner=False,
        include_debug_text=False,
        include_planner_debug=False,
    )

    report = mock_retriever.compare_query_understanding_command(args)

    assert report["summary"]["best_planner"] in {"rule", "multi_query"}
    assert set(report["planners"]) == {"rule", "multi_query"}
    assert report["planners"]["multi_query"]["planner_summary"]["negative_leak_rate"] == 0.0


def test_build_paraphrase_variants_creates_natural_query_types():
    case = {
        "case_id": "simple_positive__fixture__scene_001",
        "target": {
            "script_stage": "opening",
            "creative_purpose": ["establish_problem", "build_reality"],
            "industry": "medical_technology_and_digital_health",
        },
        "target_summary": "适合用于开场，建立真实压力。",
        "target_tags_text": "医生 屏幕 诊室 清晨",
    }

    variants = build_paraphrase_variants(case)

    assert [variant["variant_type"] for variant in variants] == [
        "explicit",
        "fuzzy",
        "style",
        "negative",
        "constraint_first",
        "human_value",
        "director_brief",
        "mixed",
    ]
    assert all(variant["target"] == case["target"] for variant in variants)
    assert any("不要炫技" in variant["user_input"] for variant in variants)
    mixed = [variant for variant in variants if variant["variant_type"] == "mixed"][0]
    assert mixed["user_input"].startswith("真正要的是")
    assert "不要让它变成技术展示" in mixed["user_input"]


def test_build_fuzzy_understanding_variants_tracks_granularity_levels():
    case = {
        "case_id": "simple_positive__fixture__scene_001",
        "expected_relation": "should_match",
        "target": {
            "script_stage": "setup",
            "creative_purpose": ["build_reality"],
            "industry": "medical_technology_and_digital_health",
        },
        "target_summary": "适合用于铺垫，建立真实感。",
        "target_tags_text": "医生 屏幕 诊室 清晨",
    }

    variants = build_fuzzy_understanding_variants(case)

    assert [variant["variant_type"] for variant in variants] == [
        "implicit_stage",
        "fuzzy_style",
        "underspecified_tone",
        "negative_style",
        "director_brief",
    ]
    assert {variant["expected_granularity"] for variant in variants} == {"stage", "purpose", "scene"}
    assert all(variant["case_type"] == "fuzzy_understanding" for variant in variants)
    assert any("别太像汇报片" in variant["user_input"] for variant in variants)


def test_summarize_paraphrase_rows_reports_stage_purpose_and_confidence():
    rows = [
        {
            "target_rank": 1,
            "stage_hit_at_1": True,
            "stage_hit_at_3": True,
            "purpose_hit_at_3": True,
            "confidence": "high",
            "top1_top2_margin": 0.08,
        },
        {
            "target_rank": 4,
            "stage_hit_at_1": False,
            "stage_hit_at_3": True,
            "purpose_hit_at_3": False,
            "confidence": "low",
            "top1_top2_margin": 0.01,
        },
    ]

    summary = summarize_paraphrase_rows(rows)

    assert summary["count"] == 2
    assert summary["target_recall_at_1"] == 0.5
    assert summary["target_recall_at_3"] == 0.5
    assert summary["target_recall_at_10"] == 1.0
    assert summary["stage_hit_at_1"] == 0.5
    assert summary["stage_hit_at_3"] == 1.0
    assert summary["purpose_hit_at_3"] == 0.5
    assert summary["low_confidence_rate"] == 0.5


def test_summarize_fuzzy_rows_reports_scene_stage_purpose_and_planner_stats():
    rows = [
        {
            "target_rank": 1,
            "stage_hit_at_1": True,
            "stage_hit_at_3": True,
            "purpose_hit_at_3": True,
            "confidence": "high",
            "top1_top2_margin": 0.08,
            "top_results": [{"constraint_hits": {}}],
        },
        {
            "target_rank": 12,
            "stage_hit_at_1": False,
            "stage_hit_at_3": True,
            "purpose_hit_at_3": False,
            "confidence": "low",
            "top1_top2_margin": 0.01,
            "top_results": [{"constraint_hits": {"negative_style": ["ad_like"]}}],
        },
    ]

    summary = summarize_fuzzy_rows(rows, {"negative_leak_rate": 0.0, "llm_call_count": 0})

    assert summary["scene_level_recall_at_1"] == 0.5
    assert summary["scene_level_recall_at_10"] == 0.5
    assert summary["stage_level_hit_at_3"] == 1.0
    assert summary["purpose_level_hit_at_3"] == 0.5
    assert summary["style_violation_at_3"] == 0.5
    assert summary["negative_leak_rate"] == 0.0
