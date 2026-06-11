from __future__ import annotations

import json

import mocktesting.mock_retriever as mock_retriever
from mocktesting.constraint_layer import (
    parse_query_constraints,
    profile_with_weights,
    score_constraints,
)
from mocktesting.embedding_cache import EmbeddingCache, cache_key
from mocktesting.embedding_text_builder import (
    build_item_channels,
    build_query_channels,
    load_review_items,
)
from mocktesting.eval_input_generator import generate_eval_inputs
from mocktesting.mock_retriever import build_metrics, search_index, select_best_tuning_result
from mocktesting.mock_retriever import (
    build_paraphrase_variants,
    build_margin_report,
    build_pairwise_pairs,
    case_split,
    filter_cases_by_fixture,
    filter_precomputed_by_fixture,
    possible_overfit,
    rank_items_for_key,
    run_llm_sample_judge,
    select_llm_sample_candidates,
    split_cases,
    summarize_paraphrase_rows,
    summarize_leave_one_fixture,
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
