from __future__ import annotations

import json

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
