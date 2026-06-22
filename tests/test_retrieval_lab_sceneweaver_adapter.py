from __future__ import annotations

import json
from types import SimpleNamespace

from retrieval_lab.corpora.sceneweaver import sceneweaver_items_from_sources
from retrieval_lab.indexes.service import build_index_manifest
from retrieval_lab.planners.service import rule_plan
from retrieval_lab.retrieval.commands import retrieval_run_command
from retrieval_lab.retrieval.service import query_tokens_for_scoring, retrieval_run


def test_sceneweaver_cards_convert_to_multichannel_items(tmp_path):
    cards = write_cards(tmp_path)

    items = sceneweaver_items_from_sources([cards], channel_policy="summary_tags")

    assert len(items) == 2
    first = items[0]
    assert first["item_id"] == "video_001::scene_001::exp_000001"
    assert set(first["channels"]) >= {"summary", "script_use", "experience", "visual_tags", "tags", "combined"}
    assert "Grounded setup" in first["channels"]["summary"]
    assert "doctor" in first["text"]
    assert first["metadata"]["script_stage"] == "setup"
    assert first["payload"]["director_strategy"] == "Stay close to the doctor while pressure builds."


def test_sceneweaver_cards_build_index_manifest(tmp_path):
    cards = write_cards(tmp_path)

    manifest = build_index_manifest(card_sources=[cards], channel_policy="experience", index_id="cards_idx")

    assert manifest["index_id"] == "cards_idx"
    assert manifest["source_dataset_id"] == "sceneweaver_experience_cards"
    assert manifest["item_count"] == 2
    assert manifest["channel_policy"] == "experience"
    assert manifest["video_counts"] == {"video_001": 2}


def test_sceneweaver_retrieval_returns_payload_and_channel_scores(tmp_path):
    cards = write_cards(tmp_path)

    artifact = retrieval_run(
        card_sources=[cards],
        queries=["need grounded setup doctor pressure"],
        planner="multi_query",
        planner_cache=None,
        ranking_key="script_use_only",
        channel_policy="all",
        top_k=2,
    )
    row = next(iter(artifact["run_rows"].values()))[0]
    top = row["top_results"][0]

    assert artifact["method"] == "retrieval_lab_native_sceneweaver_retrieval_run"
    assert top["item_id"] == "video_001::scene_001::exp_000001"
    assert top["payload"]["reuse_condition"] == "Use when setup needs human pressure before capability appears."
    assert top["channel_scores"]["script_use"] > 0
    assert "Grounded setup" in top["channels"]["summary"]


def test_sceneweaver_retrieval_cli_accepts_cards_and_query(tmp_path):
    cards = write_cards(tmp_path)
    output = tmp_path / "run.json"
    args = SimpleNamespace(
        dataset=tmp_path / "unused.json",
        split="test.md",
        limit=0,
        planner="multi_query",
        planner_cache=None,
        top_k=1,
        candidate_depth=10,
        run_name="cards_run",
        ranking_key="hybrid_rrf_constraints_signature",
        cards=[cards],
        query=["technology capability interface"],
        query_file=None,
        channel_policy="all",
        output=output,
    )

    summary = retrieval_run_command(args)["summary"]
    artifact = json.loads(output.read_text(encoding="utf-8"))
    row = artifact["run_rows"]["cards_run"][0]

    assert summary["index_item_count"] == 2
    assert row["top_results"][0]["item_id"] == "video_001::scene_002::exp_000002"
    assert row["top_results"][0]["metadata"]["script_stage"] == "technology_showcase"


def test_scoring_query_keeps_technology_as_positive_token():
    plan = rule_plan("technology capability interface", {})
    tokens = set(query_tokens_for_scoring(plan))

    assert plan["positive_query"] == "technology capability interface"
    assert plan["negative_constraints"] == []
    assert {"technology", "capability", "interface"} <= tokens


def write_cards(tmp_path):
    path = tmp_path / "analysis" / "experience_cards.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "card_id": "exp_000001",
            "source_video_id": "video_001",
            "source_scene_ids": ["scene_001"],
            "tags": tag_profile(symbolic_logic=["build_reality"], interaction_mode=["observational"], visual_motifs=["doctor"]),
            "keywords": ["doctor", "pressure", "setup"],
            "underlying_emotion": "calm pressure",
            "narrative_logic": "A doctor checks a case before the solution appears.",
            "director_strategy": "Stay close to the doctor while pressure builds.",
            "shooting_techniques": ["close observation"],
            "visual_symbols": ["hospital", "screen"],
            "style_traits": ["documentary", "human_warmth", "real_location"],
            "style_risks": [],
            "copywriting_tone": "restrained and human",
            "avoid": [],
            "emotion_temperature_range": [0.3, 0.6],
            "reuse_condition": "Use when setup needs human pressure before capability appears.",
            "script_usecase": {
                "script_stage": "setup",
                "creative_purpose": ["build_reality"],
                "best_usage": "Grounded setup for a doctor under real location pressure.",
                "risk": "Can feel slow without later payoff.",
                "confidence": 0.9,
            },
            "confidence": 0.85,
        },
        {
            "card_id": "exp_000002",
            "source_video_id": "video_001",
            "source_scene_ids": ["scene_002"],
            "tags": tag_profile(symbolic_logic=["show_technology"], interaction_mode=["screen"], visual_motifs=["interface"]),
            "keywords": ["technology", "interface", "capability"],
            "underlying_emotion": "clarity",
            "narrative_logic": "The interface reduces uncertainty.",
            "director_strategy": "Show technology as a tool used by people.",
            "shooting_techniques": ["screen insert"],
            "visual_symbols": ["dashboard", "confirmation"],
            "style_traits": [],
            "style_risks": ["tech_showoff"],
            "copywriting_tone": "precise",
            "avoid": ["cold product pitch"],
            "emotion_temperature_range": [0.2, 0.5],
            "reuse_condition": "Use when technology capability needs to enter naturally.",
            "script_usecase": {
                "script_stage": "technology_showcase",
                "creative_purpose": ["show_technology"],
                "best_usage": "Technology capability moment with interface evidence.",
                "risk": "May become feature demo if people disappear.",
                "confidence": 0.88,
            },
            "confidence": 0.8,
        },
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def tag_profile(**values):
    data = {
        "emotion_core": [],
        "audience_projection": [],
        "narrative_function": [],
        "interaction_mode": [],
        "visual_motifs": [],
        "symbolic_logic": [],
        "rhythm_pattern": [],
        "custom_tags": [],
        "evidence": [
            {
                "source_id": "scene_001",
                "source_type": "scene",
                "field": "test.md",
                "quote": "test.md evidence",
            }
        ],
        "confidence": 0.8,
    }
    data.update(values)
    return data
