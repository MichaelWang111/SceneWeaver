from __future__ import annotations

from sceneweaver.retrieval.service import retrieve_experience_matches
from sceneweaver.schemas import ExperienceCard, ScriptUseCase, TagProfile


def test_opening_query_prefers_opening_usecase_card():
    result = retrieve_experience_matches(
        query_tags=_query_tags(),
        input_text="真实但有力量的开场",
        cards=[
            _card("exp_000001", "team_work", ["show_team"]),
            _card("exp_000002", "opening", ["show_scale"]),
        ],
        top_k=2,
    )

    assert result.results[0].card_id == "exp_000002"
    assert result.results[0].script_stage == "opening"
    assert result.results[0].usecase_score > 0


def test_team_query_prefers_team_work_usecase_card():
    result = retrieve_experience_matches(
        query_tags=_query_tags(),
        input_text="年轻团队协作，不要互联网大厂味",
        cards=[
            _card("exp_000003", "growth", ["show_growth"]),
            _card("exp_000004", "team_work", ["show_team"]),
        ],
        top_k=2,
    )

    assert result.results[0].card_id == "exp_000004"
    assert result.results[0].matched_usecase["script_stage"] == ["team_work"]


def test_human_centered_technology_query_prefers_trust_or_technology_usecase_card():
    result = retrieve_experience_matches(
        query_tags=_query_tags(),
        input_text="科技向善，可信赖",
        cards=[
            _card("exp_000005", "team_work", ["show_team"]),
            _card("exp_000006", "technology_showcase", ["show_technology"]),
            _card("exp_000007", "value_expression", ["build_trust"]),
        ],
        top_k=3,
    )

    assert result.results[0].card_id == "exp_000007"
    assert result.results[0].script_stage == "value_expression"
    assert "build_trust" in result.results[0].creative_purpose


def _card(card_id: str, script_stage: str, creative_purpose: list[str]) -> ExperienceCard:
    scene_id = f"scene_{int(card_id.split('_')[1]):03d}"
    tags = TagProfile(
        symbolic_logic=["general_expression"],
        evidence=[
            {
                "source_id": scene_id,
                "source_type": "scene",
                "field": "test",
                "quote": "test evidence",
            }
        ],
        confidence=0.8,
    )
    return ExperienceCard(
        card_id=card_id,
        source_video_id="video_001",
        source_scene_ids=[scene_id],
        tags=tags,
        keywords=creative_purpose,
        underlying_emotion="test emotion",
        narrative_logic="test narrative",
        director_strategy="test strategy",
        shooting_techniques=["test technique"],
        visual_symbols=[],
        copywriting_tone="test tone",
        avoid=[],
        emotion_temperature_range=(0.3, 0.7),
        reuse_condition="test reuse",
        script_usecase=ScriptUseCase(
            script_stage=script_stage,
            creative_purpose=creative_purpose,
            best_usage=f"use for {script_stage}",
            risk="test risk",
            confidence=0.8,
        ),
        confidence=0.8,
    )


def _query_tags() -> TagProfile:
    return TagProfile(
        symbolic_logic=["general_expression"],
        evidence=[
            {
                "source_id": "query",
                "source_type": "query",
                "field": "input_text",
                "quote": "test query",
            }
        ],
        confidence=0.8,
    )
