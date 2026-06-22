from __future__ import annotations

from sceneweaver.retrieval.models import QueryUseCase
from sceneweaver.retrieval.policy import score_experience_match
from sceneweaver.retrieval.query_plan import build_query_plan
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


def test_forbidden_stage_is_filtered_and_desired_stage_is_explained():
    result = retrieve_experience_matches(
        query_tags=_query_tags(),
        input_text="画面可以借用办公室、团队这类相似元素，但不要做成开场。我真正要的是铺垫。",
        cards=[
            _card("exp_000008", "opening", ["show_scale"]),
            _card("exp_000009", "setup", ["establish_context"]),
        ],
        top_k=2,
    )

    assert [match.card_id for match in result.results] == ["exp_000009"]
    assert result.results[0].constraint_score > 0
    assert result.results[0].constraint_hits == {"desired_stage": ["setup"]}


def test_forbidden_stage_can_be_soft_penalized_with_constraint_hits():
    result = retrieve_experience_matches(
        query_tags=_query_tags(),
        input_text="不要做成开场，我真正要的是铺垫",
        cards=[
            _card("exp_000012", "opening", ["show_scale"]),
            _card("exp_000013", "setup", ["establish_context"]),
        ],
        top_k=2,
        hard_filter_forbidden_stage=False,
        semantic_scores=[0.8, 0.8],
        semantic_weight=10.0,
    )

    by_id = {match.card_id: match for match in result.results}
    assert by_id["exp_000012"].constraint_score < 0
    assert by_id["exp_000012"].constraint_hits == {"forbidden_stage": ["opening"]}
    assert result.results[0].card_id == "exp_000013"


def test_hybrid_rrf_constraints_can_use_lexical_signal_and_style_penalty():
    result = retrieve_experience_matches(
        query_tags=_query_tags(),
        input_text="要有人味、像纪录片、真实现场，不要大厂味",
        cards=[
            _card(
                "exp_000014",
                "setup",
                ["establish_context"],
                keywords=["互联网大厂", "办公", "口号"],
                style_risks=["big_company_office"],
            ),
            _card(
                "exp_000015",
                "setup",
                ["establish_context"],
                keywords=["纪录片", "真实现场", "人的温度"],
                style_traits=["documentary", "human_warmth", "real_location"],
            ),
        ],
        top_k=2,
        retrieval_workflow="hybrid_rrf_constraints",
        lexical_weight=2.0,
    )

    assert result.results[0].card_id == "exp_000015"
    assert result.results[0].lexical_score is not None
    assert result.results[0].rrf_score > 0


def test_style_penalty_exposes_negative_style_hits():
    match = score_experience_match(
        query_tags=_query_tags(),
        query_usecase=QueryUseCase(script_stage="setup", creative_purpose=["establish_context"]),
        card=_card(
            "exp_000016",
            "setup",
            ["establish_context"],
            keywords=["互联网大厂", "办公", "口号"],
            style_risks=["big_company_office"],
        ),
        query_plan=build_query_plan("要有人味，不要大厂味"),
    )

    assert match.constraint_score < 0
    assert match.constraint_hits["negative_style"] == ["big_company_office"]
    assert match.constraint_hits["negative_constraints"] == ["大厂味"]


def _card(
    card_id: str,
    script_stage: str,
    creative_purpose: list[str],
    *,
    keywords: list[str] | None = None,
    style_traits: list[str] | None = None,
    style_risks: list[str] | None = None,
) -> ExperienceCard:
    scene_id = f"scene_{int(card_id.split('_')[1]):03d}"
    tags = TagProfile(
        symbolic_logic=["general_expression"],
        evidence=[
            {
                "source_id": scene_id,
                "source_type": "scene",
                "field": "test.md",
                "quote": "test.md evidence",
            }
        ],
        confidence=0.8,
    )
    return ExperienceCard(
        card_id=card_id,
        source_video_id="video_001",
        source_scene_ids=[scene_id],
        tags=tags,
        keywords=keywords or creative_purpose,
        underlying_emotion="test.md emotion",
        narrative_logic="test.md narrative",
        director_strategy="test.md strategy",
        shooting_techniques=["test.md technique"],
        visual_symbols=[],
        style_traits=style_traits or [],
        style_risks=style_risks or [],
        copywriting_tone="test.md tone",
        avoid=[],
        emotion_temperature_range=(0.3, 0.7),
        reuse_condition="test.md reuse",
        script_usecase=ScriptUseCase(
            script_stage=script_stage,
            creative_purpose=creative_purpose,
            best_usage=f"use for {script_stage}",
            risk="test.md risk",
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
                "quote": "test.md query",
            }
        ],
        confidence=0.8,
    )
