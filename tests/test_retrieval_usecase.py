from __future__ import annotations

from sceneweaver.retrieval.usecase import build_script_usecase
from sceneweaver.schemas import TagProfile


def test_team_collaboration_card_infers_team_work_usecase():
    usecase = build_script_usecase(
        _tags(
            emotion_core=["belonging"],
            narrative_function=["culture_showcase"],
            interaction_mode=["team_collaboration"],
            symbolic_logic=["connection"],
        )
    )

    assert usecase.script_stage == "team_work"
    assert "show_team" in usecase.creative_purpose


def test_human_centered_technology_card_infers_technology_or_trust_usecase():
    usecase = build_script_usecase(
        _tags(
            emotion_core=["human_care", "trust"],
            narrative_function=["establish_trust"],
            interaction_mode=["direct_address"],
            visual_motifs=["screen"],
            symbolic_logic=["human_centered_technology"],
        )
    )

    assert usecase.script_stage in {"technology_showcase", "value_expression"}
    assert {"show_technology", "build_trust"} & set(usecase.creative_purpose)


def test_invitation_card_infers_ending_usecase():
    usecase = build_script_usecase(
        _tags(
            narrative_function=["invitation"],
        ),
        text="join us call to action",
    )

    assert usecase.script_stage == "ending"
    assert "attract_talent" in usecase.creative_purpose


def _tags(
    *,
    emotion_core: list[str] | None = None,
    audience_projection: list[str] | None = None,
    narrative_function: list[str] | None = None,
    interaction_mode: list[str] | None = None,
    visual_motifs: list[str] | None = None,
    symbolic_logic: list[str] | None = None,
    rhythm_pattern: list[str] | None = None,
) -> TagProfile:
    return TagProfile(
        emotion_core=emotion_core or [],
        audience_projection=audience_projection or [],
        narrative_function=narrative_function or [],
        interaction_mode=interaction_mode or [],
        visual_motifs=visual_motifs or [],
        symbolic_logic=symbolic_logic or ["general_expression"],
        rhythm_pattern=rhythm_pattern or [],
        evidence=[
            {
                "source_id": "scene_001",
                "source_type": "scene",
                "field": "test",
                "quote": "test evidence",
            }
        ],
        confidence=0.8,
    )
