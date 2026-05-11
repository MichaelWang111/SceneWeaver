from __future__ import annotations

from sceneweaver.analysis.tags import (
    build_film_tags,
    build_query_tags,
    build_scene_tags,
    match_experience_cards,
    retrieve_experience_cards,
)
from sceneweaver.analysis.taxonomy import TagNormalizer
from sceneweaver.pipeline.mock_pipeline import build_mock_artifacts
from sceneweaver.schemas import ExperienceCard, TagProfile


def test_scene_tags_extracts_tags_and_evidence():
    _, scene_analysis, *_ = build_mock_artifacts()

    tags = build_scene_tags(scene_analysis)

    assert tags.evidence[0].source_id == "scene_001"
    assert tags.audience_projection


def test_film_tags_aggregates_scene_tags():
    _, scene_analysis, *_ = build_mock_artifacts()

    film_tags = build_film_tags([scene_analysis])

    assert set(scene_analysis.tags.audience_projection).issubset(set(film_tags.audience_projection))


def test_query_tags_retrieves_direct_address_cards_first():
    query = build_query_tags(
        "世界五百强企业招聘宣传片，表达方式偏向于在各种场景与屏幕后的观众读者面对面对话，稳重可靠，科技向善，关注于人"
    )
    direct_address_card = _card(
        "exp_000001",
        TagProfile(
            emotion_core=["trust", "human_care"],
            audience_projection=["direct_listener", "participant"],
            narrative_function=["establish_trust", "invitation"],
            interaction_mode=["direct_address"],
            visual_motifs=["screen"],
            symbolic_logic=["human_centered_technology", "connection"],
            rhythm_pattern=["calm_direct"],
            evidence=[
                {
                    "source_id": "scene_001",
                    "source_type": "scene",
                    "field": "director_interpretation.audience_projection",
                    "quote": "观众被邀请作为倾听者，与屏幕后的人形成对话。",
                }
            ],
            confidence=0.9,
        ),
    )
    story_art_card = _card(
        "exp_000002",
        TagProfile(
            emotion_core=["ambition", "creativity"],
            audience_projection=["future_builder"],
            narrative_function=["origin_story"],
            interaction_mode=["team_collaboration"],
            visual_motifs=["silhouette", "upward_motion"],
            symbolic_logic=["becoming"],
            rhythm_pattern=["explosive_build"],
            evidence=[
                {
                    "source_id": "scene_002",
                    "source_type": "scene",
                    "field": "experience_card.director_strategy",
                    "quote": "用逆光奔跑表达青年团队突破。",
                }
            ],
            confidence=0.85,
        ),
    )

    results = retrieve_experience_cards(
        query,
        [story_art_card, direct_address_card],
        top_k=2,
    )

    assert [card.card_id for card in results] == ["exp_000001"]


def test_normalizer_maps_brief_aliases_to_canonical_tags():
    tags = build_query_tags(
        "世界五百强招聘宣传片，对屏幕后的观众面对面沟通，稳重可靠，科技向善，关注于人"
    )

    assert "trust" in tags.emotion_core
    assert "human_care" in tags.emotion_core
    assert "direct_listener" in tags.audience_projection
    assert "direct_address" in tags.interaction_mode
    assert "screen" in tags.visual_motifs
    assert "human_centered_technology" in tags.symbolic_logic
    assert "calm_direct" in tags.rhythm_pattern


def test_general_expression_does_not_match_cards():
    normalizer = TagNormalizer()
    query = normalizer.normalize_text(
        "完全没有命中规则的抽象输入",
        evidence=[
            {
                "source_id": "query",
                "source_type": "query",
                "field": "input_text",
                "quote": "完全没有命中规则的抽象输入",
            }
        ],
    )
    card = _card(
        "exp_000003",
        normalizer.normalize_text(
            "另一段也没有命中规则的内容",
            evidence=[
                {
                    "source_id": "scene_003",
                    "source_type": "scene",
                    "field": "experience_card.director_strategy",
                    "quote": "另一段也没有命中规则的内容",
                }
            ],
        ),
    )

    assert retrieve_experience_cards(query, [card]) == []


def test_match_experience_cards_returns_explainable_result():
    query = build_query_tags("稳重可靠，科技向善，对屏幕后的观众面对面对话")
    card = _card(
        "exp_000001",
        TagProfile(
            emotion_core=["trust", "human_care"],
            audience_projection=["direct_listener"],
            narrative_function=["establish_trust"],
            interaction_mode=["direct_address"],
            visual_motifs=["screen"],
            symbolic_logic=["human_centered_technology"],
            rhythm_pattern=["calm_direct"],
            evidence=[
                {
                    "source_id": "scene_001",
                    "source_type": "scene",
                    "field": "experience_card.director_strategy",
                    "quote": "让观众成为被邀请的倾听者",
                }
            ],
            confidence=0.9,
        ),
    )

    matches = match_experience_cards(query, [card])

    assert matches[0].card_id == "exp_000001"
    assert matches[0].score > 0
    assert "direct_address" in matches[0].matched_dimensions["interaction_mode"]
    assert matches[0].evidence[0].source_id == "scene_001"


def _card(card_id: str, tags: TagProfile) -> ExperienceCard:
    return ExperienceCard(
        card_id=card_id,
        source_video_id="bilibili_BVxxxx",
        source_scene_ids=[tags.evidence[0].source_id],
        tags=tags,
        keywords=["科技向善"],
        underlying_emotion="被看见与被信任",
        narrative_logic="用直接对话消解大企业距离感",
        director_strategy="让观众成为被邀请的倾听者",
        shooting_techniques=["直视镜头"],
        visual_symbols=["屏幕"],
        copywriting_tone="克制、平等、直接",
        avoid=["空泛口号"],
        emotion_temperature_range=(0.35, 0.65),
        reuse_condition="适合招聘宣传片",
        confidence=0.86,
    )
