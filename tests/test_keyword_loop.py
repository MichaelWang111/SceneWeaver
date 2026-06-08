from __future__ import annotations

import json

import pytest

from sceneweaver.analysis.keyword_loop import (
    discover_experience_card_paths,
    discover_unindexed_scene_dirs,
    run_keyword_loop,
)
from sceneweaver.schemas import ExperienceCard, TagProfile
from sceneweaver.storage.json_store import write_jsonl


class FakeTextClient:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.calls = []

    def analyze_text_json(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.data


class FakeEmbeddingBackend:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "semantic target" in text:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_keyword_loop_expands_tags_logs_candidates_and_matches_cards(tmp_path):
    output_dir = tmp_path / "video"
    cards_path = output_dir / "analysis" / "experience_cards.jsonl"
    write_jsonl(cards_path, [_experience_card()])
    client = FakeTextClient(_association_data("trust direct_address screen obsidian ritual"))

    result = run_keyword_loop(
        "trust direct_address screen obsidian ritual",
        output_dir,
        client=client,
        top_k=1,
        max_items=8,
    )

    assert result.association_analysis.input_text == "trust direct_address screen obsidian ritual"
    assert result.retrieval.results[0].card_id == "exp_000001"
    assert result.matched_card_count == 1
    assert result.top_matches[0].card_id == "exp_000001"
    assert result.top_matches[0].source_video_id == "video_001"
    assert "direct_address" in result.retrieval.results[0].matched_dimensions["interaction_mode"]
    assert client.calls[0]["max_tokens"] is not None
    assert result.association_path.endswith("_association.json")
    assert "tag_candidates.jsonl" in result.candidate_log_path
    candidate_rows = [
        json.loads(line)
        for line in (output_dir / "analysis" / "tag_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "obsidian" in {row["normalized"] for row in candidate_rows}


def test_keyword_loop_searches_all_cards_in_collection_directory(tmp_path):
    collection_dir = tmp_path / "film_analysis"
    film_a_cards = collection_dir / "film_a" / "analysis" / "experience_cards.jsonl"
    film_b_cards = collection_dir / "film_b" / "analysis" / "experience_cards.jsonl"
    write_jsonl(film_a_cards, [_experience_card(card_id="exp_000001", source_video_id="film_a")])
    write_jsonl(
        film_b_cards,
        [
            _experience_card(
                card_id="exp_000002",
                source_video_id="film_b",
                tags=TagProfile(
                    emotion_core=["ambition"],
                    audience_projection=["future_builder"],
                    narrative_function=["invitation"],
                    interaction_mode=["team_collaboration"],
                    visual_motifs=["silhouette"],
                    symbolic_logic=["becoming"],
                    rhythm_pattern=["explosive_build"],
                    evidence=[
                        {
                            "source_id": "scene_002",
                            "source_type": "scene",
                            "field": "input_text",
                            "quote": "team ambition",
                        }
                    ],
                    confidence=0.8,
                ),
            )
        ],
    )

    result = run_keyword_loop(
        "trust direct_address screen obsidian ritual",
        collection_dir,
        client=FakeTextClient(_association_data("trust direct_address screen obsidian ritual")),
        top_k=2,
        max_items=8,
    )

    assert len(result.experience_cards_paths) == 2
    assert result.searched_card_count == 2
    assert result.matched_card_count == 1
    assert result.top_matches[0].source_video_id == "film_a"
    assert result.retrieval.results[0].card.source_video_id == "film_a"
    assert (collection_dir / "keyword_loops" / "tag_candidates.jsonl").exists()
    assert discover_experience_card_paths(collection_dir) == [film_a_cards.resolve(), film_b_cards.resolve()]


def test_keyword_loop_semantic_rerank_can_promote_semantic_match(tmp_path):
    output_dir = tmp_path / "video"
    cards_path = output_dir / "analysis" / "experience_cards.jsonl"
    write_jsonl(
        cards_path,
        [
            _experience_card(card_id="exp_000001", keywords=["trust", "screen"]),
            _experience_card(
                card_id="exp_000002",
                source_video_id="semantic_film",
                tags=TagProfile(
                    emotion_core=["ambition"],
                    audience_projection=["future_builder"],
                    narrative_function=["invitation"],
                    interaction_mode=["team_collaboration"],
                    visual_motifs=["silhouette"],
                    symbolic_logic=["becoming"],
                    rhythm_pattern=["explosive_build"],
                    evidence=[
                        {
                            "source_id": "scene_002",
                            "source_type": "scene",
                            "field": "input_text",
                            "quote": "semantic target",
                        }
                    ],
                    confidence=0.8,
                ),
                keywords=["semantic target"],
            ),
        ],
    )

    result = run_keyword_loop(
        "semantic target",
        output_dir,
        client=FakeTextClient(_association_data("semantic target")),
        top_k=2,
        max_items=8,
        semantic=True,
        semantic_weight=20,
        embedding_backend=FakeEmbeddingBackend(),
    )

    assert result.semantic_enabled is True
    assert result.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert result.top_matches[0].card_id == "exp_000002"
    assert result.top_matches[0].semantic_score == 1.0
    assert result.top_matches[0].tag_score == 0


def test_keyword_loop_just_tags_uses_lightweight_tag_expansion(tmp_path):
    output_dir = tmp_path / "video"
    cards_path = output_dir / "analysis" / "experience_cards.jsonl"
    write_jsonl(cards_path, [_experience_card()])
    client = FakeTextClient(
        {
            "expanded_terms": [
                "trust",
                "direct_address",
                "screen",
                "human_centered_technology",
            ],
            "tag_hints": {
                "emotion_core": ["trust"],
                "audience_projection": ["direct_listener"],
                "narrative_function": ["establish_trust"],
                "interaction_mode": ["direct_address"],
                "visual_motifs": ["screen"],
                "symbolic_logic": ["human_centered_technology"],
                "rhythm_pattern": ["calm_direct"],
            },
            "avoid_terms": ["generic big company"],
        }
    )

    result = run_keyword_loop(
        "trust screen",
        output_dir,
        client=client,
        top_k=1,
        just_tags=True,
    )

    assert result.mode == "just-tags"
    assert result.association_analysis is None
    assert result.tag_expansion_analysis is not None
    assert result.tag_expansion_analysis.expanded_terms[0] == "trust"
    assert result.retrieval.results[0].card_id == "exp_000001"


def test_keyword_loop_intent_promotes_core_intent_match(tmp_path):
    output_dir = tmp_path / "video"
    cards_path = output_dir / "analysis" / "experience_cards.jsonl"
    write_jsonl(
        cards_path,
            [
                _experience_card(
                    card_id="exp_000010",
                source_video_id="visual_film",
                keywords=["technology", "city", "scale"],
                tags=_query_tags("trust direct_address screen human_centered_technology calm_direct"),
            ),
                _experience_card(
                    card_id="exp_000011",
                source_video_id="tencent_like",
                keywords=["科技向善", "人机共生", "面对面交流"],
                tags=TagProfile(
                    emotion_core=["human_care"],
                    audience_projection=["future_builder"],
                    narrative_function=["invitation"],
                    interaction_mode=["direct_address"],
                    visual_motifs=["screen"],
                    symbolic_logic=["human_centered_technology"],
                    rhythm_pattern=["calm_direct"],
                    evidence=[
                        {
                            "source_id": "scene_002",
                            "source_type": "scene",
                            "field": "input_text",
                            "quote": "科技向善 人机共生 面对面交流",
                        }
                    ],
                    confidence=0.8,
                ),
            ),
        ],
    )
    client = FakeTextClient(
        {
            "primary_intent": "寻找科技公司招聘宣传片中以科技向善和面对面沟通建立认同的经验卡",
            "must_match": ["科技向善", "人机共生", "面对面交流"],
            "nice_to_have": ["招聘", "发挥潜力"],
            "avoid": ["纯科技视觉", "赛博朋克城市"],
            "intent_keywords": ["human_centered_technology", "direct_address"],
            "target_audience": ["潜在求职者"],
            "selection_criteria": ["technology serves people"],
        }
    )

    result = run_keyword_loop(
        "成熟大型商业与科技公司招聘宣传片 科技向善 面对面交流",
        output_dir,
        client=client,
        top_k=2,
        intent=True,
        intent_weight=3,
    )

    assert result.mode == "intent"
    assert result.intent_analysis is not None
    assert result.association_analysis is None
    assert result.tag_expansion_analysis is None
    assert result.top_matches[0].card_id == "exp_000011"
    assert result.top_matches[0].intent_score > 0


def test_keyword_loop_requires_experience_cards(tmp_path):
    with pytest.raises(FileNotFoundError, match="experience cards not found"):
        run_keyword_loop("trust", tmp_path / "video", client=FakeTextClient(_association_data("trust")))


def test_keyword_loop_reports_scene_dirs_without_experience_cards(tmp_path):
    collection_dir = tmp_path / "film_analysis"
    indexed_cards = collection_dir / "film_a" / "analysis" / "experience_cards.jsonl"
    unindexed_scenes = collection_dir / "film_b" / "analysis" / "scenes.json"
    write_jsonl(indexed_cards, [_experience_card()])
    unindexed_scenes.parent.mkdir(parents=True, exist_ok=True)
    unindexed_scenes.write_text('{"video_id":"film_b","scene_count":0,"scenes":[]}', encoding="utf-8")

    result = run_keyword_loop(
        "trust direct_address",
        collection_dir,
        client=FakeTextClient(_association_data("trust direct_address")),
        max_items=8,
    )

    assert discover_unindexed_scene_dirs(collection_dir) == [unindexed_scenes.parent.resolve()]
    assert result.unindexed_scene_dirs == [str(unindexed_scenes.parent.resolve())]
    assert "Experience cards missing" in result.next_actions[0]


def _association_data(input_text: str) -> dict:
    categories = [
        "visual_imagery",
        "character_state",
        "action_motifs",
        "emotional_keywords",
        "narrative_seeds",
        "spatial_symbols",
        "light_color_texture",
        "copy_tone",
    ]
    return {
        "input_text": input_text,
        "query_tags": _query_tags(input_text).model_dump(mode="json"),
        "core_reading": "trust direct_address screen human_centered_technology calm_direct obsidian",
        "emotional_arc": {
            "origin": "trust",
            "development": "direct_address",
            "release": "screen",
            "arc_summary": "calm_direct",
        },
        "association_count": len(categories),
        "association_map": {category: [_association_item(category)] for category in categories},
        "director_possibilities": [
            {
                "name": "direct trust",
                "concept": "screen based direct address",
                "emotional_direction": "trust",
                "visual_direction": "screen",
                "narrative_direction": "establish trust",
            },
            {
                "name": "human tech",
                "concept": "technology serves people",
                "emotional_direction": "human care",
                "visual_direction": "interface",
                "narrative_direction": "connection",
            },
            {
                "name": "calm proof",
                "concept": "quiet proof through direct address",
                "emotional_direction": "calm",
                "visual_direction": "locked camera",
                "narrative_direction": "credibility",
            },
        ],
        "avoid_cliches": ["empty slogans"],
    }


def _association_item(category: str) -> dict:
    return {
        "term": category,
        "category": category,
        "meaning": f"{category} meaning",
        "emotion": "trust",
        "image_hint": "screen direct address",
        "usage_hint": "match direct address cards",
    }


def _experience_card(
    *,
    card_id: str = "exp_000001",
    source_video_id: str = "video_001",
    tags: TagProfile | None = None,
    keywords: list[str] | None = None,
) -> ExperienceCard:
    tags = tags or _query_tags("trust direct_address screen human_centered_technology calm_direct")
    keywords = keywords or ["trust", "screen"]
    return ExperienceCard(
        card_id=card_id,
        source_video_id=source_video_id,
        source_scene_ids=["scene_001"],
        tags=tags,
        keywords=keywords,
        underlying_emotion="trust",
        narrative_logic="establish trust",
        director_strategy="direct_address through screen",
        shooting_techniques=["direct_address"],
        visual_symbols=["screen"],
        copywriting_tone="calm",
        avoid=["empty slogans"],
        emotion_temperature_range=(0.3, 0.6),
        reuse_condition="brand trust briefs",
        confidence=0.9,
    )


def _query_tags(text: str) -> TagProfile:
    return TagProfile(
        emotion_core=["trust"],
        audience_projection=["direct_listener"],
        narrative_function=["establish_trust"],
        interaction_mode=["direct_address"],
        visual_motifs=["screen"],
        symbolic_logic=["human_centered_technology"],
        rhythm_pattern=["calm_direct"],
        evidence=[
            {
                "source_id": "query",
                "source_type": "query",
                "field": "input_text",
                "quote": text,
            }
        ],
        confidence=0.9,
    )
