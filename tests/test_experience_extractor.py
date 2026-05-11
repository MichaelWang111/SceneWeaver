from __future__ import annotations

from sceneweaver.analysis.experience_extractor import build_experience_cards, extract_experience_cards
from sceneweaver.pipeline.mock_pipeline import build_mock_artifacts
from sceneweaver.schemas import ExperienceCard
from sceneweaver.storage.json_store import read_jsonl, write_json


def test_build_experience_cards_from_scene_candidates():
    _, _, scenes, *_ = build_mock_artifacts()

    cards = build_experience_cards(scenes)

    assert len(cards) == 1
    assert cards[0].card_id == "exp_000001"
    assert cards[0].source_scene_ids == ["scene_001"]
    assert cards[0].tags.evidence[0].source_id == "scene_001"


def test_extract_experience_cards_writes_jsonl(tmp_path):
    _, _, scenes, *_ = build_mock_artifacts()
    output_dir = tmp_path / "video"
    write_json(output_dir / "analysis" / "scenes.json", scenes)

    cards = extract_experience_cards(output_dir)

    assert len(cards) == 1
    written = read_jsonl(output_dir / "analysis" / "experience_cards.jsonl", ExperienceCard)
    assert written[0].card_id == "exp_000001"
    assert written[0].tags.evidence[0].source_id == "scene_001"
