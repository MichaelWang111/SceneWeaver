from __future__ import annotations

from sceneweaver.pipeline.mock_pipeline import run_mock_pipeline
from sceneweaver.schemas import ExperienceCard, FilmAnalysis, SceneAnalysis, ScenePackage, ScenesAnalysis
from sceneweaver.storage.json_store import read_json, read_jsonl


def test_mock_pipeline_writes_valid_artifacts(tmp_path):
    output_dir = run_mock_pipeline(tmp_path / "mock_video")

    assert read_json(output_dir / "packages" / "scene_001.json", ScenePackage)
    assert read_json(output_dir / "analysis" / "scene_001.json", SceneAnalysis)
    assert read_json(output_dir / "analysis" / "scenes.json", ScenesAnalysis)
    assert read_json(output_dir / "analysis" / "film_analysis.json", FilmAnalysis)

    cards = read_jsonl(output_dir / "analysis" / "experience_cards.jsonl", ExperienceCard)
    assert len(cards) == 1
    assert cards[0].card_id == "exp_000001"

