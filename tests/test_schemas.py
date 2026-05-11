from __future__ import annotations

from pathlib import Path

import pytest

from sceneweaver.schemas import AssociationAnalysis, ExperienceCard, FilmAnalysis, SceneAnalysis, ScenePackage, TagProfile
from sceneweaver.schemas.common import TimeRange
from sceneweaver.storage.json_store import read_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "model"),
    [
        ("examples/associate.json", AssociationAnalysis),
        ("examples/scene_package.json", ScenePackage),
        ("examples/scene_analysis.json", SceneAnalysis),
        ("examples/film_analysis.json", FilmAnalysis),
        ("examples/experience_card.json", ExperienceCard),
    ],
)
def test_example_json_validates(path: str, model):
    artifact = read_json(ROOT / path, model)
    assert isinstance(artifact, model)


def test_time_range_requires_millisecond_format():
    with pytest.raises(ValueError, match="HH:MM:SS.mmm"):
        TimeRange(start="00:00:03,200", end="00:00:07.800")


def test_experience_temperature_range_must_be_sorted():
    data = read_json(ROOT / "examples/experience_card.json", ExperienceCard).model_dump(mode="json")
    data["emotion_temperature_range"] = [0.9, 0.2]
    with pytest.raises(ValueError, match="emotion_temperature_range"):
        ExperienceCard.model_validate(data)


def test_tag_profile_requires_evidence_and_tags():
    with pytest.raises(ValueError, match="evidence"):
        TagProfile(
            emotion_core=["trust"],
            evidence=[],
            confidence=0.8,
        )

    with pytest.raises(ValueError, match="at least one tag"):
        TagProfile(
            evidence=[
                {
                    "source_id": "scene_001",
                    "source_type": "scene",
                    "field": "director_interpretation.why_it_works",
                    "quote": "有效",
                }
            ],
            confidence=0.8,
        )
