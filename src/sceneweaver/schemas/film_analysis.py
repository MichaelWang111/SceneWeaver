from __future__ import annotations

from pydantic import Field, model_validator

from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.schemas.tags import TagProfile


class Rhythm(StrictBaseModel):
    overall: str
    description: str


class EmotionalCurvePoint(StrictBaseModel):
    phase: str
    emotion: str
    function: str


class VisualLanguage(StrictBaseModel):
    camera: list[str] = Field(default_factory=list)
    lighting: list[str] = Field(default_factory=list)
    symbolism: list[str] = Field(default_factory=list)


class FilmAnalysis(StrictBaseModel):
    video_id: str
    tags: TagProfile
    atmosphere: str
    tone: str
    rhythm: Rhythm
    emotional_curve: list[EmotionalCurvePoint] = Field(min_length=1)
    visual_language: VisualLanguage
    narrative_structure: str
    brand_personality: list[str] = Field(min_length=1)
    audience_projection: str
    director_language_summary: str

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fingerprint(cls, data):
        if isinstance(data, dict) and "tags" not in data and "fingerprint" in data:
            data = dict(data)
            data["tags"] = data.pop("fingerprint")
        return data

    @property
    def fingerprint(self) -> TagProfile:
        return self.tags
