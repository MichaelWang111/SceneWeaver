from __future__ import annotations

from pydantic import Field

from sceneweaver.schemas.common import StrictBaseModel


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
    atmosphere: str
    tone: str
    rhythm: Rhythm
    emotional_curve: list[EmotionalCurvePoint] = Field(min_length=1)
    visual_language: VisualLanguage
    narrative_structure: str
    brand_personality: list[str] = Field(min_length=1)
    audience_projection: str
    director_language_summary: str

