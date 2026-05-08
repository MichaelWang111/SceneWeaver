from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from sceneweaver.schemas.common import SCENE_ID_RE, StrictBaseModel, TimeRange


class VisualObservation(StrictBaseModel):
    setting: str
    characters: str
    action_change: str
    composition: str
    lighting: str
    color: str
    camera_motion: str
    confidence_notes: str


class DirectorInterpretation(StrictBaseModel):
    narrative_function: str
    emotional_function: str
    brand_personality_signal: str
    underlying_emotion: str
    audience_projection: str
    shooting_techniques: list[str] = Field(min_length=1)
    why_it_works: str


class ExperienceCandidate(StrictBaseModel):
    keywords: list[str] = Field(min_length=1)
    emotion: str
    narrative_logic: str
    techniques: list[str] = Field(min_length=1)
    reuse_condition: str


class SceneAnalysis(StrictBaseModel):
    scene_id: str
    time_range: TimeRange
    visual_observation: VisualObservation
    director_interpretation: DirectorInterpretation
    experience_candidates: list[ExperienceCandidate] = Field(default_factory=list)
    emotion_temperature: float = Field(ge=0, le=1)

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, value: str) -> str:
        if not SCENE_ID_RE.match(value):
            raise ValueError("scene_id must look like scene_001")
        return value


class ScenesAnalysis(StrictBaseModel):
    video_id: str
    source_url: str
    scene_count: int = Field(ge=0)
    scenes: list[SceneAnalysis]

    @model_validator(mode="after")
    def scene_count_matches(self) -> "ScenesAnalysis":
        if self.scene_count != len(self.scenes):
            raise ValueError("scene_count must match scenes length")
        return self

