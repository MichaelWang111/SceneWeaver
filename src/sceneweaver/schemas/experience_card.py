from __future__ import annotations

from pydantic import Field, field_validator

from sceneweaver.schemas.common import EXPERIENCE_ID_RE, SCENE_ID_RE, StrictBaseModel
from sceneweaver.schemas.fingerprint import CreativeFingerprint


class ExperienceCard(StrictBaseModel):
    card_id: str
    source_video_id: str
    source_scene_ids: list[str] = Field(min_length=1)
    fingerprint: CreativeFingerprint
    keywords: list[str] = Field(min_length=1)
    underlying_emotion: str
    narrative_logic: str
    director_strategy: str
    shooting_techniques: list[str] = Field(min_length=1)
    visual_symbols: list[str] = Field(default_factory=list)
    copywriting_tone: str
    avoid: list[str] = Field(default_factory=list)
    emotion_temperature_range: tuple[float, float]
    reuse_condition: str
    confidence: float = Field(ge=0, le=1)

    @field_validator("card_id")
    @classmethod
    def validate_card_id(cls, value: str) -> str:
        if not EXPERIENCE_ID_RE.match(value):
            raise ValueError("card_id must look like exp_000001")
        return value

    @field_validator("source_scene_ids")
    @classmethod
    def validate_scene_ids(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if not SCENE_ID_RE.match(value)]
        if invalid:
            raise ValueError(f"invalid scene ids: {', '.join(invalid)}")
        return values

    @field_validator("emotion_temperature_range")
    @classmethod
    def validate_temperature_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not 0 <= low <= high <= 1:
            raise ValueError("emotion_temperature_range must be within 0..1 and sorted")
        return value
