from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sceneweaver.schemas.common import EXPERIENCE_ID_RE, SCENE_ID_RE, StrictBaseModel
from sceneweaver.schemas.tags import TagProfile

SCRIPT_PURPOSE_RE = re.compile(r"[^0-9A-Za-z]+")

ScriptStage = Literal[
    "opening",
    "setup",
    "character_intro",
    "team_work",
    "growth",
    "technology_showcase",
    "value_expression",
    "ending",
    "transition",
    "general",
]


class ScriptUseCase(StrictBaseModel):
    script_stage: ScriptStage = "general"
    creative_purpose: list[str] = Field(default_factory=lambda: ["general_expression"], min_length=1)
    best_usage: str = "General directing reference."
    risk: str = "Needs additional creative context before direct reuse."
    confidence: float = Field(default=0.35, ge=0, le=1)

    @field_validator("creative_purpose")
    @classmethod
    def normalize_creative_purpose(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            purpose = SCRIPT_PURPOSE_RE.sub("_", value.strip().lower()).strip("_")
            if purpose and purpose not in normalized:
                normalized.append(purpose)
        return normalized or ["general_expression"]


class ExperienceCard(StrictBaseModel):
    card_id: str
    source_video_id: str
    source_scene_ids: list[str] = Field(min_length=1)
    tags: TagProfile
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
    script_usecase: ScriptUseCase = Field(default_factory=ScriptUseCase)
    confidence: float = Field(ge=0, le=1)

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
