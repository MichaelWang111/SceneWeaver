from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from sceneweaver.schemas.common import SCENE_ID_RE, StrictBaseModel
from sceneweaver.schemas.tags import TagEvidence, TagProfile

FingerprintEvidence = TagEvidence
CreativeFingerprint = TagProfile


class SceneFingerprint(StrictBaseModel):
    """Legacy wrapper kept only for reading old fingerprint artifacts."""

    scene_id: str
    source_video_id: str
    fingerprint: TagProfile

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, value: str) -> str:
        if not SCENE_ID_RE.match(value):
            raise ValueError("scene_id must look like scene_001")
        return value


class FilmFingerprint(StrictBaseModel):
    """Legacy wrapper kept only for reading old fingerprint artifacts."""

    video_id: str
    scene_count: int = Field(ge=0)
    fingerprint: TagProfile
    scenes: list[SceneFingerprint]

    @model_validator(mode="after")
    def scene_count_matches(self) -> "FilmFingerprint":
        if self.scene_count != len(self.scenes):
            raise ValueError("scene_count must match scenes length")
        return self
