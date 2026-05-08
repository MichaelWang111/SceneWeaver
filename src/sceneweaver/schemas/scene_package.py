from __future__ import annotations

from pydantic import Field, field_validator

from sceneweaver.schemas.common import SCENE_ID_RE, StrictBaseModel, SubtitleSegment, TimeRange


class FrameSet(StrictBaseModel):
    start: str
    middle: str
    end: str


class SceneMetadata(StrictBaseModel):
    scene_index: int = Field(ge=1)
    source_url: str
    language: str = "zh-CN"


class ScenePackage(StrictBaseModel):
    scene_id: str
    source_video_id: str
    time_range: TimeRange
    frames: FrameSet
    subtitle_segment: SubtitleSegment = Field(default_factory=SubtitleSegment)
    metadata: SceneMetadata

    @field_validator("scene_id")
    @classmethod
    def validate_scene_id(cls, value: str) -> str:
        if not SCENE_ID_RE.match(value):
            raise ValueError("scene_id must look like scene_001")
        return value

