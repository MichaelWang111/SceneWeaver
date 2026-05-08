from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}$")
SCENE_ID_RE = re.compile(r"^scene_\d{3,}$")
EXPERIENCE_ID_RE = re.compile(r"^exp_\d{6,}$")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TimeRange(StrictBaseModel):
    start: str
    end: str
    duration_seconds: float | None = Field(default=None, ge=0)

    @field_validator("start", "end")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not TIME_RE.match(value):
            raise ValueError("time must use HH:MM:SS.mmm")
        return value


class SubtitleItem(StrictBaseModel):
    start: str
    end: str
    text: str

    @field_validator("start", "end")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not TIME_RE.match(value):
            raise ValueError("time must use HH:MM:SS.mmm")
        return value


class SubtitleSegment(StrictBaseModel):
    text: str = ""
    items: list[SubtitleItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_text_matches_items(self) -> "SubtitleSegment":
        if not self.text and self.items:
            self.text = " ".join(item.text for item in self.items)
        return self

