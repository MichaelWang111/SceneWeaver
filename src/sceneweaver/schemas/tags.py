from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sceneweaver.schemas.common import SCENE_ID_RE, StrictBaseModel

TAG_RE = re.compile(r"[^0-9A-Za-z]+")


class TagEvidence(StrictBaseModel):
    source_id: str = Field(min_length=1)
    source_type: Literal["scene", "film", "card", "query"]
    field: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    note: str = ""

    @model_validator(mode="after")
    def validate_scene_source_id(self) -> "TagEvidence":
        if self.source_type == "scene" and not SCENE_ID_RE.match(self.source_id):
            raise ValueError("scene evidence source_id must look like scene_001")
        return self


class TagCandidate(StrictBaseModel):
    candidate: str = Field(min_length=1)
    normalized: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: Literal["scene", "film", "card", "query"]
    field: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    status: Literal["pending", "merged", "rejected"] = "pending"
    suggested_action: Literal["review", "merge_alias", "new_canonical", "reject"] = "review"
    note: str = ""


class TagProfile(StrictBaseModel):
    emotion_core: list[str] = Field(default_factory=list)
    audience_projection: list[str] = Field(default_factory=list)
    narrative_function: list[str] = Field(default_factory=list)
    interaction_mode: list[str] = Field(default_factory=list)
    visual_motifs: list[str] = Field(default_factory=list)
    symbolic_logic: list[str] = Field(default_factory=list)
    rhythm_pattern: list[str] = Field(default_factory=list)
    custom_tags: list[str] = Field(default_factory=list)
    evidence: list[TagEvidence] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator(
        "emotion_core",
        "audience_projection",
        "narrative_function",
        "interaction_mode",
        "visual_motifs",
        "symbolic_logic",
        "rhythm_pattern",
        "custom_tags",
    )
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            tag = TAG_RE.sub("_", value.strip().lower()).strip("_")
            if tag and tag not in normalized:
                normalized.append(tag)
        return normalized

    @model_validator(mode="after")
    def require_at_least_one_tag(self) -> "TagProfile":
        if not any(
            [
                self.emotion_core,
                self.audience_projection,
                self.narrative_function,
                self.interaction_mode,
                self.visual_motifs,
                self.symbolic_logic,
                self.rhythm_pattern,
                self.custom_tags,
            ]
        ):
            raise ValueError("tag profile must contain at least one tag")
        return self
