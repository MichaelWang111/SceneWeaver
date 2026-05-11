from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.schemas.tags import TagProfile


class AssociationItem(StrictBaseModel):
    term: str = Field(min_length=1)
    category: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    emotion: str = Field(min_length=1)
    image_hint: str = Field(min_length=1)
    usage_hint: str = Field(min_length=1)


class AssociationMap(StrictBaseModel):
    visual_imagery: list[AssociationItem] = Field(min_length=1)
    character_state: list[AssociationItem] = Field(min_length=1)
    action_motifs: list[AssociationItem] = Field(min_length=1)
    emotional_keywords: list[AssociationItem] = Field(min_length=1)
    narrative_seeds: list[AssociationItem] = Field(min_length=1)
    spatial_symbols: list[AssociationItem] = Field(min_length=1)
    light_color_texture: list[AssociationItem] = Field(min_length=1)
    copy_tone: list[AssociationItem] = Field(min_length=1)

    @property
    def total_count(self) -> int:
        return sum(len(items) for items in self.model_dump().values())


class EmotionalArc(StrictBaseModel):
    origin: str = Field(min_length=1)
    development: str = Field(min_length=1)
    release: str = Field(min_length=1)
    arc_summary: str = Field(min_length=1)


class DirectorPossibility(StrictBaseModel):
    name: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    emotional_direction: str = Field(min_length=1)
    visual_direction: str = Field(min_length=1)
    narrative_direction: str = Field(min_length=1)


class AssociationAnalysis(StrictBaseModel):
    input_text: str = Field(min_length=1)
    query_tags: TagProfile
    core_reading: str = Field(min_length=1)
    emotional_arc: EmotionalArc
    association_count: int = Field(ge=8, le=120)
    association_map: AssociationMap
    director_possibilities: list[DirectorPossibility] = Field(min_length=3, max_length=5)
    avoid_cliches: list[str] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_query_fingerprint(cls, data):
        if isinstance(data, dict) and "query_tags" not in data and "query_fingerprint" in data:
            data = dict(data)
            data["query_tags"] = data.pop("query_fingerprint")
        return data

    @property
    def query_fingerprint(self) -> TagProfile:
        return self.query_tags

    @field_validator("avoid_cliches")
    @classmethod
    def validate_avoid_cliches(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("avoid_cliches cannot contain empty strings")
        return values

    @model_validator(mode="after")
    def association_count_matches(self) -> "AssociationAnalysis":
        total_count = self.association_map.total_count
        if self.association_count != total_count:
            raise ValueError("association_count must match total association items")
        return self
