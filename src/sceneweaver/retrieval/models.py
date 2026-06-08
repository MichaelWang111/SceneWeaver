from __future__ import annotations

from pydantic import Field

from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.schemas.experience_card import ScriptStage
from sceneweaver.schemas.tags import TagProfile


class QueryUseCase(StrictBaseModel):
    script_stage: ScriptStage = "general"
    creative_purpose: list[str] = Field(default_factory=lambda: ["general_expression"], min_length=1)
    confidence: float = Field(default=0.35, ge=0, le=1)


class RetrievalWeights(StrictBaseModel):
    script_stage_match: float = 3.0
    creative_purpose_match: float = 1.5
    quality: float = 0.5


class RetrievalRequest(StrictBaseModel):
    input_text: str = ""
    query_tags: TagProfile
    query_usecase: QueryUseCase | None = None
    top_k: int = Field(default=5, ge=1)
    semantic_weight: float = Field(default=0.0, ge=0)
    intent_weight: float = Field(default=0.0, ge=0)
