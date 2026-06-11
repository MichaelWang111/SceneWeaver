from __future__ import annotations

from pydantic import Field

from sceneweaver.schemas.common import StrictBaseModel
from sceneweaver.schemas.experience_card import ScriptStage
from sceneweaver.schemas.tags import TagProfile


class QueryUseCase(StrictBaseModel):
    script_stage: ScriptStage = "general"
    creative_purpose: list[str] = Field(default_factory=lambda: ["general_expression"], min_length=1)
    confidence: float = Field(default=0.35, ge=0, le=1)


class QueryPlan(StrictBaseModel):
    original_text: str = ""
    positive_query: str = ""
    desired_stage: list[ScriptStage] = Field(default_factory=list)
    forbidden_stage: list[ScriptStage] = Field(default_factory=list)
    positive_purposes: list[str] = Field(default_factory=list)
    positive_style: list[str] = Field(default_factory=list)
    negative_style: list[str] = Field(default_factory=list)
    style_constraints: dict[str, list[str]] = Field(default_factory=dict)
    negative_constraints: list[str] = Field(default_factory=list)
    visual_hints: list[str] = Field(default_factory=list)


class RetrievalWeights(StrictBaseModel):
    script_stage_match: float = 3.0
    creative_purpose_match: float = 1.5
    quality: float = 0.5
    desired_stage_bonus: float = 1.0
    forbidden_stage_penalty: float = 6.0
    negative_constraint_penalty: float = 1.0
    style_bonus: float = 0.8
    style_penalty: float = 1.5


class RetrievalRequest(StrictBaseModel):
    input_text: str = ""
    query_tags: TagProfile
    query_usecase: QueryUseCase | None = None
    query_plan: QueryPlan | None = None
    top_k: int = Field(default=5, ge=1)
    semantic_weight: float = Field(default=0.0, ge=0)
    lexical_weight: float = Field(default=2.0, ge=0)
    intent_weight: float = Field(default=0.0, ge=0)
    retrieval_workflow: str = "semantic_constraints"
    constraints_enabled: bool = True
    hard_filter_forbidden_stage: bool = True
