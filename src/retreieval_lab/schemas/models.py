from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


SCHEMA_VERSION = "2026-06-12"

KnownStage = Literal[
    "general",
    "opening",
    "setup",
    "technology_showcase",
    "team_work",
    "value_expression",
    "outcome",
    "ending",
]
ConstraintPolarity = Literal["must", "must_not", "should", "should_not"]
ConstraintStrength = Literal["hard", "soft", "boost", "penalty"]
ConstraintKind = Literal["stage", "style", "purpose", "visual", "scene_signature", "text", "metadata", "safety"]
PlannerSource = Literal["rule", "multi_query", "hyde_card", "llm", "human", "bootstrap", "legacy_adapter", "system"]


class LabSchemaModel(BaseModel):
    """Forward-compatible base model for Retrieval Lab records."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class EvidenceRefModel(LabSchemaModel):
    source: str = Field(default="", description="Where this value came from: rule name, model, human note, or artifact path.")
    field: str = Field(default="", description="Source field used as evidence.")
    text: str = Field(default="", description="Short evidence text. Keep long prompts/debug text out of default reports.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SceneSignatureModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    people: list[str] = Field(default_factory=list)
    place: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    emotion_function: list[str] = Field(default_factory=list)
    narrative_position: list[str] = Field(default_factory=list)
    camera_experience: list[str] = Field(default_factory=list)
    script_reuse_pattern: list[str] = Field(default_factory=list)
    raw_positive_query: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: list[EvidenceRefModel] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        result = dict(data)
        if "action" in result and "actions" not in result:
            result["actions"] = result["action"]
        if "emotional_function" in result and "emotion_function" not in result:
            result["emotion_function"] = result["emotional_function"]
        return result

    @field_validator(
        "people",
        "place",
        "actions",
        "objects",
        "emotion_function",
        "narrative_position",
        "camera_experience",
        "script_reuse_pattern",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        return string_list(value)

    @model_validator(mode="after")
    def _not_empty_signature(self) -> "SceneSignatureModel":
        fields = (
            self.people,
            self.place,
            self.actions,
            self.objects,
            self.emotion_function,
            self.narrative_position,
            self.camera_experience,
            self.script_reuse_pattern,
        )
        if not any(fields) and not self.raw_positive_query:
            raise ValueError("scene_signature must contain at least one signal or raw_positive_query")
        return self


class QueryConstraintModel(LabSchemaModel):
    constraint_id: str = ""
    kind: ConstraintKind = "text"
    polarity: ConstraintPolarity = "should"
    values: list[str] = Field(default_factory=list, min_length=1)
    operator: Literal["equals", "contains", "overlap", "semantic", "exists"] = "overlap"
    strength: ConstraintStrength = "soft"
    source: PlannerSource = "rule"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value: Any) -> list[str]:
        return string_list(value)


class QueryRewriteModel(LabSchemaModel):
    rewrite_id: str = ""
    text: str = Field(min_length=1)
    purpose: Literal["semantic_recall", "lexical_recall", "hyde", "disambiguation", "style_alignment"] = "semantic_recall"
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    target_channels: list[str] = Field(default_factory=list)
    source: PlannerSource = "multi_query"

    @field_validator("target_channels", mode="before")
    @classmethod
    def _coerce_channels(cls, value: Any) -> list[str]:
        return string_list(value)


class QueryAmbiguityModel(LabSchemaModel):
    level: Literal["low", "medium", "high"] = "low"
    reasons: list[str] = Field(default_factory=list)
    needs_review: bool = False
    suggested_resolution: str = ""

    @field_validator("reasons", mode="before")
    @classmethod
    def _coerce_reasons(cls, value: Any) -> list[str]:
        return string_list(value)


class PlannerProvenanceModel(LabSchemaModel):
    planner: str = "rule"
    planner_version: str = ""
    source: PlannerSource = "rule"
    cache_key: str = ""
    cache_hit: bool = False
    llm_used: bool = False
    fallback_to: str = ""
    fallback_reason: str = ""
    latency_ms: float | None = Field(default=None, ge=0.0)


class QueryPlanModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    query_id: str = ""
    planner: str = "rule"
    original_text: str = Field(min_length=1)
    positive_query: str = Field(min_length=1)
    language: str = "zh-CN"
    desired_stage: list[str] = Field(default_factory=list)
    forbidden_stage: list[str] = Field(default_factory=list)
    positive_purposes: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    visual_hints: list[str] = Field(default_factory=list)
    positive_style: list[str] = Field(default_factory=list)
    negative_style: list[str] = Field(default_factory=list)
    scene_signature: SceneSignatureModel | None = None
    hard_constraints: list[QueryConstraintModel] = Field(default_factory=list)
    soft_constraints: list[QueryConstraintModel] = Field(default_factory=list)
    ambiguity: QueryAmbiguityModel = Field(default_factory=QueryAmbiguityModel)
    rewrites: list[QueryRewriteModel] = Field(default_factory=list)
    hyde_text: str = ""
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    provenance: PlannerProvenanceModel = Field(default_factory=PlannerProvenanceModel)
    planner_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_plan(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        result = dict(data)
        if "planner_name" in result and "planner" not in result:
            result["planner"] = result["planner_name"]
        if "scene_signature" in result and result["scene_signature"] == {}:
            result["scene_signature"] = None
        result["rewrites"] = normalize_rewrites(result.get("rewrites"))
        result["hard_constraints"] = normalize_constraints(result.get("hard_constraints"), strength="hard")
        result["soft_constraints"] = normalize_constraints(result.get("soft_constraints"), strength="soft")
        if "provenance" not in result:
            metadata = result.get("planner_metadata", {}) if isinstance(result.get("planner_metadata"), dict) else {}
            result["provenance"] = {
                "planner": result.get("planner", "rule"),
                "source": metadata.get("source", "rule") if metadata.get("source") in PlannerSource.__args__ else "rule",
                "cache_hit": bool(metadata.get("cache_hit", False)),
                "llm_used": bool(metadata.get("llm_used", False)),
                "fallback_to": str(metadata.get("fallback_to", "")),
                "fallback_reason": str(metadata.get("fallback_reason", "")),
            }
        return result

    @field_validator(
        "desired_stage",
        "forbidden_stage",
        "positive_purposes",
        "negative_constraints",
        "visual_hints",
        "positive_style",
        "negative_style",
        mode="before",
    )
    @classmethod
    def _coerce_string_lists(cls, value: Any) -> list[str]:
        return string_list(value)

    @model_validator(mode="after")
    def _negative_terms_do_not_leak(self) -> "QueryPlanModel":
        negative_terms = set(self.forbidden_stage) | set(self.negative_style) | set(self.negative_constraints)
        for constraint in [*self.hard_constraints, *self.soft_constraints]:
            if constraint.polarity in {"must_not", "should_not"}:
                negative_terms.update(constraint.values)
        negative_terms = {term.strip().lower() for term in negative_terms if len(term.strip()) > 1}
        positive_texts = [self.positive_query, self.hyde_text, *[rewrite.text for rewrite in self.rewrites]]
        for term in negative_terms:
            for text in positive_texts:
                if term and term in text.lower():
                    raise ValueError(f"negative term leaked into positive retrieval text: {term}")
        return self


class ChannelScoreModel(LabSchemaModel):
    channel: str
    score: float
    weight: float = 1.0


class CandidateScoreModel(LabSchemaModel):
    item_id: str = Field(min_length=1)
    rank: int | None = Field(default=None, ge=1)
    score: float | None = None
    final_score: float | None = None
    semantic_score: float | None = None
    lexical_score: float | None = None
    constraint_score: float | None = None
    signature_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    constraint_hits: dict[str, list[str]] = Field(default_factory=dict)
    channel_scores: list[ChannelScoreModel] = Field(default_factory=list)
    explanation: str = ""

    @model_validator(mode="after")
    def _has_score(self) -> "CandidateScoreModel":
        if self.score is None and self.final_score is None and self.rerank_score is None:
            raise ValueError("candidate requires at least one score-like field")
        return self


class RunRowModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    run_name: str = ""
    case_id: str = Field(min_length=1)
    user_input: str = ""
    query_plan: QueryPlanModel | dict[str, Any] | None = None
    target_item_id: str = ""
    target_stage: str = ""
    target_purposes: list[str] = Field(default_factory=list)
    target_rank: int | None = Field(default=None, ge=1)
    ranking_key: str = ""
    top_results: list[CandidateScoreModel] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = Field(default=None, ge=0.0)

    @field_validator("target_purposes", mode="before")
    @classmethod
    def _coerce_purposes(cls, value: Any) -> list[str]:
        return string_list(value)


class QrelVoteModel(LabSchemaModel):
    grade: int = Field(ge=0, le=3)
    judge_type: Literal["bootstrap", "human", "llm"] = "bootstrap"
    judge_id: str = ""
    judge_version: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    created_at: str = ""


class QrelModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    query_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    grade: int = Field(ge=0, le=3)
    reason: str = ""
    source: str = "bootstrap"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    grade_votes: list[QrelVoteModel] = Field(default_factory=list)
    needs_adjudication: bool = False
    pooled_from: list[str] = Field(default_factory=list)

    @field_validator("pooled_from", mode="before")
    @classmethod
    def _coerce_pooled_from(cls, value: Any) -> list[str]:
        return string_list(value)


class IndexManifestModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    index_id: str = Field(min_length=1)
    source_dataset_id: str = ""
    item_count: int = Field(ge=0)
    channels: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    lexical_tokenizer: str = ""
    fingerprint: str = ""
    cache_paths: list[str] = Field(default_factory=list)
    created_at: str = ""

    @field_validator("channels", "cache_paths", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> list[str]:
        return string_list(value)


class RetrievalRunConfigModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    workflow: str = Field(min_length=1)
    ranking_key: str = ""
    query_planner: str = "rule"
    top_k: int = Field(default=10, ge=1)
    candidate_depth: int = Field(default=100, ge=1)
    index_id: str = ""
    constraints_enabled: bool = True
    llm_enabled: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)


class LLMJudgementModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    query_id: str = Field(min_length=1)
    candidate_item_id: str = Field(min_length=1)
    judgement_type: Literal["qrel_grade", "rerank", "constraint_check"] = "qrel_grade"
    grade: int | None = Field(default=None, ge=0, le=3)
    should_veto: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""
    model: str = ""
    prompt_fingerprint: str = ""
    payload_chars: int | None = Field(default=None, ge=0)


class CapabilityCycleModel(LabSchemaModel):
    schema_version: str = SCHEMA_VERSION
    cycle_id: str = Field(min_length=1)
    label: str = ""
    created_at: str = ""
    git_sha: str = ""
    is_origin: bool = False
    input_reports: list[dict[str, Any]] = Field(default_factory=list)
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    delta_vs_previous: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)


SCHEMA_MODELS: dict[str, type[LabSchemaModel]] = {
    "scene_signature": SceneSignatureModel,
    "query_constraint": QueryConstraintModel,
    "query_rewrite": QueryRewriteModel,
    "query_plan": QueryPlanModel,
    "candidate_score": CandidateScoreModel,
    "run_row": RunRowModel,
    "qrel": QrelModel,
    "index_manifest": IndexManifestModel,
    "retrieval_run_config": RetrievalRunConfigModel,
    "llm_judgement": LLMJudgementModel,
    "capability_cycle": CapabilityCycleModel,
}

SCHEMA_DESCRIPTIONS: dict[str, str] = {
    "scene_signature": "Structured scene identity signals for exact-scene ranking.",
    "query_constraint": "Hard/soft positive and negative query constraints.",
    "query_rewrite": "A planner-generated retrieval text variant.",
    "query_plan": "Canonical query understanding contract for rule, multi-query, HyDE, and LLM planners.",
    "candidate_score": "One scored retrieval candidate with metadata, score channels, and explanations.",
    "run_row": "One query/run result row used by artifact-first evaluation.",
    "qrel": "Graded relevance judgement with votes, confidence, and adjudication metadata.",
    "index_manifest": "Prepared index/cache identity, channel, and fingerprint metadata.",
    "retrieval_run_config": "Search workflow configuration used to make a run reproducible.",
    "llm_judgement": "Small sampled LLM judgement contract for qrels/rerank checks.",
    "capability_cycle": "Longitudinal capability cycle record.",
}


def schema_model(name: str) -> type[LabSchemaModel]:
    key = name.replace("-", "_")
    try:
        return SCHEMA_MODELS[key]
    except KeyError as exc:
        raise ValueError(f"unknown Retrieval Lab schema: {name}") from exc


def schema_catalog(*, include_json_schema: bool = False) -> list[dict[str, Any]]:
    rows = []
    for name, model in sorted(SCHEMA_MODELS.items()):
        row: dict[str, Any] = {
            "name": name,
            "schema_version": SCHEMA_VERSION,
            "description": SCHEMA_DESCRIPTIONS.get(name, ""),
            "required_fields": sorted(model.model_json_schema().get("required", [])),
            "field_count": len(model.model_fields),
        }
        if include_json_schema:
            row["json_schema"] = model.model_json_schema()
        rows.append(row)
    return rows


def json_schema_for(name: str) -> dict[str, Any]:
    return schema_model(name).model_json_schema()


def validate_record(schema_name: str, record: dict[str, Any]) -> dict[str, Any]:
    model = schema_model(schema_name)
    try:
        normalized = model.model_validate(record).model_dump(mode="json", exclude_none=True)
    except ValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {
                    "loc": ".".join(str(part) for part in error.get("loc", [])),
                    "message": error.get("msg", ""),
                    "type": error.get("type", ""),
                }
                for error in exc.errors()
            ],
        }
    return {"valid": True, "normalized": normalized, "errors": []}


def validate_records(schema_name: str, records: Iterable[dict[str, Any]], *, max_issues: int = 50) -> dict[str, Any]:
    issues = []
    valid_count = 0
    normalized_sample: list[dict[str, Any]] = []
    record_count = 0
    for index, record in enumerate(records):
        record_count += 1
        result = validate_record(schema_name, record)
        if result["valid"]:
            valid_count += 1
            if len(normalized_sample) < 3:
                normalized_sample.append(result["normalized"])
        elif len(issues) < max_issues:
            issues.append({"row_index": index, "errors": result["errors"]})
    return {
        "schema_name": schema_name.replace("-", "_"),
        "record_count": record_count,
        "valid_count": valid_count,
        "invalid_count": record_count - valid_count,
        "valid_rate": round(valid_count / max(1, record_count), 6),
        "issues": issues,
        "normalized_sample": normalized_sample,
    }


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, tuple | set):
        value = list(value)
    if isinstance(value, list):
        result = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def normalize_rewrites(value: Any) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(value if isinstance(value, list) else string_list(value)):
        if isinstance(item, dict):
            row = dict(item)
        else:
            row = {"text": str(item)}
        row.setdefault("rewrite_id", f"rewrite_{index + 1}")
        rows.append(row)
    return rows


def normalize_constraints(value: Any, *, strength: ConstraintStrength) -> list[dict[str, Any]]:
    rows = []
    values = value if isinstance(value, list) else string_list(value)
    for index, item in enumerate(values):
        if isinstance(item, dict):
            row = dict(item)
        else:
            text = str(item)
            kind, _, raw_value = text.partition(":")
            row = {
                "kind": kind if kind in ConstraintKind.__args__ else "text",
                "values": [raw_value or text],
                "polarity": "must_not" if "forbidden" in text or "negative" in text else "should",
            }
        row.setdefault("constraint_id", f"{strength}_{index + 1}")
        row.setdefault("strength", strength)
        rows.append(row)
    return rows
