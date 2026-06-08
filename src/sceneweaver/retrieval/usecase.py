from __future__ import annotations

from collections import defaultdict

from sceneweaver.retrieval.models import QueryUseCase
from sceneweaver.schemas.experience_card import ScriptStage, ScriptUseCase
from sceneweaver.schemas.tags import TagProfile

_STAGE_PRIORITY: dict[ScriptStage, int] = {
    "team_work": 90,
    "value_expression": 80,
    "technology_showcase": 70,
    "growth": 60,
    "opening": 50,
    "ending": 40,
    "character_intro": 30,
    "setup": 20,
    "transition": 10,
    "general": 0,
}

_USAGE_TEXT: dict[ScriptStage, tuple[str, str]] = {
    "opening": (
        "Use as an opening atmosphere or scale-establishing reference.",
        "May feel empty if it is not connected to a human point of view.",
    ),
    "setup": (
        "Use to establish the working environment before the core character or value appears.",
        "May become generic location coverage if no later scene gives it narrative purpose.",
    ),
    "character_intro": (
        "Use when introducing an employee, creator, founder, or representative subject.",
        "May over-focus on the individual if the brief needs collective energy.",
    ),
    "team_work": (
        "Use in a team culture, collaboration, or collective action segment.",
        "May feel like generic office montage if the collaboration logic is not explicit.",
    ),
    "growth": (
        "Use in a personal growth, potential, or future-facing development segment.",
        "May become slogan-like if the concrete work process is missing.",
    ),
    "technology_showcase": (
        "Use in a technology capability or human-centered technology segment.",
        "May become cold or purely functional if people are absent.",
    ),
    "value_expression": (
        "Use when the script needs trust, values, credibility, or direct audience alignment.",
        "May feel too declarative if overused without visual evidence.",
    ),
    "ending": (
        "Use near the ending call-to-action or invitation to join.",
        "May feel premature if used before the audience understands the value proposition.",
    ),
    "transition": (
        "Use as a rhythm bridge between two script sections.",
        "May not carry enough meaning as a standalone reference.",
    ),
    "general": (
        "Use as a general directing reference when no stronger script role is available.",
        "Needs additional creative context before direct reuse.",
    ),
}


def infer_query_usecase(input_text: str, query_tags: TagProfile) -> QueryUseCase:
    usecase = build_script_usecase(query_tags, text=input_text, base_confidence=query_tags.confidence)
    return QueryUseCase(
        script_stage=usecase.script_stage,
        creative_purpose=usecase.creative_purpose,
        confidence=usecase.confidence,
    )


def build_script_usecase(
    tags: TagProfile,
    *,
    text: str = "",
    base_confidence: float = 0.35,
) -> ScriptUseCase:
    haystack = _haystack(tags, text)
    stage_scores: dict[ScriptStage, float] = defaultdict(float)
    purposes: list[str] = []

    def add(stage: ScriptStage, purpose: str, score: float) -> None:
        stage_scores[stage] += score
        if purpose not in purposes:
            purposes.append(purpose)

    if _has(haystack, "culture_showcase", "team_collaboration", "belonging", "team", "collaboration", "团队", "协作", "共创"):
        add("team_work", "show_team", 4.0)
    if _has(haystack, "becoming", "ambition", "future_builder", "growth", "potential", "成长", "年轻", "潜力", "未来"):
        add("growth", "show_growth", 3.0)
    if _has(haystack, "human_centered_technology", "screen", "technology", "interface", "科技", "技术", "屏幕", "人本科技"):
        add("technology_showcase", "show_technology", 2.5)
    if _has(
        haystack,
        "trust",
        "human_care",
        "establish_trust",
        "direct_address",
        "direct_listener",
        "calm_direct",
        "credibility",
        "可信",
        "可靠",
        "信任",
        "价值观",
        "科技向善",
        "面对面",
    ):
        add("value_expression", "build_trust", 3.5)
    if _has(haystack, "scale_reveal", "awe", "urban_architecture", "opening", "scale", "开场", "气质", "规模", "宏大"):
        add("opening", "show_scale", 3.0)
    if _has(haystack, "invitation", "join", "call_to_action", "ending", "结尾", "加入", "号召", "邀请"):
        add("ending", "attract_talent", 3.0)
    if _has(haystack, "character", "employee", "interview", "人物", "员工", "访谈", "个人"):
        add("character_intro", "introduce_people", 2.5)
    if _has(haystack, "setting", "environment", "workplace", "setup", "环境", "办公室", "工厂", "校园"):
        add("setup", "establish_context", 2.0)
    if _has(haystack, "transition", "contrast_cut", "转场", "过渡"):
        add("transition", "bridge_sections", 2.0)

    if not stage_scores:
        return ScriptUseCase(
            script_stage="general",
            creative_purpose=["general_expression"],
            best_usage=_USAGE_TEXT["general"][0],
            risk=_USAGE_TEXT["general"][1],
            confidence=max(0.35, min(0.6, base_confidence)),
        )

    stage = max(
        stage_scores,
        key=lambda candidate: (stage_scores[candidate], _STAGE_PRIORITY[candidate]),
    )
    best_usage, risk = _USAGE_TEXT[stage]
    confidence = min(0.95, max(0.45, base_confidence, 0.4 + len(purposes) * 0.08))
    return ScriptUseCase(
        script_stage=stage,
        creative_purpose=purposes,
        best_usage=best_usage,
        risk=risk,
        confidence=round(confidence, 3),
    )


def _haystack(tags: TagProfile, text: str) -> str:
    parts = [text]
    data = tags.model_dump(mode="json")
    for value in data.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
    return " ".join(parts).lower()


def _has(haystack: str, *terms: str) -> bool:
    return any(term.lower() in haystack for term in terms)
