from __future__ import annotations

import re
from typing import Any

from sceneweaver.retrieval.models import QueryPlan, QueryUseCase, RetrievalWeights
from sceneweaver.retrieval.style import infer_card_style_risks, infer_card_style_traits
from sceneweaver.retrieval.style import parse_negative_styles, parse_positive_styles
from sceneweaver.schemas import ExperienceCard
from sceneweaver.schemas.experience_card import ScriptStage

STAGE_ALIASES: dict[ScriptStage, tuple[str, ...]] = {
    "opening": ("opening", "open", "开场", "开头", "起始", "开篇"),
    "setup": ("setup", "铺垫", "需求建立", "背景建立", "问题建立", "前情"),
    "character_intro": ("character_intro", "character intro", "人物出场", "人物介绍", "介绍人物"),
    "team_work": ("team_work", "team work", "团队协作", "团队", "协作", "共创"),
    "growth": ("growth", "成长", "升温", "潜力", "发展"),
    "technology_showcase": (
        "technology_showcase",
        "technology showcase",
        "technology entrance",
        "技术展示",
        "技术入场",
        "科技展示",
        "产品展示",
        "功能展示",
    ),
    "value_expression": ("value_expression", "value expression", "价值表达", "建立信任", "信任表达"),
    "ending": ("ending", "结尾", "收束", "结尾收束", "号召"),
    "transition": ("transition", "过渡", "转场", "承接"),
    "general": ("general", "通用"),
}

PURPOSE_ALIASES = {
    "establish_problem": ("建立问题", "问题建立", "现实压力"),
    "establish_need": ("建立需求", "需求建立"),
    "build_reality": ("建立真实感", "真实感", "真实现场"),
    "build_trust": ("建立信任", "可信赖", "信任"),
    "show_technology": ("表现技术", "技术能力", "科技能力"),
    "show_team": ("表现团队", "团队协作", "团队能力"),
    "show_collaboration": ("表现协作", "协作"),
    "express_value": ("表达价值", "价值表达", "品牌价值"),
    "human_centered_ai": ("人本技术", "科技向善", "技术服务人"),
    "keep_human_warmth": ("人的温度", "人味", "有人味"),
    "bridge_sections": ("承接", "过渡"),
    "attract_talent": ("吸引人才", "招聘", "加入"),
}

NEGATIVE_SPAN_RE = re.compile(r"(不要做成|不要|避免|不想|不是|别|拒绝)([^，。；,;]*)")
VISUAL_HINT_RE = re.compile(r"画面可以(?:有|借用)(.+?)这类")
DESIRED_PATTERNS = (
    re.compile(r"我真正要的是([^，。；,;]+)"),
    re.compile(r"真正要的是([^，。；,;]+)"),
    re.compile(r"要更像([^，。；,;]+)"),
    re.compile(r"更像([^，。；,;]+)"),
    re.compile(r"我需要(?:一个|一段)?([^，。；,;]+)"),
    re.compile(r"想找(?:一个|一段)?([^，。；,;]+)"),
)

NEGATIVE_ALIAS_CANDIDATES = {
    "大厂味": ("大厂", "互联网大厂", "泛泛办公", "空泛办公", "generic office"),
    "广告感": ("广告", "硬广", "口号", "slogan", "campaign"),
    "炫技": ("炫技", "功能说明", "技术堆砌", "冷冰冰", "纯科技"),
}


def build_query_plan(input_text: str, *, extra_positive_text: str = "") -> QueryPlan:
    original = _clean_text(input_text)
    negative_constraints = _parse_negative_constraints(original)
    forbidden_stage = _parse_forbidden_stage(original)
    positive_without_negatives = _strip_negative_spans(original)
    desired_stage = _parse_desired_stage(positive_without_negatives)
    positive_purposes = _parse_positive_purposes(positive_without_negatives)
    positive_style = parse_positive_styles(positive_without_negatives)
    negative_style = _parse_negative_styles(original)
    visual_hints = _parse_visual_hints(original)
    positive_query = _clean_text(" ".join(part for part in (positive_without_negatives, extra_positive_text) if part))
    return QueryPlan(
        original_text=original,
        positive_query=positive_query or original,
        desired_stage=desired_stage,
        forbidden_stage=forbidden_stage,
        positive_purposes=positive_purposes,
        positive_style=positive_style,
        negative_style=negative_style,
        style_constraints={
            "positive": positive_style,
            "negative": negative_style,
        },
        negative_constraints=negative_constraints,
        visual_hints=visual_hints,
    )


def query_usecase_from_plan(plan: QueryPlan, fallback: QueryUseCase) -> QueryUseCase:
    stage = plan.desired_stage[0] if plan.desired_stage else fallback.script_stage
    purposes = _dedupe([*plan.positive_purposes, *fallback.creative_purpose])
    return QueryUseCase(
        script_stage=stage,
        creative_purpose=purposes or ["general_expression"],
        confidence=max(fallback.confidence, 0.75 if plan.desired_stage else fallback.confidence),
    )


def card_has_forbidden_stage(plan: QueryPlan | None, card: ExperienceCard) -> bool:
    if plan is None:
        return False
    return card.script_usecase.script_stage in set(plan.forbidden_stage)


def score_query_constraints(
    plan: QueryPlan | None,
    card: ExperienceCard,
    *,
    weights: RetrievalWeights | None = None,
) -> tuple[float, dict[str, list[str]]]:
    if plan is None:
        return 0.0, {}
    active_weights = weights or RetrievalWeights()
    score = 0.0
    hits: dict[str, list[str]] = {}
    card_stage = card.script_usecase.script_stage
    if card_stage in set(plan.forbidden_stage):
        score -= active_weights.forbidden_stage_penalty
        hits["forbidden_stage"] = [card_stage]
        return round(score, 3), hits
    if card_stage in set(plan.desired_stage):
        score += active_weights.desired_stage_bonus
        hits["desired_stage"] = [card_stage]
    negative_hits = _negative_constraint_hits(plan.negative_constraints, card)
    if negative_hits:
        score -= active_weights.negative_constraint_penalty * len(negative_hits)
        hits["negative_constraints"] = negative_hits
    style_risks = infer_card_style_risks(card)
    style_traits = infer_card_style_traits(card)
    negative_style_hits = [style for style in plan.negative_style if style in style_risks]
    if negative_style_hits:
        score -= active_weights.style_penalty * len(negative_style_hits)
        hits["negative_style"] = negative_style_hits
    positive_style_hits = [style for style in plan.positive_style if style in style_traits]
    if positive_style_hits:
        score += active_weights.style_bonus * len(positive_style_hits)
        hits["positive_style"] = positive_style_hits
    return round(score, 3), hits


def _parse_negative_constraints(text: str) -> list[str]:
    constraints: list[str] = []
    for _prefix, segment in NEGATIVE_SPAN_RE.findall(text):
        clean = _clean_negative_constraint(segment)
        if clean and clean not in constraints:
            constraints.append(clean)
    return constraints


def _parse_negative_styles(text: str) -> list[str]:
    styles: list[str] = []
    for _prefix, segment in NEGATIVE_SPAN_RE.findall(text):
        for style in parse_negative_styles(segment):
            if style not in styles:
                styles.append(style)
    return styles


def _parse_forbidden_stage(text: str) -> list[ScriptStage]:
    stages: list[ScriptStage] = []
    for _prefix, segment in NEGATIVE_SPAN_RE.findall(text):
        for stage in _stage_mentions(segment):
            if stage != "general" and stage not in stages:
                stages.append(stage)
    return stages


def _parse_desired_stage(text: str) -> list[ScriptStage]:
    stages: list[ScriptStage] = []
    for pattern in DESIRED_PATTERNS:
        for segment in pattern.findall(text):
            for stage in _stage_mentions(segment):
                if stage != "general" and stage not in stages:
                    stages.append(stage)
    if stages:
        return stages
    for stage in _stage_mentions(text):
        if stage != "general" and stage not in stages:
            stages.append(stage)
    return stages


def _parse_positive_purposes(text: str) -> list[str]:
    purposes: list[str] = []
    for purpose, aliases in PURPOSE_ALIASES.items():
        if any(alias.lower() in text.lower() for alias in aliases):
            purposes.append(purpose)
    return purposes


def _parse_visual_hints(text: str) -> list[str]:
    match = VISUAL_HINT_RE.search(text)
    if not match:
        return []
    return [part.strip() for part in re.split(r"[、，,]", match.group(1)) if part.strip()]


def _stage_mentions(text: str) -> list[ScriptStage]:
    lower = text.lower()
    stages: list[ScriptStage] = []
    for stage, aliases in STAGE_ALIASES.items():
        if any(alias.lower() in lower for alias in aliases) and stage not in stages:
            stages.append(stage)
    return stages


def _strip_negative_spans(text: str) -> str:
    return _clean_text(NEGATIVE_SPAN_RE.sub("", text))


def _negative_constraint_hits(constraints: list[str], card: ExperienceCard) -> list[str]:
    card_text = _card_constraint_text(card)
    hits: list[str] = []
    for constraint in constraints:
        candidates = _negative_candidates(constraint)
        if any(candidate and candidate in card_text for candidate in candidates):
            hits.append(constraint)
    return hits


def _negative_candidates(constraint: str) -> list[str]:
    clean = _clean_negative_constraint(constraint)
    candidates = [clean]
    for key, aliases in NEGATIVE_ALIAS_CANDIDATES.items():
        if key in clean:
            candidates.extend(aliases)
    return _dedupe([candidate.lower() for candidate in candidates if candidate])


def _card_constraint_text(card: ExperienceCard) -> str:
    tag_parts: list[str] = []
    for value in card.tags.model_dump(mode="json").values():
        if isinstance(value, list):
            tag_parts.extend(str(item) for item in value if isinstance(item, str))
    usecase = card.script_usecase
    parts: list[str] = [
        *card.keywords,
        card.underlying_emotion,
        card.narrative_logic,
        card.director_strategy,
        *card.shooting_techniques,
        *card.visual_symbols,
        card.copywriting_tone,
        card.reuse_condition,
        usecase.script_stage,
        *usecase.creative_purpose,
        usecase.best_usage,
        usecase.risk,
        *tag_parts,
    ]
    return " ".join(part for part in parts if part).lower()


def _clean_negative_constraint(text: str) -> str:
    cleaned = _clean_text(text)
    for prefix in ("做成", "强调", "变成"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned.strip()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip(" ，。；,;")


def _dedupe(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
