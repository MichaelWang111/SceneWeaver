from __future__ import annotations

from sceneweaver.schemas import ExperienceCard

POSITIVE_STYLE_ALIASES = {
    "human_warmth": ("有人味", "人味", "人的温度", "human warmth", "human-centered", "human centered"),
    "documentary": ("纪录片", "纪实", "观察", "documentary", "observational"),
    "real_location": ("真实现场", "真实场景", "现场感", "real location", "on location"),
}

NEGATIVE_STYLE_ALIASES = {
    "big_company_office": ("大厂味", "大厂", "互联网大厂", "泛泛办公", "空泛办公", "generic office"),
    "ad_like": ("广告感", "广告", "宣传片腔", "硬广", "口号", "slogan", "campaign"),
    "tech_showoff": ("炫技", "技术炫耀", "技术堆砌", "功能说明", "产品说明", "纯科技", "冷冰冰"),
}


def parse_positive_styles(text: str) -> list[str]:
    return _parse_styles(text, POSITIVE_STYLE_ALIASES)


def parse_negative_styles(text: str) -> list[str]:
    return _parse_styles(text, NEGATIVE_STYLE_ALIASES)


def infer_card_style_traits(card: ExperienceCard) -> list[str]:
    explicit = list(getattr(card, "style_traits", []) or [])
    inferred = _parse_styles(_card_style_text(card), POSITIVE_STYLE_ALIASES)
    return _dedupe([*explicit, *inferred])


def infer_card_style_risks(card: ExperienceCard) -> list[str]:
    explicit = list(getattr(card, "style_risks", []) or [])
    inferred = _parse_styles(_card_style_text(card), NEGATIVE_STYLE_ALIASES)
    return _dedupe([*explicit, *inferred])


def style_text(styles: list[str]) -> str:
    return " ".join(styles)


def _parse_styles(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    lower = str(text or "").lower()
    hits: list[str] = []
    for style, terms in aliases.items():
        if any(term.lower() in lower for term in terms):
            hits.append(style)
    return hits


def _card_style_text(card: ExperienceCard) -> str:
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
        *card.avoid,
        usecase.best_usage,
        usecase.risk,
        *usecase.creative_purpose,
    ]
    return " ".join(part for part in parts if part)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
