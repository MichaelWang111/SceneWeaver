from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from mocktesting.eval_input_generator import STAGE_WORDS

DEFAULT_CONSTRAINT_PROFILE_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_constraint_profile.json"

DEFAULT_PROFILE = {
    "version": 1,
    "weights": {
        "desired_stage_bonus": 0.12,
        "forbidden_stage_penalty": 0.18,
        "negative_constraint_penalty": 0.08,
    },
    "stage_aliases": {},
    "negative_aliases": {},
}

DESIRED_PATTERNS = [
    re.compile(r"我真正要的是([^，。；,]+)"),
    re.compile(r"要更像([^，。；,]+)"),
    re.compile(r"更像([^，。；,]+)"),
]

FORBIDDEN_PATTERNS = [
    re.compile(r"不要做成([^，。；,]+)"),
    re.compile(r"避免([^，。；,]+)"),
    re.compile(r"不要([^，。；,]+)"),
    re.compile(r"不是([^，。；,]+)"),
]

NEGATIVE_PATTERNS = [
    re.compile(r"(不要|避免|不想|不是|别|拒绝)([^，。；,]+)"),
]

VISUAL_HINT_RE = re.compile(r"画面可以(?:有|借用)(.+?)这类")


def parse_query_constraints(user_input: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    active_profile = profile or DEFAULT_PROFILE
    stage_aliases = stage_alias_map(active_profile)
    desired = parse_stage_mentions(user_input, DESIRED_PATTERNS, stage_aliases)
    forbidden = parse_stage_mentions(user_input, FORBIDDEN_PATTERNS, stage_aliases)
    negative_constraints = parse_negative_constraints(user_input, stage_aliases)
    visual_hints = parse_visual_hints(user_input)
    return {
        "desired_stage": desired,
        "forbidden_stage": forbidden,
        "negative_constraints": negative_constraints,
        "visual_hints": visual_hints,
    }


def score_constraints(
    query_constraints: dict[str, Any],
    item_metadata: dict[str, Any],
    profile: dict[str, Any] | None = None,
) -> tuple[float, dict[str, list[str]]]:
    active_profile = profile or DEFAULT_PROFILE
    weights = active_profile.get("weights", {})
    item_stage = item_metadata.get("script_stage", "")
    desired = set(query_constraints.get("desired_stage", []))
    forbidden = set(query_constraints.get("forbidden_stage", []))
    negative_constraints = query_constraints.get("negative_constraints", [])
    score = 0.0
    hits: dict[str, list[str]] = {}

    if item_stage in forbidden:
        score -= float(weights.get("forbidden_stage_penalty", 0.0))
        hits["forbidden_stage"] = [item_stage]
        return round(score, 6), hits

    if item_stage in desired:
        score += float(weights.get("desired_stage_bonus", 0.0))
        hits["desired_stage"] = [item_stage]

    negative_hits = stage_negative_alias_hits(negative_constraints, item_metadata, active_profile)
    if negative_hits:
        score -= float(weights.get("negative_constraint_penalty", 0.0)) * len(negative_hits)
        hits["negative_constraints"] = negative_hits

    return round(score, 6), hits


def load_constraint_profile(path: Path = DEFAULT_CONSTRAINT_PROFILE_PATH) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_PROFILE))
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = json.loads(json.dumps(DEFAULT_PROFILE))
    profile.update({key: value for key, value in data.items() if key not in {"weights", "stage_aliases", "negative_aliases"}})
    profile["weights"].update(data.get("weights", {}))
    profile["stage_aliases"].update(data.get("stage_aliases", {}))
    profile["negative_aliases"].update(data.get("negative_aliases", {}))
    return profile


def write_constraint_profile(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_alias_map(profile: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for stage, word in STAGE_WORDS.items():
        aliases[word] = stage
        aliases[stage] = stage
        aliases[stage.replace("_", " ")] = stage
    aliases.update(profile.get("stage_aliases", {}))
    return aliases


def parse_stage_mentions(
    text: str,
    patterns: list[re.Pattern[str]],
    aliases: dict[str, str],
) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        for match in pattern.findall(text):
            segment = match if isinstance(match, str) else "".join(match)
            for alias, stage in aliases.items():
                if alias and alias in segment and stage not in hits:
                    hits.append(stage)
    return hits


def parse_negative_constraints(text: str, aliases: dict[str, str]) -> list[str]:
    constraints: list[str] = []
    for pattern in NEGATIVE_PATTERNS:
        for match in pattern.findall(text):
            segment = "".join(match).strip()
            if segment and segment not in constraints:
                constraints.append(segment)
    return constraints


def parse_visual_hints(text: str) -> list[str]:
    match = VISUAL_HINT_RE.search(text)
    if not match:
        return []
    return [part.strip() for part in re.split(r"[、，,]", match.group(1)) if part.strip()]


def stage_negative_alias_hits(
    negative_constraints: list[str],
    item_metadata: dict[str, Any],
    profile: dict[str, Any],
) -> list[str]:
    item_text = " ".join(
        str(value)
        for value in (
            item_metadata.get("script_stage", ""),
            item_metadata.get("script_use_sentence", ""),
            " ".join(item_metadata.get("creative_purpose", [])),
            item_metadata.get("style", ""),
            item_metadata.get("industry", ""),
        )
        if value
    )
    aliases = profile.get("negative_aliases", {})
    hits: list[str] = []
    for constraint in negative_constraints:
        candidates = aliases.get(constraint, [constraint])
        if isinstance(candidates, str):
            candidates = [candidates]
        if any(candidate and candidate in item_text for candidate in candidates):
            hits.append(constraint)
    return hits


def profile_with_weights(
    *,
    desired_stage_bonus: float,
    forbidden_stage_penalty: float,
    negative_constraint_penalty: float,
    base_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = json.loads(json.dumps(base_profile or DEFAULT_PROFILE))
    profile["weights"] = {
        "desired_stage_bonus": desired_stage_bonus,
        "forbidden_stage_penalty": forbidden_stage_penalty,
        "negative_constraint_penalty": negative_constraint_penalty,
    }
    return profile
