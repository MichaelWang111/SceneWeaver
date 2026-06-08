from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

from sceneweaver.analysis.associate_analyzer import (
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    AssociateLLMClient,
    _call_llm_with_heartbeat,
)
from sceneweaver.analysis.tags import build_query_tags
from sceneweaver.llm.client import VisionLLMClient
from sceneweaver.schemas import TagExpansionAnalysis
from sceneweaver.storage.json_store import write_json

DEFAULT_TAG_EXPANSION_MAX_TOKENS = 1800
LogFn = Callable[[str], None]


def expand_input_tags(
    input_text: str,
    *,
    client: AssociateLLMClient | None = None,
    output_path: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    log: LogFn | None = None,
    stream_callback: Callable[[str], None] | None = None,
    reasoning_callback: Callable[[str], None] | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
) -> TagExpansionAnalysis:
    started_at = perf_counter()
    clean_input = input_text.strip()
    if not clean_input:
        raise ValueError("input_text cannot be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if retries < 0:
        raise ValueError("retries must be >= 0")

    _log(log, "Loading lightweight tag expansion prompt.")
    llm_client = client or VisionLLMClient()
    raw = _call_llm_with_heartbeat(
        client=llm_client,
        system_prompt=build_tag_expansion_system_prompt(),
        user_prompt=build_tag_expansion_user_prompt(clean_input),
        max_tokens=DEFAULT_TAG_EXPANSION_MAX_TOKENS,
        timeout_seconds=timeout_seconds,
        retries=retries,
        stream_callback=stream_callback,
        reasoning_callback=reasoning_callback,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        log=log,
        started_at=started_at,
    )
    if stream_callback is not None:
        stream_callback("\n")
    _log(log, f"LLM tag expansion received after {perf_counter() - started_at:.1f}s. Validating JSON schema.")
    raw["input_text"] = clean_input
    expanded_text = _tag_expansion_text(raw)
    raw["expanded_text"] = expanded_text
    raw["query_tags"] = build_query_tags(
        clean_input,
        extra_text=expanded_text,
        candidate_log_path=output_path.parent / "tag_candidates.jsonl" if output_path is not None else None,
    ).model_dump(mode="json")
    analysis = TagExpansionAnalysis.model_validate(raw)
    if output_path is not None:
        _log(log, f"Writing lightweight tag expansion JSON: {output_path}")
        write_json(output_path, analysis)
    _log(log, f"Tag expansion complete: expanded_terms={len(analysis.expanded_terms)}")
    return analysis


def build_tag_expansion_system_prompt() -> str:
    return """你是导演经验检索系统的标签扩展器。
只做关键词到标签层面的语义扩展，不写场景方案，不写分镜，不写故事，不输出大段创意联想。
输出必须是 JSON，字段只包含 expanded_terms、tag_hints、avoid_terms。

要求：
1. expanded_terms: 12 到 32 个短词或短语，用于检索同义、近义、导演语义相邻的经验卡。
2. tag_hints: 按 emotion_core、audience_projection、narrative_function、interaction_mode、visual_motifs、symbolic_logic、rhythm_pattern 七个维度给出候选语义词。
3. avoid_terms: 输入中容易导致误检的泛词、空话、过强品牌限定。
4. 不要输出具体场景、角色、镜头设计或故事段落。
5. 不要解释 JSON。"""


def build_tag_expansion_user_prompt(input_text: str) -> str:
    return f"""请对以下关键词做轻量标签扩展，用于经验卡检索。

input_text: {input_text}

请只输出 JSON：
{{
  "expanded_terms": ["..."],
  "tag_hints": {{
    "emotion_core": ["..."],
    "audience_projection": ["..."],
    "narrative_function": ["..."],
    "interaction_mode": ["..."],
    "visual_motifs": ["..."],
    "symbolic_logic": ["..."],
    "rhythm_pattern": ["..."]
  }},
  "avoid_terms": ["..."]
}}"""


def _tag_expansion_text(raw: dict) -> str:
    parts: list[str] = []
    for item in raw.get("expanded_terms", []):
        parts.append(str(item))
    tag_hints = raw.get("tag_hints")
    if isinstance(tag_hints, dict):
        for values in tag_hints.values():
            if isinstance(values, list):
                parts.extend(str(value) for value in values)
    return " ".join(part for part in parts if part)


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)
