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
from sceneweaver.schemas import CreativeIntentAnalysis
from sceneweaver.storage.json_store import write_json

DEFAULT_INTENT_MAX_TOKENS = 1600
LogFn = Callable[[str], None]


def analyze_creative_intent(
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
) -> CreativeIntentAnalysis:
    started_at = perf_counter()
    clean_input = input_text.strip()
    if not clean_input:
        raise ValueError("input_text cannot be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if retries < 0:
        raise ValueError("retries must be >= 0")

    _log(log, "Loading creative intent prompt.")
    llm_client = client or VisionLLMClient()
    raw = _call_llm_with_heartbeat(
        client=llm_client,
        system_prompt=build_intent_system_prompt(),
        user_prompt=build_intent_user_prompt(clean_input),
        max_tokens=DEFAULT_INTENT_MAX_TOKENS,
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
    _log(log, f"Creative intent received after {perf_counter() - started_at:.1f}s. Validating JSON schema.")
    raw["input_text"] = clean_input
    expanded_text = _intent_expanded_text(raw)
    raw["expanded_text"] = expanded_text
    raw["query_tags"] = build_query_tags(
        clean_input,
        extra_text=expanded_text,
        candidate_log_path=output_path.parent / "tag_candidates.jsonl" if output_path is not None else None,
    ).model_dump(mode="json")
    analysis = CreativeIntentAnalysis.model_validate(raw)
    if output_path is not None:
        _log(log, f"Writing creative intent JSON: {output_path}")
        write_json(output_path, analysis)
    _log(log, f"Creative intent complete: must_match={len(analysis.must_match)}, avoid={len(analysis.avoid)}")
    return analysis


def build_intent_system_prompt() -> str:
    return """你是导演经验检索系统的创作意图解析器。
你的任务不是扩写标签、不是写场景、不是给分镜方案，而是判断创作者输入背后的核心检索意图。
输出必须是 JSON，字段只包含 primary_intent、must_match、nice_to_have、avoid、intent_keywords、target_audience、selection_criteria。

要求：
1. primary_intent: 用一句话说明创作者真正想从经验卡里找到什么。
2. must_match: 4 到 10 个必须优先匹配的语义条件，避免泛词，保留关键品牌/行业/受众关系/价值观。
3. nice_to_have: 3 到 8 个加分条件。
4. avoid: 3 到 8 个容易误导检索排序的内容，例如只有宏大科技视觉、纯赛博城市、空泛口号、无人物交流等。
5. intent_keywords: 6 到 16 个短词，用于检索排序，不要无限扩展。
6. target_audience: 1 到 5 个受众或沟通对象。
7. selection_criteria: 3 到 8 条经验卡选择标准，描述什么样的卡片应该排在前面。
8. 不要输出具体场景、角色、镜头设计、故事段落或解释文字。"""


def build_intent_user_prompt(input_text: str) -> str:
    return f"""请解析以下创作关键词背后的核心检索意图，用于经验卡排序。

input_text: {input_text}

请只输出 JSON：
{{
  "primary_intent": "...",
  "must_match": ["..."],
  "nice_to_have": ["..."],
  "avoid": ["..."],
  "intent_keywords": ["..."],
  "target_audience": ["..."],
  "selection_criteria": ["..."]
}}"""


def _intent_expanded_text(raw: dict) -> str:
    parts = [str(raw.get("primary_intent", ""))]
    for field in (
        "must_match",
        "nice_to_have",
        "intent_keywords",
        "target_audience",
        "selection_criteria",
    ):
        values = raw.get(field)
        if isinstance(values, list):
            parts.extend(str(value) for value in values)
    return " ".join(part for part in parts if part)


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)
