from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Callable, Protocol

from sceneweaver.analysis.tags import build_query_tags
from sceneweaver.llm.client import VisionLLMClient
from sceneweaver.schemas import AssociationAnalysis
from sceneweaver.storage.json_store import write_json

DEFAULT_MAX_ITEMS = 72
DEFAULT_TEXT_MAX_TOKENS = 9000
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_RETRIES = 1
LogFn = Callable[[str], None]


class AssociateLLMClient(Protocol):
    def analyze_text_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        retries: int = 0,
        stream_callback: Callable[[str], None] | None = None,
        reasoning_callback: Callable[[str], None] | None = None,
        enable_thinking: bool | None = None,
        thinking_budget: int | None = None,
    ) -> dict:
        ...


def associate_input(
    input_text: str,
    *,
    client: AssociateLLMClient | None = None,
    prompt_path: Path | None = None,
    output_path: Path | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    log: LogFn | None = None,
    stream_callback: Callable[[str], None] | None = None,
    reasoning_callback: Callable[[str], None] | None = None,
    enable_thinking: bool | None = None,
    thinking_budget: int | None = None,
) -> AssociationAnalysis:
    started_at = perf_counter()
    clean_input = input_text.strip()
    if not clean_input:
        raise ValueError("input_text cannot be empty")
    if not 8 <= max_items <= 120:
        raise ValueError("max_items must be between 8 and 120")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if retries < 0:
        raise ValueError("retries must be >= 0")

    _log(log, f"Loading associate prompt: {prompt_path or 'prompts/associate.md'}")
    system_prompt = load_associate_prompt(prompt_path)
    _log(log, f"Building associate request: input_chars={len(clean_input)}, max_items={max_items}")
    user_prompt = build_associate_user_prompt(clean_input, max_items=max_items)
    llm_client = client or VisionLLMClient()
    _log(
        log,
        "Submitting LLM request: "
        f"max_tokens={DEFAULT_TEXT_MAX_TOKENS}, timeout_seconds={timeout_seconds:g}, retries={retries}",
    )
    raw = _call_llm_with_heartbeat(
        client=llm_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=DEFAULT_TEXT_MAX_TOKENS,
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
    _log(log, f"LLM response received after {perf_counter() - started_at:.1f}s. Validating JSON schema.")
    candidate_log_path = output_path.parent / "tag_candidates.jsonl" if output_path is not None else None
    raw["query_tags"] = build_query_tags(
        clean_input,
        extra_text=_association_tag_text(raw),
        candidate_log_path=candidate_log_path,
    ).model_dump(mode="json")
    analysis = AssociationAnalysis.model_validate(raw)
    if output_path is not None:
        _log(log, f"Writing association JSON: {output_path}")
        write_json(output_path, analysis)
    _log(log, f"Associate complete: association_count={analysis.association_count}")
    return analysis


def load_associate_prompt(prompt_path: Path | None = None) -> str:
    path = prompt_path or Path(__file__).resolve().parents[3] / "prompts" / "associate.md"
    return path.read_text(encoding="utf-8")


def build_associate_user_prompt(input_text: str, *, max_items: int = DEFAULT_MAX_ITEMS) -> str:
    return f"""请将以下输入扩展为导演/编剧可用的联想材料。

input_text: {input_text}
max_items: {max_items}

请根据 input_text 生成结构化 JSON。`association_count` 必须等于所有 association 条目的实际总数，并尽量接近 max_items。"""


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)


def _call_llm_with_heartbeat(
    *,
    client: AssociateLLMClient,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None,
    timeout_seconds: float,
    retries: int,
    stream_callback: Callable[[str], None] | None,
    reasoning_callback: Callable[[str], None] | None,
    enable_thinking: bool | None,
    thinking_budget: int | None,
    log: LogFn | None,
    started_at: float,
) -> dict:
    stop_heartbeat = Event()
    heartbeat_thread: Thread | None = None
    active_stream_callback = stream_callback
    active_reasoning_callback = reasoning_callback

    if stream_callback is not None:
        def mark_stream_started(chunk: str) -> None:
            stop_heartbeat.set()
            stream_callback(chunk)

        active_stream_callback = mark_stream_started
    if reasoning_callback is not None:
        def mark_reasoning_started(chunk: str) -> None:
            stop_heartbeat.set()
            reasoning_callback(chunk)

        active_reasoning_callback = mark_reasoning_started

    if log is not None:
        if stream_callback is not None or reasoning_callback is not None:
            _log(log, "Waiting for provider stream. First chunk can take tens of seconds on long JSON requests.")

        def heartbeat() -> None:
            while not stop_heartbeat.wait(10):
                _log(log, f"Still waiting for LLM response... elapsed={perf_counter() - started_at:.1f}s")

        heartbeat_thread = Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

    try:
        return client.analyze_text_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
            stream_callback=active_stream_callback,
            reasoning_callback=active_reasoning_callback,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.2)


def _association_tag_text(raw: dict) -> str:
    parts: list[str] = []
    for key in ("core_reading", "avoid_cliches"):
        value = raw.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    emotional_arc = raw.get("emotional_arc")
    if isinstance(emotional_arc, dict):
        parts.extend(str(value) for value in emotional_arc.values())
    association_map = raw.get("association_map")
    if isinstance(association_map, dict):
        for items in association_map.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        parts.extend(
                            str(item.get(field, ""))
                            for field in ("term", "meaning", "emotion", "image_hint", "usage_hint")
                        )
    return " ".join(part for part in parts if part)
