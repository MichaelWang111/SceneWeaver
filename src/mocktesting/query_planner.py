from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mocktesting.embedding_text_builder import build_query_channels, target_channel_for_query
from mocktesting.eval_input_generator import PURPOSE_WORDS, STAGE_WORDS, canonical_stage
from sceneweaver.retrieval.query_plan import build_query_plan
from sceneweaver.retrieval.style import NEGATIVE_STYLE_ALIASES, POSITIVE_STYLE_ALIASES

PLANNER_VERSION = 1
VALID_QUERY_PLANNERS = (
    "rule",
    "multi_query",
    "hyde_card",
    "llm_structured",
    "llm_multi_query",
    "llm_hyde_card",
)
LLM_QUERY_PLANNERS = {"llm_structured", "llm_multi_query", "llm_hyde_card"}
DEFAULT_PLANNER_CACHE_NAME = "mock_query_planner_cache.jsonl"
NEGATIVE_SPAN_RE = re.compile(r"(不要做成|不要|避免|不想|不是|别|拒绝)([^，。；,;]*)")


@dataclass
class ExperimentalQueryPlan:
    planner: str
    original_text: str
    positive_query: str
    desired_stage: list[str] = field(default_factory=list)
    forbidden_stage: list[str] = field(default_factory=list)
    positive_purposes: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)
    visual_hints: list[str] = field(default_factory=list)
    positive_style: list[str] = field(default_factory=list)
    negative_style: list[str] = field(default_factory=list)
    scene_signature: dict[str, Any] = field(default_factory=dict)
    hard_constraints: list[str] = field(default_factory=list)
    soft_constraints: list[str] = field(default_factory=list)
    ambiguity: dict[str, Any] = field(default_factory=dict)
    rewrites: list[str] = field(default_factory=list)
    hyde_text: str = ""
    confidence: float = 0.35
    planner_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentalQueryPlan":
        return cls(
            planner=str(data.get("planner", "rule")),
            original_text=str(data.get("original_text", "")),
            positive_query=str(data.get("positive_query", "")),
            desired_stage=_string_list(data.get("desired_stage")),
            forbidden_stage=_string_list(data.get("forbidden_stage")),
            positive_purposes=_string_list(data.get("positive_purposes")),
            negative_constraints=_string_list(data.get("negative_constraints")),
            visual_hints=_string_list(data.get("visual_hints")),
            positive_style=_string_list(data.get("positive_style")),
            negative_style=_string_list(data.get("negative_style")),
            scene_signature=_dict_value(data.get("scene_signature")),
            hard_constraints=_string_list(data.get("hard_constraints")),
            soft_constraints=_string_list(data.get("soft_constraints")),
            ambiguity=_dict_value(data.get("ambiguity")),
            rewrites=_string_list(data.get("rewrites")),
            hyde_text=str(data.get("hyde_text", "")),
            confidence=_bounded_float(data.get("confidence"), default=0.35),
            planner_metadata=_dict_value(data.get("planner_metadata")),
        )

    def semantic_texts(self, *, max_texts: int = 0) -> list[str]:
        texts = _dedupe([self.positive_query, *self.rewrites, self.hyde_text])
        if max_texts > 0:
            return texts[:max_texts]
        return texts


@dataclass
class PlannerBatchResult:
    plans: list[ExperimentalQueryPlan]
    stats: dict[str, Any]


class QueryPlannerCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._rows: dict[str, dict[str, Any]] | None = None

    def get(self, planner: str, user_input: str) -> ExperimentalQueryPlan | None:
        if self.path is None:
            return None
        rows = self._load()
        row = rows.get(cache_key(planner, user_input))
        if not row:
            return None
        try:
            plan = ExperimentalQueryPlan.from_dict(row["plan"])
        except Exception:
            return None
        plan.planner_metadata = {
            **plan.planner_metadata,
            "cache_hit": True,
        }
        return plan

    def set(self, planner: str, user_input: str, plan: ExperimentalQueryPlan) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": cache_key(planner, user_input),
            "planner": planner,
            "version": PLANNER_VERSION,
            "input_hash": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
            "plan": plan.to_dict(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if self._rows is not None:
            self._rows[row["key"]] = row

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        rows: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("key", ""))
                if key:
                    rows[key] = row
        self._rows = rows
        return rows


def cache_key(planner: str, user_input: str) -> str:
    payload = json.dumps(
        {"version": PLANNER_VERSION, "planner": planner, "user_input": user_input},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_queries(
    user_inputs: list[str],
    *,
    planner: str = "rule",
    cache_path: Path | None = None,
    llm_sample_size: int = 0,
    timeout_seconds: float = 60.0,
    retries: int = 0,
    require_llm: bool = False,
    llm_client: Any | None = None,
    include_debug: bool = False,
) -> PlannerBatchResult:
    if planner not in VALID_QUERY_PLANNERS:
        raise ValueError(f"Unknown query planner: {planner}")
    cache = QueryPlannerCache(cache_path)
    stats: dict[str, Any] = {
        "planner": planner,
        "input_count": len(user_inputs),
        "cache_hits": 0,
        "cache_misses": 0,
        "llm_call_count": 0,
        "fallback_count": 0,
        "planner_errors": [],
        "negative_leak_count": 0,
    }
    plans: list[ExperimentalQueryPlan] = []
    remaining_llm_calls = max(0, llm_sample_size)
    for user_input in user_inputs:
        plan: ExperimentalQueryPlan | None = None
        if planner in LLM_QUERY_PLANNERS:
            plan = cache.get(planner, user_input)
            if plan is not None:
                stats["cache_hits"] += 1
            else:
                stats["cache_misses"] += 1
                if remaining_llm_calls > 0:
                    remaining_llm_calls -= 1
                    stats["llm_call_count"] += 1
                    try:
                        plan = llm_plan_query(
                            user_input,
                            planner=planner,
                            llm_client=llm_client,
                            timeout_seconds=timeout_seconds,
                            retries=retries,
                            include_debug=include_debug,
                        )
                        cache.set(planner, user_input, plan)
                    except Exception as exc:
                        if require_llm:
                            raise
                        stats["fallback_count"] += 1
                        stats["planner_errors"].append({"input": user_input, "error": str(exc)})
                        plan = fallback_plan_query(user_input, requested_planner=planner, error=str(exc))
                elif require_llm:
                    raise RuntimeError(
                        f"LLM planner {planner!r} requires a cached plan or llm_sample_size > 0."
                    )
                else:
                    stats["fallback_count"] += 1
                    plan = fallback_plan_query(user_input, requested_planner=planner, error="llm_sample_size_exhausted")
        else:
            plan = local_plan_query(user_input, planner=planner)

        plan = sanitize_plan(plan)
        if plan_has_negative_leak(plan):
            stats["negative_leak_count"] += 1
        plans.append(plan)
    stats["cache_hit_rate"] = round(stats["cache_hits"] / max(1, stats["cache_hits"] + stats["cache_misses"]), 6)
    stats["negative_leak_rate"] = round(stats["negative_leak_count"] / max(1, len(plans)), 6)
    return PlannerBatchResult(plans=plans, stats=stats)


def local_plan_query(user_input: str, *, planner: str) -> ExperimentalQueryPlan:
    if planner == "rule":
        return rule_plan_query(user_input)
    if planner == "multi_query":
        plan = rule_plan_query(user_input)
        plan.planner = planner
        plan.rewrites = generate_rewrites(plan)
        plan.confidence = max(plan.confidence, 0.55)
        plan.planner_metadata["query_text_count"] = len(plan.semantic_texts())
        return plan
    if planner == "hyde_card":
        plan = rule_plan_query(user_input)
        plan.planner = planner
        plan.hyde_text = generate_hyde_text(plan)
        plan.confidence = max(plan.confidence, 0.55)
        plan.planner_metadata["query_text_count"] = len(plan.semantic_texts())
        return plan
    raise ValueError(f"Unsupported local planner: {planner}")


def rule_plan_query(user_input: str) -> ExperimentalQueryPlan:
    base = build_query_plan(user_input)
    confidence = 0.75 if base.desired_stage or base.positive_purposes or base.positive_style else 0.4
    plan = ExperimentalQueryPlan(
        planner="rule",
        original_text=base.original_text,
        positive_query=base.positive_query,
        desired_stage=list(base.desired_stage),
        forbidden_stage=list(base.forbidden_stage),
        positive_purposes=list(base.positive_purposes),
        negative_constraints=list(base.negative_constraints),
        visual_hints=list(base.visual_hints),
        positive_style=list(base.positive_style),
        negative_style=list(base.negative_style),
        scene_signature=rule_scene_signature(base.positive_query, list(base.visual_hints), list(base.positive_purposes)),
        hard_constraints=[f"forbidden_stage:{stage}" for stage in base.forbidden_stage],
        soft_constraints=[f"negative_style:{style}" for style in base.negative_style],
        ambiguity=estimate_ambiguity(base.original_text, list(base.desired_stage), list(base.positive_purposes)),
        confidence=confidence,
        planner_metadata={"source": "sceneweaver.retrieval.build_query_plan"},
    )
    return sanitize_plan(plan)


def fallback_plan_query(user_input: str, *, requested_planner: str, error: str) -> ExperimentalQueryPlan:
    plan = rule_plan_query(user_input)
    plan.planner = requested_planner
    plan.planner_metadata = {
        **plan.planner_metadata,
        "fallback_to": "rule",
        "fallback_reason": error,
    }
    return plan


def llm_plan_query(
    user_input: str,
    *,
    planner: str,
    llm_client: Any | None = None,
    timeout_seconds: float,
    retries: int,
    include_debug: bool,
) -> ExperimentalQueryPlan:
    client = llm_client
    if client is None:
        from sceneweaver.llm.client import VisionLLMClient

        client = VisionLLMClient()
    payload = {
        "task": "把用户检索需求解析为结构化 Query Understanding。只输出 JSON。",
        "rules": [
            "不要把用户否定的对象放进 positive_query。",
            "明确不要/避免/别/拒绝的内容进入 forbidden_stage、negative_style 或 soft_constraints。",
            "模糊表达可以进入 ambiguity，并给出低 confidence。",
            "rewrites 是正向检索改写，不包含负向约束。",
            "hyde_text 是理想经验卡或导演笔记，不包含负向约束。",
        ],
        "allowed_stage": sorted(set(STAGE_WORDS) | {"general"}),
        "allowed_positive_style": sorted(POSITIVE_STYLE_ALIASES),
        "allowed_negative_style": sorted(NEGATIVE_STYLE_ALIASES),
        "output_schema": {
            "positive_query": "string",
            "desired_stage": ["string"],
            "forbidden_stage": ["string"],
            "positive_purposes": ["string"],
            "negative_constraints": ["string"],
            "visual_hints": ["string"],
            "positive_style": ["string"],
            "negative_style": ["string"],
            "scene_signature": {
                "people": ["string"],
                "place": ["string"],
                "actions": ["string"],
                "objects": ["string"],
                "emotional_function": "string",
            },
            "hard_constraints": ["string"],
            "soft_constraints": ["string"],
            "ambiguity": {"level": "low|medium|high", "reasons": ["string"]},
            "rewrites": ["string"],
            "hyde_text": "string",
            "confidence": "0..1 number",
        },
        "planner": planner,
        "user_input": user_input,
    }
    response = client.analyze_text_json(
        system_prompt="你是企业视频检索 Query Understanding 模块。只输出 JSON。",
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=900,
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    plan = plan_from_llm_response(user_input, planner=planner, response=response)
    plan.planner_metadata = {
        **plan.planner_metadata,
        "llm_used": True,
    }
    if include_debug:
        plan.planner_metadata["llm_raw_response"] = response
    return plan


def plan_from_llm_response(user_input: str, *, planner: str, response: dict[str, Any]) -> ExperimentalQueryPlan:
    base = rule_plan_query(user_input)
    positive_query = str(response.get("positive_query") or base.positive_query or user_input)
    rewrites = _string_list(response.get("rewrites"))
    hyde_text = str(response.get("hyde_text", ""))
    if planner == "llm_structured":
        rewrites = []
        hyde_text = ""
    elif planner == "llm_multi_query" and not rewrites:
        rewrites = generate_rewrites(base)
    elif planner == "llm_hyde_card" and not hyde_text:
        hyde_text = generate_hyde_text(base)
    plan = ExperimentalQueryPlan(
        planner=planner,
        original_text=user_input,
        positive_query=positive_query,
        desired_stage=_valid_stages(_string_list(response.get("desired_stage"))) or base.desired_stage,
        forbidden_stage=_valid_stages(_string_list(response.get("forbidden_stage"))) or base.forbidden_stage,
        positive_purposes=_dedupe([*_string_list(response.get("positive_purposes")), *base.positive_purposes]),
        negative_constraints=_dedupe([*_string_list(response.get("negative_constraints")), *base.negative_constraints]),
        visual_hints=_dedupe([*_string_list(response.get("visual_hints")), *base.visual_hints]),
        positive_style=_valid_styles(_string_list(response.get("positive_style")), positive=True) or base.positive_style,
        negative_style=_valid_styles(_string_list(response.get("negative_style")), positive=False) or base.negative_style,
        scene_signature=_dict_value(response.get("scene_signature")) or base.scene_signature,
        hard_constraints=_string_list(response.get("hard_constraints")) or base.hard_constraints,
        soft_constraints=_string_list(response.get("soft_constraints")) or base.soft_constraints,
        ambiguity=_dict_value(response.get("ambiguity")) or base.ambiguity,
        rewrites=rewrites,
        hyde_text=hyde_text,
        confidence=_bounded_float(response.get("confidence"), default=base.confidence),
        planner_metadata={"source": "llm_structured_json"},
    )
    return sanitize_plan(plan)


def sanitize_plan(plan: ExperimentalQueryPlan) -> ExperimentalQueryPlan:
    base = rule_plan_query(plan.original_text) if plan.planner != "rule" else None
    negative_terms = negative_terms_for_plan(plan)
    plan.positive_query = sanitize_positive_text(plan.positive_query or plan.original_text, negative_terms)
    if not plan.positive_query and base is not None:
        plan.positive_query = base.positive_query
    plan.desired_stage = _dedupe([canonical_stage(stage) for stage in plan.desired_stage if stage])
    plan.forbidden_stage = _dedupe([canonical_stage(stage) for stage in plan.forbidden_stage if stage])
    plan.positive_purposes = _dedupe(plan.positive_purposes)
    plan.negative_constraints = _dedupe(plan.negative_constraints)
    plan.visual_hints = _dedupe(plan.visual_hints)
    plan.positive_style = _valid_styles(plan.positive_style, positive=True)
    plan.negative_style = _valid_styles(plan.negative_style, positive=False)
    plan.rewrites = [
        text
        for text in _dedupe(sanitize_positive_text(text, negative_terms) for text in plan.rewrites)
        if text and text != plan.positive_query
    ][:5]
    plan.hyde_text = sanitize_positive_text(plan.hyde_text, negative_terms)
    plan.confidence = _bounded_float(plan.confidence, default=0.35)
    plan.hard_constraints = _dedupe(plan.hard_constraints)
    plan.soft_constraints = _dedupe(plan.soft_constraints)
    return plan


def build_query_channels_for_plan(plan: ExperimentalQueryPlan, *, max_query_texts: int = 0) -> list[dict[str, Any]]:
    texts = plan.semantic_texts(max_texts=max_query_texts) or [plan.original_text]
    query_count = max(1, len(texts))
    channels: list[dict[str, Any]] = []
    for text_index, text in enumerate(texts):
        for channel in build_query_channels(text):
            target_channel = channel.get("target_channel") or target_channel_for_query(channel["channel"])
            row = dict(channel)
            row["target_channel"] = target_channel
            if query_count > 1:
                row["channel"] = f"{channel['channel']}__q{text_index}"
            row["weight"] = float(row.get("weight", 0.0)) / query_count
            channels.append(row)
    return channels


def planner_constraints(plan: ExperimentalQueryPlan) -> dict[str, Any]:
    return {
        "desired_stage": list(plan.desired_stage),
        "forbidden_stage": list(plan.forbidden_stage),
        "negative_constraints": list(plan.negative_constraints),
        "visual_hints": list(plan.visual_hints),
    }


def plan_has_negative_leak(plan: ExperimentalQueryPlan) -> bool:
    negative_terms = [term.lower() for term in negative_terms_for_plan(plan) if term]
    if not negative_terms:
        return False
    texts = [plan.positive_query, *plan.rewrites, plan.hyde_text]
    return any(term in text.lower() for term in negative_terms for text in texts if text)


def negative_terms_for_plan(plan: ExperimentalQueryPlan) -> list[str]:
    terms: list[str] = []
    for _prefix, segment in NEGATIVE_SPAN_RE.findall(plan.original_text):
        clean = segment.strip()
        if clean:
            terms.append(clean)
    terms.extend(plan.negative_constraints)
    for style in plan.negative_style:
        terms.extend(NEGATIVE_STYLE_ALIASES.get(style, (style,)))
    return _dedupe([term.strip() for term in terms if term and len(term.strip()) > 1])


def sanitize_positive_text(text: str, negative_terms: list[str]) -> str:
    clean = NEGATIVE_SPAN_RE.sub("", str(text or ""))
    for term in sorted(set(negative_terms), key=len, reverse=True):
        if term:
            clean = clean.replace(term, "")
    return re.sub(r"\s+", " ", clean).strip(" ，,。；;")


def generate_rewrites(plan: ExperimentalQueryPlan) -> list[str]:
    stage = stage_text(plan.desired_stage)
    purposes = purpose_text(plan.positive_purposes)
    style = style_text(plan.positive_style)
    visual = "、".join(plan.visual_hints[:3])
    candidates = [
        plan.positive_query,
        f"需要一个{stage}段落，完成{purposes}，导演经验要服务脚本用途。",
        f"检索适合{stage}的经验卡，重点是{purposes}。",
        f"镜头应承担{stage}功能，情绪和信息都指向{purposes}。",
    ]
    if style:
        candidates.append(f"风格偏{style}，但核心仍是{stage}和{purposes}。")
    if visual:
        candidates.append(f"画面可参考{visual}，但排序优先看{stage}和{purposes}。")
    return [text for text in _dedupe(candidates) if text and text != plan.positive_query][:5]


def generate_hyde_text(plan: ExperimentalQueryPlan) -> str:
    stage = stage_text(plan.desired_stage)
    purposes = purpose_text(plan.positive_purposes)
    style = style_text(plan.positive_style) or "克制、真实、服务叙事"
    signature = signature_text(plan.scene_signature)
    return (
        f"理想经验卡：适合用于{stage}。脚本用途是{purposes}。"
        f"导演经验：画面和剪辑不抢戏，先让观众理解这一段为什么存在；"
        f"风格应当{style}。场景线索：{signature or plan.positive_query}。"
    )


def rule_scene_signature(positive_query: str, visual_hints: list[str], purposes: list[str]) -> dict[str, Any]:
    return {
        "people": [],
        "place": visual_hints[:2],
        "actions": [],
        "objects": visual_hints[2:5],
        "emotional_function": purpose_text(purposes),
        "raw_positive_query": positive_query,
    }


def estimate_ambiguity(text: str, desired_stage: list[str], purposes: list[str]) -> dict[str, Any]:
    reasons: list[str] = []
    if not desired_stage:
        reasons.append("missing_explicit_stage")
    if not purposes:
        reasons.append("missing_explicit_purpose")
    fuzzy_terms = ("高级", "有温度", "真实一点", "别端着", "现场感", "自然")
    if any(term in text for term in fuzzy_terms):
        reasons.append("fuzzy_style_language")
    level = "high" if len(reasons) >= 2 else "medium" if reasons else "low"
    return {"level": level, "reasons": reasons}


def stage_text(stages: list[str]) -> str:
    if not stages:
        return "合适的脚本阶段"
    return "、".join(STAGE_WORDS.get(stage, stage.replace("_", " ")) for stage in stages[:2])


def purpose_text(purposes: list[str]) -> str:
    if not purposes:
        return "明确的创作目的"
    return "、".join(PURPOSE_WORDS.get(purpose, purpose.replace("_", " ")) for purpose in purposes[:3])


def style_text(styles: list[str]) -> str:
    labels = {
        "human_warmth": "有人味",
        "documentary": "纪录片观察",
        "real_location": "真实现场",
    }
    return "、".join(labels.get(style, style.replace("_", " ")) for style in styles)


def signature_text(signature: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("people", "place", "actions", "objects", "emotional_function"):
        value = signature.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif value:
            parts.append(str(value))
    return "、".join(_dedupe(parts))


def _valid_stages(values: list[str]) -> list[str]:
    valid = set(STAGE_WORDS) | {"general"}
    result = []
    for value in values:
        stage = canonical_stage(value)
        if stage in valid and stage not in result:
            result.append(stage)
    return result


def _valid_styles(values: list[str], *, positive: bool) -> list[str]:
    valid = set(POSITIVE_STYLE_ALIASES if positive else NEGATIVE_STYLE_ALIASES)
    return _dedupe([value for value in values if value in valid])


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _dedupe(values: list[str] | Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result
