from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
import hashlib
import json
import re
import time
from typing import Any

from retreieval_lab.artifacts import data_sha256, read_json, read_jsonl, write_json, write_jsonl
from retreieval_lab.schemas import SCHEMA_VERSION, QueryPlanModel, validate_record
from sceneweaver.retrieval.query_plan import build_query_plan


PLANNER_VERSION = "retrieval_lab_planner_v3"
DEFAULT_PLANNER_CACHE_PATH = Path(".tmp") / "retrieval_lab" / "planner_cache.jsonl"
DEFAULT_PLANNER_PLAN_OUTPUT = Path(".tmp") / "retrieval_lab" / "planner_plan.json"
DEFAULT_PLANNER_COMPARE_OUTPUT = Path(".tmp") / "retrieval_lab" / "planner_compare.json"
DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT = Path(".tmp") / "retrieval_lab" / "planner_cache_audit.json"

NEGATIVE_SPAN_RE = re.compile(
    r"(不要做成|不要|别|避免|不想|不是|拒绝|without|avoid|exclude|do not|don't|not|no)\s*([^,;，。；]*)",
    flags=re.IGNORECASE,
)

STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "opening": ("opening", "open", "开场", "开头", "起始", "第一幕"),
    "setup": ("setup", "铺垫", "需求建立", "背景建立", "问题建立", "前情", "进入现场", "现场"),
    "technology_showcase": (
        "technology_showcase",
        "technology showcase",
        "technology entrance",
        "技术展示",
        "技术入场",
        "科技展示",
        "产品展示",
        "功能展示",
        "技术炫耀",
    ),
    "team_work": ("team_work", "team work", "团队协作", "团队", "协作", "共创"),
    "value_expression": ("value_expression", "value expression", "价值表达", "建立信任", "品牌价值"),
    "outcome": ("outcome", "结果", "成果", "成效", "结果展示"),
    "ending": ("ending", "结尾", "收束", "号召", "结束"),
    "general": ("general", "通用"),
}

FORBIDDEN_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "opening": ("opening", "open", "开场", "开头", "第一幕"),
    "setup": ("setup", "铺垫", "需求铺垫"),
    "technology_showcase": (
        "technology_showcase",
        "technology showcase",
        "technology entrance",
        "技术展示",
        "技术入场",
        "科技展示",
        "产品展示",
        "功能展示",
        "技术炫耀",
    ),
    "team_work": ("team_work", "team work", "团队协作"),
    "value_expression": ("value_expression", "value expression", "价值表达", "品牌价值"),
    "outcome": ("outcome", "结果展示", "成果展示"),
    "ending": ("ending", "结尾", "收束", "号召", "结束"),
}

PURPOSE_ALIASES: dict[str, tuple[str, ...]] = {
    "establish_problem": ("建立问题", "问题建立", "现实压力", "痛点", "矛盾"),
    "establish_need": ("建立需求", "需求建立", "需求铺垫"),
    "build_reality": ("真实感", "真实现场", "现场感", "grounded", "reality"),
    "build_trust": ("建立信任", "可信", "信任", "trust"),
    "show_technology": ("表现技术", "技术能力", "科技能力", "technology"),
    "show_team": ("表现团队", "团队能力", "team"),
    "express_value": ("表达价值", "价值表达", "品牌价值", "value"),
    "keep_human_warmth": ("人的温度", "有人味", "human warmth", "human"),
}

POSITIVE_STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "human_warmth": ("有人味", "人味", "人的温度", "human warmth", "human-centered", "human centered"),
    "documentary": ("纪录片", "纪实", "观察", "documentary", "observational"),
    "real_location": ("真实现场", "真实场景", "现场感", "real location", "on location"),
}

NEGATIVE_STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "big_company_office": ("大厂味", "大厂", "互联网大厂", "泛泛办公", "空泛办公", "generic office"),
    "ad_like": ("广告感", "广告", "宣传片腔", "硬广", "口号", "slogan", "campaign", "ad-like", "advertising"),
    "tech_showoff": ("炫技", "技术炫耀", "技术堆砌", "功能说明", "产品说明", "纯科技", "tech showoff"),
    "product_pitch": ("产品卖点", "卖点堆叠", "product pitch", "sales pitch"),
    "corporate_report_tone": ("汇报片", "企业汇报", "corporate report"),
}

PLANNER_NAMES = ("rule", "multi_query", "hyde_card", "fake_llm", "legacy_adapter")


class PlannerCache:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._rows: dict[str, dict[str, Any]] | None = None

    def get(self, planner: str, user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if self.path is None:
            return None
        row = self._load().get(planner_cache_key(planner, user_input, config or {}))
        if not row:
            return None
        plan = row.get("plan")
        if not isinstance(plan, dict):
            return None
        plan = dict(plan)
        provenance = dict(plan.get("provenance", {}))
        provenance["cache_hit"] = True
        provenance["cache_key"] = row.get("key", "")
        plan["provenance"] = provenance
        return plan

    def set(self, planner: str, user_input: str, plan: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": planner_cache_key(planner, user_input, config or {}),
            "planner": planner,
            "schema_version": SCHEMA_VERSION,
            "planner_version": PLANNER_VERSION,
            "input_hash": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
            "config_fingerprint": data_sha256(config or {}),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "plan": plan,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if self._rows is not None:
            self._rows[row["key"]] = row

    def audit(self) -> dict[str, Any]:
        rows = list(self._load().values()) if self.path is not None else []
        planners = Counter(str(row.get("planner", "")) for row in rows)
        invalid_rows = 0
        for row in rows:
            plan = row.get("plan")
            if not isinstance(plan, dict) or not validate_record("query_plan", plan)["valid"]:
                invalid_rows += 1
        return {
            "path": str(self.path) if self.path else "",
            "exists": bool(self.path and self.path.exists()),
            "row_count": len(rows),
            "invalid_row_count": invalid_rows,
            "planner_counts": dict(sorted(planners.items())),
            "schema_version_counts": dict(sorted(Counter(str(row.get("schema_version", "")) for row in rows).items())),
        }

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        rows: dict[str, dict[str, Any]] = {}
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8-sig").splitlines():
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


def planner_cache_key(planner: str, user_input: str, config: dict[str, Any]) -> str:
    return data_sha256(
        {
            "planner": planner,
            "schema_version": SCHEMA_VERSION,
            "planner_version": PLANNER_VERSION,
            "input_hash": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
            "config_fingerprint": data_sha256(config),
        }
    )


def planner_registry() -> dict[str, Callable[[str, dict[str, Any]], dict[str, Any]]]:
    return {
        "rule": rule_plan,
        "multi_query": multi_query_plan,
        "hyde_card": hyde_card_plan,
        "fake_llm": fake_llm_plan,
        "legacy_adapter": legacy_adapter_plan,
    }


def plan_many(
    user_inputs: list[str],
    *,
    planner: str = "rule",
    cache_path: Path | None = None,
    use_cache: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if planner not in planner_registry():
        raise ValueError(f"unknown planner: {planner}")
    cache = PlannerCache(cache_path if use_cache else None)
    plans = []
    cache_hits = 0
    cache_misses = 0
    invalid_count = 0
    negative_leak_count = 0
    started = time.perf_counter()
    for user_input in user_inputs:
        plan = cache.get(planner, user_input, config or {})
        if plan is not None:
            cache_hits += 1
        else:
            cache_misses += 1
            plan = planner_registry()[planner](user_input, config or {})
            plan = normalize_plan(plan)
            cache.set(planner, user_input, plan, config or {})
        validation = validate_record("query_plan", plan)
        if not validation["valid"]:
            invalid_count += 1
            plan = {**plan, "validation_errors": validation["errors"]}
        else:
            plan = validation["normalized"]
        if plan_has_negative_leak(plan):
            negative_leak_count += 1
        plans.append(plan)
    elapsed = round(time.perf_counter() - started, 6)
    return {
        "method": "retrieval_lab_planner_plan",
        "summary": {
            "planner": planner,
            "input_count": len(user_inputs),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": round(cache_hits / max(1, cache_hits + cache_misses), 6),
            "invalid_plan_count": invalid_count,
            "negative_leak_count": negative_leak_count,
            "negative_leak_rate": round(negative_leak_count / max(1, len(user_inputs)), 6),
            "elapsed_seconds": elapsed,
        },
        "plans": plans,
        "planner_registry": planner_metadata(),
    }


def compare_planners(
    user_inputs: list[str],
    *,
    planners: list[str],
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    runs: dict[str, dict[str, Any]] = {}
    for planner in planners:
        runs[planner] = plan_many(user_inputs, planner=planner, cache_path=cache_path, use_cache=use_cache)
    baseline_name = planners[0] if planners else "rule"
    baseline_plans = runs.get(baseline_name, {}).get("plans", [])
    rows = []
    for index, user_input in enumerate(user_inputs):
        baseline = baseline_plans[index] if index < len(baseline_plans) else {}
        row = {"query_index": index, "user_input": user_input, "baseline": baseline_name, "planner_diffs": {}}
        for planner in planners:
            plan = runs[planner]["plans"][index]
            row["planner_diffs"][planner] = planner_diff(baseline, plan)
        rows.append(row)
    metrics = {planner: run["summary"] for planner, run in runs.items()}
    elapsed = round(time.perf_counter() - started, 6)
    return {
        "method": "retrieval_lab_planner_compare",
        "summary": {
            "planner_count": len(planners),
            "input_count": len(user_inputs),
            "baseline_planner": baseline_name,
            "best_planner": select_best_planner(metrics),
            "elapsed_seconds": elapsed,
        },
        "planner_metrics": metrics,
        "comparisons": rows,
        "planner_registry": planner_metadata(),
    }


def rule_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = build_query_plan(user_input)
    positive_text = sanitize_positive_text(base.positive_query or user_input, user_input)
    desired_stage = merge_unique([*base.desired_stage, *parse_positive_stages(positive_text)])
    forbidden_stage = parse_forbidden_stages(user_input)
    positive_style = merge_unique([*base.positive_style, *parse_styles(positive_text, POSITIVE_STYLE_ALIASES)])
    negative_style = merge_unique([*base.negative_style, *parse_negative_styles_from_query(user_input)])
    positive_purposes = merge_unique([*base.positive_purposes, *parse_purposes(positive_text)])
    negative_constraints = merge_unique([*base.negative_constraints, *negative_segments(user_input)])
    plan = {
        "planner": "rule",
        "original_text": str(base.original_text or user_input),
        "positive_query": positive_text or user_input,
        "desired_stage": desired_stage,
        "forbidden_stage": forbidden_stage,
        "positive_purposes": positive_purposes,
        "negative_constraints": negative_constraints,
        "visual_hints": list(base.visual_hints),
        "positive_style": positive_style,
        "negative_style": negative_style,
        "scene_signature": scene_signature(positive_text, desired_stage, positive_purposes, positive_style),
        "hard_constraints": [
            constraint_row("stage", "must_not", [stage], "hard", reason="explicit forbidden stage")
            for stage in forbidden_stage
        ],
        "soft_constraints": [
            constraint_row("style", "should_not", [style], "soft", reason="explicit negative style")
            for style in negative_style
        ],
        "ambiguity": ambiguity(positive_text, desired_stage, positive_purposes),
        "rewrites": [],
        "hyde_text": "",
        "confidence": confidence_for_plan(desired_stage, positive_purposes, positive_style),
        "provenance": {
            "planner": "rule",
            "planner_version": PLANNER_VERSION,
            "source": "rule",
            "llm_used": False,
        },
        "planner_metadata": {"base_source": "sceneweaver.retrieval.build_query_plan"},
    }
    return plan


def multi_query_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = rule_plan(user_input, config)
    plan["planner"] = "multi_query"
    plan["rewrites"] = rewrite_rows(generate_rewrites(plan))
    plan["confidence"] = max(float(plan.get("confidence", 0.35)), 0.55)
    plan["provenance"] = {**plan["provenance"], "planner": "multi_query", "source": "multi_query"}
    return plan


def hyde_card_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = rule_plan(user_input, config)
    plan["planner"] = "hyde_card"
    plan["hyde_text"] = sanitize_positive_text(generate_hyde_text(plan), user_input)
    plan["confidence"] = max(float(plan.get("confidence", 0.35)), 0.55)
    plan["provenance"] = {**plan["provenance"], "planner": "hyde_card", "source": "hyde_card"}
    return plan


def fake_llm_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = multi_query_plan(user_input, config)
    plan["planner"] = "fake_llm"
    plan["hyde_text"] = sanitize_positive_text(generate_hyde_text(plan), user_input)
    signature = dict(plan.get("scene_signature", {}))
    signature["confidence"] = max(float(signature.get("confidence", 0.5)), 0.75)
    signature.setdefault("evidence", []).append(
        {"source": "fake_llm", "field": "planner", "text": "deterministic local fake judgement", "confidence": 0.75}
    )
    plan["scene_signature"] = signature
    plan["confidence"] = max(float(plan.get("confidence", 0.35)), 0.7)
    plan["provenance"] = {
        **plan["provenance"],
        "planner": "fake_llm",
        "source": "system",
        "llm_used": False,
    }
    plan["planner_metadata"] = {**plan.get("planner_metadata", {}), "fake_llm": True}
    return plan


def legacy_adapter_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    from mocktesting.query_planner import plan_queries

    result = plan_queries([user_input], planner="rule", cache_path=None)
    legacy_plan = result.plans[0].to_dict()
    plan = {
        **legacy_plan,
        "planner": "legacy_adapter",
        "scene_signature": legacy_plan.get("scene_signature") or scene_signature(
            legacy_plan.get("positive_query", user_input),
            legacy_plan.get("desired_stage", []),
            legacy_plan.get("positive_purposes", []),
            legacy_plan.get("positive_style", []),
        ),
        "provenance": {
            "planner": "legacy_adapter",
            "planner_version": PLANNER_VERSION,
            "source": "legacy_adapter",
            "llm_used": False,
        },
        "planner_metadata": {
            **legacy_plan.get("planner_metadata", {}),
            "legacy_source": "mocktesting.query_planner.plan_queries",
        },
    }
    return plan


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan = apply_forbidden_precedence(plan)
    validation = validate_record("query_plan", plan)
    if validation["valid"]:
        return validation["normalized"]
    cleaned = dict(plan)
    cleaned = apply_forbidden_precedence(cleaned)
    cleaned["positive_query"] = sanitize_positive_text_for_plan(
        str(cleaned.get("positive_query") or cleaned.get("original_text", "")),
        cleaned,
    )
    if not cleaned["positive_query"]:
        cleaned["positive_query"] = "general scene"
    cleaned["rewrites"] = [
        row
        for row in rewrite_rows(
            sanitize_positive_text_for_plan(row.get("text", ""), cleaned)
            for row in cleaned.get("rewrites", [])
            if isinstance(row, dict)
        )
        if row.get("text") and row.get("text") != cleaned["positive_query"]
    ]
    cleaned["hyde_text"] = sanitize_positive_text_for_plan(str(cleaned.get("hyde_text", "")), cleaned)
    return QueryPlanModel.model_validate(cleaned).model_dump(mode="json", exclude_none=True)


def load_queries(
    *,
    queries: list[str] | None = None,
    input_file: Path | None = None,
    dataset_path: Path | None = None,
    split: str = "test",
    limit: int = 0,
) -> list[str]:
    values = [query for query in queries or [] if query]
    if input_file:
        values.extend(load_queries_from_file(input_file))
    if dataset_path:
        from retreieval_lab.datasets import read_cases

        values.extend(str(case.get("user_input", "")) for case in read_cases(dataset_path, split=split, limit=limit))
    return [value for value in values if value]


def load_queries_from_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
        return [query_from_record(row) for row in rows if query_from_record(row)]
    data = read_json(path)
    if isinstance(data, list):
        return [query_from_record(row) for row in data if query_from_record(row)]
    if isinstance(data, dict):
        if isinstance(data.get("cases"), list):
            return [query_from_record(row) for row in data["cases"] if query_from_record(row)]
        if isinstance(data.get("queries"), list):
            return [query_from_record(row) for row in data["queries"] if query_from_record(row)]
        value = query_from_record(data)
        return [value] if value else []
    if isinstance(data, str):
        return [data]
    return []


def query_from_record(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        for key in ("user_input", "query", "text", "original_text"):
            value = row.get(key)
            if value:
                return str(value)
    return ""


def audit_cache(path: Path | None = DEFAULT_PLANNER_CACHE_PATH) -> dict[str, Any]:
    cache = PlannerCache(path)
    audit = cache.audit()
    return {
        "method": "retrieval_lab_planner_cache_audit",
        "summary": audit,
        "cache": audit,
    }


def write_planner_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, {**report, "fingerprint": data_sha256(report)})


def write_plans_jsonl(path: Path, plans: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, plans)


def planner_metadata() -> dict[str, Any]:
    return {
        name: {
            "native": name != "legacy_adapter",
            "uses_real_llm": False,
            "schema": "query_plan",
            "version": PLANNER_VERSION,
        }
        for name in PLANNER_NAMES
    }


def planner_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = ("desired_stage", "forbidden_stage", "positive_style", "negative_style", "positive_purposes")
    changed = {}
    for field in fields:
        if left.get(field, []) != right.get(field, []):
            changed[field] = {"baseline": left.get(field, []), "candidate": right.get(field, [])}
    return {
        "changed_field_count": len(changed),
        "changed_fields": changed,
        "confidence_delta": round(float(right.get("confidence", 0.0)) - float(left.get("confidence", 0.0)), 6),
        "negative_leak": plan_has_negative_leak(right),
    }


def select_best_planner(metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics:
        return ""
    return sorted(
        metrics,
        key=lambda name: (
            -float(metrics[name].get("negative_leak_rate", 0.0)),
            -float(metrics[name].get("invalid_plan_count", 0.0)),
            float(metrics[name].get("cache_hit_rate", 0.0)),
            -float(metrics[name].get("elapsed_seconds", 0.0)),
        ),
        reverse=True,
    )[0]


def parse_positive_stages(text: str) -> list[str]:
    return stage_mentions(text)


def parse_forbidden_stages(text: str) -> list[str]:
    stages: list[str] = []
    for segment in negative_segments(text):
        stages.extend(stage_mentions(segment, aliases=FORBIDDEN_STAGE_ALIASES))
    return [stage for stage in merge_unique(stages) if stage != "general"]


def stage_mentions(text: str, *, aliases: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    lower = str(text or "").lower()
    stages = []
    active_aliases = aliases or STAGE_ALIASES
    for stage, stage_aliases in active_aliases.items():
        if any(alias.lower() in lower for alias in stage_aliases):
            stages.append(stage)
    return merge_unique(stages)


def parse_purposes(text: str) -> list[str]:
    lower = str(text or "").lower()
    return [purpose for purpose, aliases in PURPOSE_ALIASES.items() if any(alias.lower() in lower for alias in aliases)]


def parse_styles(text: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    lower = str(text or "").lower()
    return [style for style, terms in aliases.items() if any(term.lower() in lower for term in terms)]


def parse_negative_styles_from_query(text: str) -> list[str]:
    hits: list[str] = []
    for segment in negative_segments(text):
        hits.extend(parse_styles(segment, NEGATIVE_STYLE_ALIASES))
    return merge_unique(hits)


def negative_segments(text: str) -> list[str]:
    segments = [segment.strip() for _prefix, segment in NEGATIVE_SPAN_RE.findall(str(text or "")) if segment.strip()]
    return merge_unique(segments)


def negative_terms(text: str, plan: dict[str, Any] | None = None) -> list[str]:
    terms = list(negative_segments(text))
    if plan:
        terms.extend(str(value) for value in plan.get("negative_constraints", []) or [])
        for style in plan.get("negative_style", []) or []:
            terms.extend(NEGATIVE_STYLE_ALIASES.get(str(style), (str(style),)))
        for stage in plan.get("forbidden_stage", []) or []:
            terms.extend(STAGE_ALIASES.get(str(stage), (str(stage),)))
    for style, aliases in NEGATIVE_STYLE_ALIASES.items():
        if style in str(text):
            terms.extend(aliases)
    return [term for term in merge_unique(terms) if len(term) > 1]


def sanitize_positive_text(text: str, original_text: str) -> str:
    clean = NEGATIVE_SPAN_RE.sub("", str(text or ""))
    for term in sorted(negative_terms(original_text), key=len, reverse=True):
        clean = re.sub(re.escape(term), "", clean, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", clean).strip(" ,;，。；")


def sanitize_positive_text_for_plan(text: str, plan: dict[str, Any]) -> str:
    clean = sanitize_positive_text(text, str(plan.get("original_text", "")))
    for term in sorted(negative_terms(str(plan.get("original_text", "")), plan), key=len, reverse=True):
        clean = re.sub(re.escape(term), "", clean, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", clean).strip(" ,;，。；")


def apply_forbidden_precedence(plan: dict[str, Any]) -> dict[str, Any]:
    result = dict(plan)
    forbidden = set(str(stage) for stage in result.get("forbidden_stage", []) or [])
    if forbidden:
        result["desired_stage"] = [stage for stage in result.get("desired_stage", []) or [] if str(stage) not in forbidden]
    return result


def plan_has_negative_leak(plan: dict[str, Any]) -> bool:
    terms = [term.lower() for term in negative_terms(str(plan.get("original_text", "")), plan) if term]
    if not terms:
        return False
    texts = [str(plan.get("positive_query", "")), str(plan.get("hyde_text", ""))]
    texts.extend(str(row.get("text", "")) for row in plan.get("rewrites", []) if isinstance(row, dict))
    return any(term in text.lower() for term in terms for text in texts if text)


def scene_signature(
    positive_text: str,
    desired_stage: list[str],
    positive_purposes: list[str],
    positive_style: list[str],
) -> dict[str, Any]:
    return {
        "raw_positive_query": positive_text,
        "narrative_position": desired_stage,
        "emotion_function": positive_purposes,
        "camera_experience": positive_style,
        "confidence": confidence_for_plan(desired_stage, positive_purposes, positive_style),
        "evidence": [{"source": "rule", "field": "positive_query", "text": positive_text[:160], "confidence": 0.6}],
    }


def constraint_row(kind: str, polarity: str, values: list[str], strength: str, *, reason: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "polarity": polarity,
        "values": values,
        "strength": strength,
        "source": "rule",
        "confidence": 0.8,
        "reason": reason,
    }


def ambiguity(text: str, desired_stage: list[str], positive_purposes: list[str]) -> dict[str, Any]:
    reasons = []
    if not desired_stage:
        reasons.append("missing_explicit_stage")
    if not positive_purposes:
        reasons.append("missing_explicit_purpose")
    if any(term in text for term in ("高级", "有人味", "真实一点", "别端着", "现场感", "自然")):
        reasons.append("fuzzy_style_language")
    level = "high" if len(reasons) >= 2 else "medium" if reasons else "low"
    return {"level": level, "reasons": reasons, "needs_review": level == "high"}


def confidence_for_plan(stages: list[str], purposes: list[str], styles: list[str]) -> float:
    score = 0.35
    if stages:
        score += 0.2
    if purposes:
        score += 0.15
    if styles:
        score += 0.1
    return round(min(0.9, score), 6)


def generate_rewrites(plan: dict[str, Any]) -> list[str]:
    stage = " ".join(plan.get("desired_stage", []) or ["general scene"])
    purpose = " ".join(plan.get("positive_purposes", []) or ["clear narrative purpose"])
    style = " ".join(plan.get("positive_style", []) or [])
    query = str(plan.get("positive_query", ""))
    rows = [
        query,
        f"{stage} scene for {purpose}",
        f"director experience card focused on {purpose}",
        f"retrieve a reusable script moment for {stage}",
    ]
    if style:
        rows.append(f"{style} style while serving {purpose}")
    return [sanitize_positive_text(row, str(plan.get("original_text", ""))) for row in merge_unique(rows) if row]


def generate_hyde_text(plan: dict[str, Any]) -> str:
    stage = ", ".join(plan.get("desired_stage", []) or ["appropriate script stage"])
    purpose = ", ".join(plan.get("positive_purposes", []) or ["clear narrative purpose"])
    style = ", ".join(plan.get("positive_style", []) or ["restrained and story-led"])
    query = str(plan.get("positive_query", ""))
    return (
        f"Ideal experience card: useful for {stage}. "
        f"Script purpose: {purpose}. "
        f"Director note: prioritize exact scene function over visual similarity. "
        f"Style: {style}. Scene clue: {query}."
    )


def rewrite_rows(texts: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for index, text in enumerate(merge_unique([str(value) for value in texts if str(value).strip()]), start=1):
        rows.append(
            {
                "rewrite_id": f"rewrite_{index}",
                "text": text,
                "purpose": "semantic_recall",
                "weight": 1.0,
                "target_channels": ["script_use", "combined"],
                "source": "multi_query",
            }
        )
    return rows[:5]


def merge_unique(values: Iterable[Any]) -> list[Any]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


__all__ = [
    "DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT",
    "DEFAULT_PLANNER_CACHE_PATH",
    "DEFAULT_PLANNER_COMPARE_OUTPUT",
    "DEFAULT_PLANNER_PLAN_OUTPUT",
    "PLANNER_NAMES",
    "PlannerCache",
    "audit_cache",
    "compare_planners",
    "load_queries",
    "plan_many",
    "planner_cache_key",
    "planner_registry",
    "write_planner_report",
    "write_plans_jsonl",
]
