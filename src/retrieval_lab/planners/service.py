from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
import hashlib
import json
import re
import time
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_json, read_jsonl, write_json, write_jsonl
from retrieval_lab.schemas import SCHEMA_VERSION, QueryPlanModel, validate_record
from sceneweaver.retrieval.query_plan import build_query_plan


PLANNER_VERSION = "retrieval_lab_planner_v9"
DEFAULT_PLANNER_CACHE_PATH = Path(".tmp") / "retrieval_lab" / "planner_cache.jsonl"
DEFAULT_PLANNER_PLAN_OUTPUT = Path(".tmp") / "retrieval_lab" / "planner_plan.json"
DEFAULT_PLANNER_COMPARE_OUTPUT = Path(".tmp") / "retrieval_lab" / "planner_compare.json"
DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT = Path(".tmp") / "retrieval_lab" / "planner_cache_audit.json"
DEFAULT_REAL_LLM_PLANNER_MAX_TOKENS = 800
DEFAULT_REAL_LLM_PLANNER_TIMEOUT_SECONDS = 45.0

NEGATIVE_SPAN_RE = re.compile(
    r"(不要做成|不要|别|避免|不想|不是|拒绝|\b(?:without|avoid|exclude|do not|don't|not|no)\b)\s*([^,;，。；]*)",
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
        "技术细节",
        "质控细节",
        "设备校准",
        "校准复核",
        "操作界面",
        "确认提示",
    ),
    "team_work": ("team_work", "team work", "团队协作", "团队", "协作", "共创"),
    "value_expression": ("value_expression", "value expression", "价值表达", "建立信任", "品牌价值"),
    "outcome": ("outcome", "结果", "成果", "成效", "结果展示"),
    "scale_reveal": ("scale_reveal", "scale reveal", "规模揭示", "规模展开", "网络呈现", "影响范围", "规模感"),
    "growth": ("growth", "成长", "发展", "长期主义", "长期积累"),
    "character_intro": ("character_intro", "character intro", "人物介绍", "人物引入", "角色建立", "具体人物"),
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
    "scale_reveal": ("scale_reveal", "scale reveal", "规模揭示", "规模展开", "网络呈现", "影响范围", "规模感"),
    "growth": ("growth", "成长", "发展", "长期主义", "长期积累"),
    "character_intro": ("character_intro", "character intro", "人物介绍", "人物引入", "角色建立", "具体人物"),
    "ending": ("ending", "结尾", "收束", "号召", "结束"),
}

PURPOSE_ALIASES: dict[str, tuple[str, ...]] = {
    "opening": ("opening", "开场建立", "开场问题", "开场真实处境"),
    "setup": ("setup", "前情铺垫", "需求建立", "质控", "复核记录", "等待确认"),
    "technology_showcase": (
        "technology_showcase",
        "技术能力被自然看见",
        "技术能力",
        "技术细节",
        "质控细节",
        "设备校准",
        "校准复核",
        "操作界面",
        "确认提示",
    ),
    "team_work": ("team_work", "团队一起推进", "团队协作"),
    "value_expression": ("value_expression", "价值被具体场景承接", "价值表达"),
    "outcome": ("outcome", "结果和成效自然显现", "结果成效"),
    "growth": ("growth", "成长变化", "长期积累"),
    "character_intro": ("character_intro", "人物进入故事", "人物进入", "人物引入"),
    "ending": ("ending", "结尾收束", "留下信任"),
    "establish_problem": ("establish_problem", "建立问题", "问题建立", "现实压力", "痛点", "矛盾"),
    "establish_need": ("establish_need", "建立需求", "需求建立", "需求铺垫", "质控", "复核", "专业度"),
    "build_reality": ("build_reality", "真实感", "真实现场", "真实处境", "现场感", "建立语境", "grounded", "reality"),
    "build_trust": ("build_trust", "建立信任", "可信", "信任", "trust"),
    "show_pressure": ("show_pressure", "压力", "现实压力", "压力呈现", "紧张", "等待确认", "复核"),
    "show_distance": ("show_distance", "距离感", "空间距离", "地域距离"),
    "close_loop": ("close_loop", "闭环", "收束闭环", "前后呼应"),
    "show_outcome": ("show_outcome", "展示结果", "成果展示", "结果展示", "outcome"),
    "scale_reveal": ("scale_reveal", "规模揭示", "规模展开", "scale reveal"),
    "show_scale": ("show_scale", "展示规模", "规模展示", "show scale"),
    "show_network": ("show_network", "网络呈现", "关系网络", "协同网络", "协作链路", "show network"),
    "show_growth": ("show_growth", "成长", "发展", "growth"),
    "show_long_termism": ("show_long_termism", "长期主义", "长期价值", "long termism"),
    "introduce_people": ("introduce_people", "人物引入", "介绍人物", "具体人物", "人物进入", "introduce people"),
    "build_empathy": ("build_empathy", "共情", "建立共情", "人的连接", "empathy"),
    "show_team": ("show_team", "表现团队", "团队能力", "team"),
    "show_collaboration": ("show_collaboration", "协作", "协同", "协作过程", "collaboration"),
    "show_technology": ("show_technology", "表现技术", "技术能力", "科技能力", "technology", "质控", "校准", "复核"),
    "prove_capability": (
        "prove_capability",
        "证明能力",
        "能力证明",
        "可靠性",
        "prove capability",
        "专业度",
        "确认提示",
        "质控复核",
        "校准复核",
        "复核",
    ),
    "express_value": ("express_value", "表达价值", "价值表达", "品牌价值", "value"),
    "land_value": ("land_value", "价值落地", "落地价值", "land value"),
    "leave_trust": ("leave_trust", "留下信任", "信任收束", "leave trust"),
    "humanize_professional": ("humanize_professional", "专业的人味", "专业但有人味", "humanize professional"),
    "humanize_technology": ("humanize_technology", "技术有人味", "科技有人味", "humanize technology"),
    "stabilize_emotion": ("stabilize_emotion", "稳定情绪", "安定感", "stabilize emotion"),
    "avoid_overclaim": ("avoid_overclaim", "避免夸大", "不过度承诺", "avoid overclaim"),
    "connect_feedback_to_mission": ("connect_feedback_to_mission", "反馈连接使命", "使命反馈"),
    "show_face_to_face_communication": ("show_face_to_face_communication", "面对面沟通", "face to face communication"),
    "keep_human_warmth": ("keep_human_warmth", "人的温度", "有人味", "human warmth", "human"),
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
    "slogan_driven": ("口号感", "口号驱动", "slogan driven", "slogan-heavy", "tagline"),
    "generic_brand_film": ("品牌片", "品牌质感", "generic brand film", "brand film"),
    "fortune_500_polish": ("世界500强", "五百强", "fortune 500", "fortune-500", "polished corporate"),
    "tech_coldness": ("科技冷感", "冰冷科技", "tech coldness", "cold technology"),
}

RISKY_POSITIVE_TEXT_TERMS = (
    "advertising",
    "campaign",
    "slogan",
    "product pitch",
    "sales pitch",
    "corporate report",
    "generic brand film",
    "fortune 500",
    "fortune-500",
)

NATURAL_LABEL_PHRASES: dict[str, str] = {
    "opening": "开场建立问题和真实处境",
    "setup": "前情铺垫和需求建立",
    "technology_showcase": "技术能力的自然呈现",
    "team_work": "团队协作和共同推进",
    "value_expression": "价值表达和信任建立",
    "outcome": "结果和成效呈现",
    "scale_reveal": "规模展开和关系网络显现",
    "growth": "成长变化和长期积累",
    "character_intro": "人物进入和共情建立",
    "ending": "结尾收束和留下信任",
    "establish_problem": "建立问题和现实压力",
    "establish_need": "建立需求和前情铺垫",
    "build_reality": "真实处境和现场感",
    "build_trust": "建立可信感和信任",
    "show_pressure": "呈现现实压力",
    "show_distance": "呈现空间距离和现实距离",
    "close_loop": "形成前后呼应的闭环",
    "show_outcome": "呈现结果和成效",
    "show_scale": "展示规模感",
    "show_network": "呈现关系网络和协作链路",
    "show_growth": "呈现成长变化",
    "show_long_termism": "呈现长期主义和持续积累",
    "introduce_people": "引入具体人物",
    "build_empathy": "建立共情和人的连接",
    "show_team": "呈现团队能力",
    "show_collaboration": "呈现协作过程",
    "show_technology": "呈现技术能力",
    "prove_capability": "证明能力和可靠性",
    "express_value": "表达价值和意义",
    "land_value": "让价值落到具体场景",
    "leave_trust": "在结尾留下信任",
    "humanize_professional": "让专业表达更有人味",
    "humanize_technology": "让技术表达更有人味",
    "stabilize_emotion": "稳定情绪和建立安定感",
    "avoid_overclaim": "克制表达避免夸大",
    "connect_feedback_to_mission": "把反馈连接到使命",
    "show_face_to_face_communication": "呈现面对面沟通",
    "keep_human_warmth": "保留人的温度",
    "human_warmth": "人的温度",
    "documentary": "纪录片式观察",
    "real_location": "真实现场",
}

PLANNER_NAMES = ("rule", "multi_query", "hyde_card", "fake_llm", "llm_structured", "style_safe_llm_structured", "legacy_adapter")


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
            for line in self.path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
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
        "llm_structured": llm_structured_plan,
        "style_safe_llm_structured": style_safe_llm_structured_plan,
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
    llm_call_count = 0
    llm_plan_count = 0
    llm_fallback_count = 0
    llm_timing_count = 0
    llm_request_seconds_total = 0.0
    llm_total_seconds_total = 0.0
    llm_max_request_seconds = 0.0
    llm_prompt_chars_total = 0
    llm_response_chars_total = 0
    budget_stopped = False
    estimated_llm_cost_cny = 0.0
    started = time.perf_counter()
    for user_input in user_inputs:
        plan_config = dict(config or {})
        plan = cache.get(planner, user_input, config or {})
        cache_hit = plan is not None
        if plan is not None:
            cache_hits += 1
        else:
            cache_misses += 1
            if planner in {"llm_structured", "style_safe_llm_structured"} and bool(plan_config.get("require_llm", False)):
                sample_limit = int(plan_config.get("llm_sample_size", 0) or 0)
                budget_cny = float(plan_config.get("budget_cny", 20.0) or 20.0)
                cny_per_1k = float(plan_config.get("cny_per_1k_tokens", 0.01) or 0.01)
                estimated_cost = estimate_planner_cost_cny(user_input, cny_per_1k)
                if sample_limit <= 0 or llm_call_count >= sample_limit or estimated_llm_cost_cny + estimated_cost > budget_cny:
                    plan_config["require_llm"] = False
                    budget_stopped = True
                else:
                    estimated_llm_cost_cny += estimated_cost
            plan = planner_registry()[planner](user_input, plan_config)
            plan = normalize_plan(plan)
            cache.set(planner, user_input, plan, plan_config)
        validation = validate_record("query_plan", plan)
        if not validation["valid"]:
            invalid_count += 1
            plan = {**plan, "validation_errors": validation["errors"]}
        else:
            plan = validation["normalized"]
        if plan_has_negative_leak(plan):
            negative_leak_count += 1
        if bool(plan.get("provenance", {}).get("llm_used", False)):
            llm_plan_count += 1
            if not cache_hit:
                llm_call_count += 1
            if bool(plan.get("provenance", {}).get("fallback_used", False)):
                llm_fallback_count += 1
            timing = plan.get("planner_metadata", {}).get("llm_timing", {})
            if isinstance(timing, dict) and not cache_hit:
                llm_timing_count += 1
                request_seconds = float(timing.get("llm_request_seconds", 0.0) or 0.0)
                total_seconds = float(timing.get("total_seconds", 0.0) or 0.0)
                llm_request_seconds_total += request_seconds
                llm_total_seconds_total += total_seconds
                llm_max_request_seconds = max(llm_max_request_seconds, request_seconds)
                llm_prompt_chars_total += int(timing.get("prompt_chars", 0) or 0)
                llm_response_chars_total += int(timing.get("response_chars", 0) or 0)
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
            "llm_call_count": llm_call_count,
            "llm_plan_count": llm_plan_count,
            "llm_fallback_count": llm_fallback_count,
            "estimated_llm_cost_cny": round(estimated_llm_cost_cny, 6),
            "budget_stopped": budget_stopped,
            "llm_timing_count": llm_timing_count,
            "llm_request_seconds_total": round(llm_request_seconds_total, 6),
            "llm_request_seconds_avg": round(llm_request_seconds_total / max(1, llm_timing_count), 6),
            "llm_total_seconds_total": round(llm_total_seconds_total, 6),
            "llm_total_seconds_avg": round(llm_total_seconds_total / max(1, llm_timing_count), 6),
            "llm_max_request_seconds": round(llm_max_request_seconds, 6),
            "llm_prompt_chars_total": llm_prompt_chars_total,
            "llm_response_chars_total": llm_response_chars_total,
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
    plan["rewrites"] = rewrite_rows(
        generate_rewrites(plan, natural_language=bool((config or {}).get("natural_language_rewrites", False)))
    )
    plan["confidence"] = max(float(plan.get("confidence", 0.35)), 0.55)
    plan["provenance"] = {**plan["provenance"], "planner": "multi_query", "source": "multi_query"}
    return plan


def hyde_card_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = rule_plan(user_input, config)
    plan["planner"] = "hyde_card"
    plan["hyde_text"] = sanitize_positive_text(
        generate_hyde_text(plan, natural_language=bool((config or {}).get("natural_language_rewrites", False))),
        user_input,
    )
    plan["confidence"] = max(float(plan.get("confidence", 0.35)), 0.55)
    plan["provenance"] = {**plan["provenance"], "planner": "hyde_card", "source": "hyde_card"}
    return plan


def fake_llm_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = multi_query_plan(user_input, config)
    plan["planner"] = "fake_llm"
    plan["hyde_text"] = sanitize_positive_text(
        generate_hyde_text(plan, natural_language=bool((config or {}).get("natural_language_rewrites", False))),
        user_input,
    )
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


def llm_structured_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    if bool(config.get("require_llm", False)):
        return real_llm_structured_plan(user_input, config)
    plan = fake_llm_plan(user_input, config)
    plan["planner"] = "llm_structured"
    plan["provenance"] = {
        **plan["provenance"],
        "planner": "llm_structured",
        "source": "system",
        "llm_used": False,
    }
    plan["planner_metadata"] = {
        **plan.get("planner_metadata", {}),
        "llm_structured_mode": "fake_schema",
        "budget_cny": float(config.get("budget_cny", 20.0) or 20.0),
    }
    return plan


def style_safe_llm_structured_plan(user_input: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    base = multi_query_plan(user_input, config)
    llm_plan = real_llm_structured_plan(user_input, config) if bool(config.get("require_llm", False)) else fake_llm_plan(user_input, config)
    plan = {**base}
    plan["planner"] = "style_safe_llm_structured"
    plan["desired_stage"] = merge_unique([*base.get("desired_stage", []), *llm_plan.get("desired_stage", [])])
    plan["forbidden_stage"] = merge_unique([*base.get("forbidden_stage", []), *llm_plan.get("forbidden_stage", [])])
    plan["positive_purposes"] = merge_unique([*base.get("positive_purposes", []), *llm_plan.get("positive_purposes", [])])
    plan["positive_style"] = merge_unique([*base.get("positive_style", []), *llm_plan.get("positive_style", [])])
    plan["negative_style"] = llm_negative_style_values(llm_plan.get("negative_style"), base, user_input)
    if isinstance(llm_plan.get("scene_signature"), dict):
        plan["scene_signature"] = {**base.get("scene_signature", {}), **llm_plan["scene_signature"]}
    plan["positive_query"] = sanitize_positive_text_for_plan(str(base.get("positive_query", "")), plan)
    natural_language = bool(config.get("natural_language_rewrites", False))
    plan["rewrites"] = rewrite_rows(generate_rewrites(plan, natural_language=natural_language))
    plan["hyde_text"] = sanitize_positive_text_for_plan(generate_hyde_text(plan, natural_language=natural_language), plan)
    plan["confidence"] = max(float(base.get("confidence", 0.35)), float(llm_plan.get("confidence", 0.0) or 0.0), 0.7)
    plan["provenance"] = {
        **base.get("provenance", {}),
        "planner": "style_safe_llm_structured",
        "source": "system",
        "llm_used": bool(llm_plan.get("provenance", {}).get("llm_used", False)),
        "fallback_used": bool(llm_plan.get("provenance", {}).get("fallback_used", False)),
    }
    plan["planner_metadata"] = {
        **base.get("planner_metadata", {}),
        "style_safe_llm_structured": True,
        "source_planner": llm_plan.get("planner", ""),
        "ignored_llm_positive_text": True,
        "llm_metadata": llm_plan.get("planner_metadata", {}),
    }
    llm_metadata = llm_plan.get("planner_metadata", {}) if isinstance(llm_plan.get("planner_metadata", {}), dict) else {}
    if isinstance(llm_metadata.get("llm_timing"), dict):
        plan["planner_metadata"]["llm_timing"] = llm_metadata["llm_timing"]
    if isinstance(llm_metadata.get("llm_request"), dict):
        plan["planner_metadata"]["llm_request"] = llm_metadata["llm_request"]
    return plan


def real_llm_structured_plan(user_input: str, config: dict[str, Any]) -> dict[str, Any]:
    from sceneweaver.llm.client import VisionLLMClient
    from sceneweaver.llm.client import llm_config_metadata

    started = time.perf_counter()
    base_started = time.perf_counter()
    base = multi_query_plan(user_input, config)
    base_plan_seconds = time.perf_counter() - base_started
    prompt_started = time.perf_counter()
    system_prompt = (
        "You are a retrieval query understanding planner. Return JSON only. "
        "Keep answers concise. Do not put negative constraints into positive_query, rewrites, or hyde_text."
    )
    user_prompt = json.dumps(
        {
            "query": user_input,
            "allowed_stages": sorted(STAGE_ALIASES),
            "allowed_positive_style": sorted(POSITIVE_STYLE_ALIASES),
            "allowed_negative_style": sorted(NEGATIVE_STYLE_ALIASES),
            "schema": {
                "positive_query": "string",
                "desired_stage": ["allowed stage"],
                "forbidden_stage": ["allowed stage"],
                "positive_purposes": ["string"],
                "positive_style": ["allowed style"],
                "negative_style": ["allowed style"],
                "scene_signature": {
                    "people": ["string"],
                    "place": ["string"],
                    "actions": ["string"],
                    "objects": ["string"],
                    "emotion_function": "string",
                    "narrative_position": "string",
                    "camera_experience": "string",
                },
                "rewrites": [{"text": "positive retrieval rewrite"}],
                "hyde_text": "positive ideal experience card note",
                "confidence": "0..1",
            },
        },
        ensure_ascii=False,
    )
    prompt_build_seconds = time.perf_counter() - prompt_started
    max_tokens = int(config.get("max_tokens", DEFAULT_REAL_LLM_PLANNER_MAX_TOKENS) or DEFAULT_REAL_LLM_PLANNER_MAX_TOKENS)
    timeout_seconds = float(
        config.get("timeout_seconds", DEFAULT_REAL_LLM_PLANNER_TIMEOUT_SECONDS) or DEFAULT_REAL_LLM_PLANNER_TIMEOUT_SECONDS
    )
    retries = int(config.get("retries", 0) or 0)
    enable_thinking = bool(config.get("llm_enable_thinking", False))
    thinking_budget = int(config.get("thinking_budget", 0) or 0)
    client = VisionLLMClient()
    llm_provider_metadata = llm_config_metadata(client.config)
    request_started = time.perf_counter()
    try:
        response = client.analyze_text_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            retries=retries,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
    except RuntimeError as exc:
        llm_request_seconds = time.perf_counter() - request_started
        return llm_structured_error_fallback_plan(
            base,
            client=client,
            config={
                **llm_provider_metadata,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "retries": retries,
                "enable_thinking": enable_thinking,
                "thinking_budget": thinking_budget,
            },
            error=exc,
            started=started,
            base_plan_seconds=base_plan_seconds,
            prompt_build_seconds=prompt_build_seconds,
            llm_request_seconds=llm_request_seconds,
            prompt_chars=len(system_prompt) + len(user_prompt),
        )
    llm_request_seconds = time.perf_counter() - request_started
    normalize_started = time.perf_counter()
    merged = {**base}
    merged["planner"] = "llm_structured"
    merged["positive_query"] = sanitize_positive_text_for_plan(str(response.get("positive_query") or base.get("positive_query", "")), base)
    merged["desired_stage"] = enum_values(response.get("desired_stage"), STAGE_ALIASES)
    merged["forbidden_stage"] = enum_values(response.get("forbidden_stage"), FORBIDDEN_STAGE_ALIASES)
    merged["positive_style"] = enum_values(response.get("positive_style"), POSITIVE_STYLE_ALIASES)
    merged["negative_style"] = llm_negative_style_values(response.get("negative_style"), base, user_input)
    merged["positive_purposes"] = merge_unique([*base.get("positive_purposes", []), *string_values(response.get("positive_purposes"))])
    raw_rewrites = response.get("rewrites", [])
    if not isinstance(raw_rewrites, list):
        raw_rewrites = []
    merged["rewrites"] = rewrite_rows(
        sanitize_positive_text_for_plan(str(row.get("text", "")), merged)
        for row in raw_rewrites
        if isinstance(row, dict)
    )
    merged["hyde_text"] = sanitize_positive_text_for_plan(str(response.get("hyde_text", "")), merged)
    if isinstance(response.get("scene_signature"), dict):
        merged["scene_signature"] = {**base.get("scene_signature", {}), **response["scene_signature"]}
    merged["confidence"] = max(float(base.get("confidence", 0.35)), safe_float(response.get("confidence"), 0.7))
    merged["provenance"] = {
        **base.get("provenance", {}),
        "planner": "llm_structured",
        "source": "llm",
        "llm_used": True,
    }
    merged["planner_metadata"] = {
        **base.get("planner_metadata", {}),
        "llm_structured_mode": "real",
        "prompt_version": "retrieval_lab_llm_structured_v2",
        "llm_request": {
            **llm_provider_metadata,
            "model": client.config.model,
            "base_url": client.config.base_url,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
        },
    }
    merged["planner_metadata"]["llm_timing"] = {
        "base_plan_seconds": round(base_plan_seconds, 6),
        "prompt_build_seconds": round(prompt_build_seconds, 6),
        "llm_request_seconds": round(llm_request_seconds, 6),
        "normalization_seconds": round(time.perf_counter() - normalize_started, 6),
        "total_seconds": round(time.perf_counter() - started, 6),
        "prompt_chars": len(system_prompt) + len(user_prompt),
        "response_chars": len(json.dumps(response, ensure_ascii=False)),
    }
    return merged


def llm_structured_error_fallback_plan(
    base: dict[str, Any],
    *,
    client: Any,
    config: dict[str, Any],
    error: Exception,
    started: float,
    base_plan_seconds: float,
    prompt_build_seconds: float,
    llm_request_seconds: float,
    prompt_chars: int,
) -> dict[str, Any]:
    plan = {**base}
    plan["planner"] = "llm_structured"
    plan["provenance"] = {
        **base.get("provenance", {}),
        "planner": "llm_structured",
        "source": "llm_error_fallback",
        "llm_used": True,
        "fallback_used": True,
    }
    plan["planner_metadata"] = {
        **base.get("planner_metadata", {}),
        "llm_structured_mode": "real_error_fallback",
        "prompt_version": "retrieval_lab_llm_structured_v2",
        "llm_error": str(error)[:800],
        "llm_request": {
            "model": client.config.model,
            "base_url": client.config.base_url,
            **config,
        },
        "llm_timing": {
            "base_plan_seconds": round(base_plan_seconds, 6),
            "prompt_build_seconds": round(prompt_build_seconds, 6),
            "llm_request_seconds": round(llm_request_seconds, 6),
            "normalization_seconds": 0.0,
            "total_seconds": round(time.perf_counter() - started, 6),
            "prompt_chars": prompt_chars,
            "response_chars": 0,
        },
    }
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
    split: str = "test.md",
    limit: int = 0,
) -> list[str]:
    values = [query for query in queries or [] if query]
    if input_file:
        values.extend(load_queries_from_file(input_file))
    if dataset_path:
        from retrieval_lab.datasets import read_cases

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
    segments = []
    for prefix, segment in NEGATIVE_SPAN_RE.findall(str(text or "")):
        segment = segment.strip()
        if not segment:
            continue
        if str(prefix).strip() == "别" and segment.startswith("的"):
            continue
        segments.append(segment)
    return merge_unique(segments)


def negative_terms(text: str, plan: dict[str, Any] | None = None) -> list[str]:
    segments = negative_segments(text)
    terms = list(segments)
    negative_context = " ".join(segments).lower()
    if plan:
        constraints = [str(value) for value in plan.get("negative_constraints", []) or []]
        terms.extend(constraints)
        negative_context = " ".join([negative_context, *constraints]).lower()
        for style in plan.get("negative_style", []) or []:
            terms.extend(NEGATIVE_STYLE_ALIASES.get(str(style), (str(style),)))
        for stage in plan.get("forbidden_stage", []) or []:
            stage_key = str(stage)
            terms.append(stage_key)
            terms.extend(
                alias
                for alias in STAGE_ALIASES.get(stage_key, ())
                if alias.lower() in negative_context or stage_key.lower() in negative_context
            )
    for style, aliases in NEGATIVE_STYLE_ALIASES.items():
        if style in str(text):
            terms.extend(aliases)
    return [term for term in merge_unique(terms) if len(term) > 1]


def sanitize_positive_text(text: str, original_text: str) -> str:
    clean = NEGATIVE_SPAN_RE.sub("", str(text or ""))
    for term in sorted(negative_terms(original_text), key=len, reverse=True):
        clean = re.sub(re.escape(term), "", clean, flags=re.IGNORECASE)
    clean = remove_risky_positive_terms(clean)
    return re.sub(r"\s+", " ", clean).strip(" ,;，。；")


def sanitize_positive_text_for_plan(text: str, plan: dict[str, Any]) -> str:
    clean = sanitize_positive_text(text, str(plan.get("original_text", "")))
    for term in sorted(negative_terms(str(plan.get("original_text", "")), plan), key=len, reverse=True):
        clean = re.sub(re.escape(term), "", clean, flags=re.IGNORECASE)
    clean = remove_risky_positive_terms(clean)
    return re.sub(r"\s+", " ", clean).strip(" ,;，。；")


def remove_risky_positive_terms(text: str) -> str:
    clean = str(text or "")
    for term in sorted(RISKY_POSITIVE_TEXT_TERMS, key=len, reverse=True):
        clean = re.sub(re.escape(term), "", clean, flags=re.IGNORECASE)
    return clean


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


def generate_rewrites(plan: dict[str, Any], *, natural_language: bool = False) -> list[str]:
    stage_values = list(plan.get("desired_stage", []) or ["general scene"])
    purpose_values = list(plan.get("positive_purposes", []) or ["clear narrative purpose"])
    style_values = list(plan.get("positive_style", []) or [])
    stage = label_phrase(stage_values, natural_language=natural_language)
    purpose = label_phrase(purpose_values, natural_language=natural_language)
    style = label_phrase(style_values, natural_language=natural_language)
    query = str(plan.get("positive_query", ""))
    rows = [
        query,
        f"寻找服务于{purpose}的{stage}",
        f"导演经验卡重点放在{purpose}",
        f"找一个可以复用的{stage}段落",
    ]
    if style:
        rows.append(f"保持{style}，同时服务于{purpose}")
    return [sanitize_positive_text(row, str(plan.get("original_text", ""))) for row in merge_unique(rows) if row]


def generate_hyde_text(plan: dict[str, Any], *, natural_language: bool = False) -> str:
    stage = label_phrase(list(plan.get("desired_stage", []) or ["appropriate script stage"]), natural_language=natural_language)
    purpose = label_phrase(list(plan.get("positive_purposes", []) or ["clear narrative purpose"]), natural_language=natural_language)
    style = label_phrase(list(plan.get("positive_style", []) or ["restrained and story-led"]), natural_language=natural_language)
    query = str(plan.get("positive_query", ""))
    return (
        f"理想经验卡：适合{stage}。"
        f"叙事功能：{purpose}。"
        f"导演笔记：优先匹配场景功能，而不只看画面相似。"
        f"风格：{style}。线索：{query}。"
    )


def label_phrase(values: list[Any], *, natural_language: bool) -> str:
    labels = [str(value) for value in values if str(value).strip()]
    if not natural_language:
        return " ".join(labels)
    return "、".join(NATURAL_LABEL_PHRASES.get(label, label.replace("_", " ")) for label in labels)


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


def string_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def enum_values(value: Any, allowed: dict[str, tuple[str, ...]]) -> list[str]:
    allowed_names = set(allowed)
    return [item for item in string_values(value) if item in allowed_names]


def llm_negative_style_values(value: Any, base_plan: dict[str, Any], user_input: str) -> list[str]:
    base_values = string_values(base_plan.get("negative_style"))
    has_explicit_negative = bool(
        base_values
        or string_values(base_plan.get("negative_constraints"))
        or NEGATIVE_SPAN_RE.search(user_input)
    )
    if not has_explicit_negative:
        return []
    return merge_unique([*base_values, *enum_values(value, NEGATIVE_STYLE_ALIASES)])


def safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def estimate_planner_cost_cny(user_input: str, cny_per_1k_tokens: float) -> float:
    estimated_tokens = max(1, int((len(user_input) + 1800) / 3.5))
    return estimated_tokens / 1000.0 * max(0.0, cny_per_1k_tokens)


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
