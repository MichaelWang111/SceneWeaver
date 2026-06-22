from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

from retrieval_lab.artifacts import data_sha256, read_jsonl, write_json
from retrieval_lab.datasets import DEFAULT_DATASET_PATH, read_cases
from retrieval_lab.evaluators import analyze_failure_rows, graded_metrics, recall_bound_rows, recall_bound_summary
from retrieval_lab.experiments.constants import CORE_EXPERIMENT_COMMANDS
from retrieval_lab.experiments.runs import cases_from_run_rows, run_artifact_summary
from retrieval_lab.indexes import build_index_manifest, index_items_from_cases
from retrieval_lab.indexes.service import write_index_manifest
from retrieval_lab.planners import DEFAULT_PLANNER_CACHE_PATH, compare_planners, plan_many
from retrieval_lab.qrels import (
    active_qrels_samples,
    load_qrels,
    pooled_qrels_from_run_rows,
    pooled_qrels_summary,
    qrels_audit_summary,
    qrels_trust_level,
    reranker_candidate_summary,
    write_qrels,
)
from retrieval_lab.ranking import (
    rerank_row_by_qrels,
    rerank_row_by_rule,
    rerank_run_rows,
    rerank_run_rows_by_workflow,
    row_with_reranked_results,
    rule_rerank_score,
)
from retrieval_lab.reports import markdown_report
from retrieval_lab.retrieval.service import (
    prepare_retrieval_index,
    retrieve_case,
    retrieval_run_from_cases,
)
from retrieval_lab.experiments.tuning import leave_one_fixture_out_report, tune_constraints_report


DEFAULT_CORE_REPORT_PATH = Path(".tmp") / "retrieval_lab" / "core_experiment_latest.json"
DEFAULT_CORE_MARKDOWN_PATH = Path(".tmp") / "retrieval_lab" / "core_experiment_latest.md"
DEFAULT_CORE_QRELS_PATH = Path(".tmp") / "pooled_qrels_next.jsonl"
DEFAULT_HARD_NEGATIVE_PATH = Path(".tmp") / "retrieval_lab" / "hard_negatives_latest.jsonl"

DEFAULT_WORKFLOW_KEYS = (
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
)

ANTI_OVERFIT_WORKFLOWS = (
    ("metadata_assisted_hybrid", "fuzzy", "hybrid_rrf_constraints_signature"),
    ("metadata_assisted_style_safe", "fuzzy", "style_safe_signature"),
    ("natural_fuzzy_hybrid", "natural_fuzzy", "hybrid_rrf_constraints_signature"),
    ("natural_fuzzy_style_safe", "natural_fuzzy", "style_safe_signature"),
)

INTERNAL_METADATA_LABELS = {
    "opening",
    "setup",
    "technology_showcase",
    "team_work",
    "value_expression",
    "outcome",
    "scale_reveal",
    "growth",
    "character_intro",
    "ending",
    "establish_problem",
    "establish_need",
    "build_reality",
    "build_trust",
    "show_pressure",
    "show_distance",
    "close_loop",
    "show_outcome",
    "show_scale",
    "show_network",
    "show_growth",
    "show_long_termism",
    "introduce_people",
    "build_empathy",
    "show_team",
    "show_collaboration",
    "show_technology",
    "prove_capability",
    "express_value",
    "land_value",
    "leave_trust",
    "humanize_professional",
    "humanize_technology",
    "stabilize_emotion",
    "avoid_overclaim",
    "connect_feedback_to_mission",
    "show_face_to_face_communication",
    "keep_human_warmth",
}

NATURAL_STAGE_TEXT = {
    "opening": "开场里的问题和真实处境",
    "setup": "前情铺垫里的现实需求",
    "technology_showcase": "技术能力被自然看见的时刻",
    "team_work": "团队一起推进事情的过程",
    "value_expression": "价值被具体场景承接的段落",
    "outcome": "结果和成效自然显现的段落",
    "scale_reveal": "规模、网络和影响范围慢慢展开的段落",
    "growth": "成长变化和长期积累被看见的段落",
    "character_intro": "人物进入故事并让人产生共情的段落",
    "ending": "结尾收束并留下信任的段落",
}

NATURAL_PURPOSE_TEXT = {
    "opening": "开场建立问题和真实处境",
    "setup": "前情铺垫和需求建立",
    "technology_showcase": "技术能力被自然看见",
    "team_work": "团队一起推进事情",
    "value_expression": "价值被具体场景承接",
    "outcome": "结果和成效自然显现",
    "growth": "成长变化和长期积累被看见",
    "character_intro": "人物进入故事并让人产生共情",
    "ending": "结尾收束并留下信任",
    "establish_problem": "把问题和现实压力讲清楚",
    "establish_need": "让需求自然成立",
    "build_reality": "让观众相信这是一个真实处境",
    "build_trust": "建立可信感",
    "show_pressure": "呈现现实压力",
    "show_distance": "呈现距离和阻隔",
    "close_loop": "形成前后呼应",
    "show_outcome": "呈现结果和成效",
    "scale_reveal": "让规模逐渐展开",
    "show_scale": "展示规模感",
    "show_network": "呈现关系网络和协作链路",
    "show_growth": "呈现成长变化",
    "show_long_termism": "表达长期积累",
    "introduce_people": "让具体人物进入故事",
    "build_empathy": "建立共情",
    "show_team": "呈现团队能力",
    "show_collaboration": "呈现协作过程",
    "show_technology": "呈现技术能力",
    "prove_capability": "证明能力和可靠性",
    "express_value": "表达价值和意义",
    "land_value": "让价值落到具体场景",
    "leave_trust": "在收束时留下信任",
    "humanize_professional": "让专业表达更有人味",
    "humanize_technology": "让技术表达更有人味",
    "stabilize_emotion": "稳定情绪",
    "avoid_overclaim": "保持克制不过度承诺",
    "connect_feedback_to_mission": "把反馈和使命连接起来",
    "show_face_to_face_communication": "呈现面对面沟通",
    "keep_human_warmth": "保留人的温度",
}


def native_core_experiment_command(args: Any, command: str) -> dict[str, Any]:
    started = time.perf_counter()
    if command == "search":
        report = search_report(args)
    elif command in {"evaluate", "evaluate-hybrid", "compare-ranking-workflows", "validate-ranking-keys"}:
        report = workflow_evaluation_report(args, command=command, cases=variant_cases_for_qrels(args))
    elif command == "compare-query-understanding":
        report = query_understanding_report(args)
    elif command == "validate-fuzzy-understanding":
        report = fuzzy_understanding_report(args)
    elif command == "validate-paraphrase-stress":
        report = paraphrase_report(args)
    elif command == "evaluate-fuzzy-multirelevance":
        report = fuzzy_multirelevance_report(args)
    elif command == "evaluate-anti-overfit-fuzzy":
        report = anti_overfit_fuzzy_report(args)
    elif command in {"evaluate-graded", "evaluate-pooled"}:
        report = graded_or_pooled_report(args, command=command)
    elif command in {"build-pooled-qrels", "build-graded-qrels"}:
        report = build_qrels_report(args, command=command)
    elif command == "sample-active-qrels":
        report = sample_active_qrels_report(args)
    elif command in {"compare-strong-baselines", "compare-rerank-upper-bound"}:
        report = rerank_upper_bound_report(args, command=command)
    elif command in {"validate-rerank-gate", "compare-rerank-gates"}:
        report = rerank_gate_report(args, command=command)
    elif command in {"validate-style-negatives", "validate-style-risk-mining"}:
        report = style_risk_report(args, command=command)
    elif command == "mine-hard-negatives":
        report = mine_hard_negatives_report(args)
    elif command == "validate-scene-signature":
        report = workflow_evaluation_report(
            args,
            command=command,
            cases=load_command_cases(args),
            forced_ranking_keys=("hybrid_rrf_constraints", "hybrid_rrf_constraints_signature", "adaptive_signature"),
        )
    elif command == "analyze-failures":
        report = native_failure_report(args)
    elif command == "analyze-recall-bound":
        report = native_recall_bound_report(args)
    elif command == "build-index":
        report = native_build_index_report(args)
    elif command == "compact-embedding-cache":
        report = native_compact_cache_report(args)
    elif command == "tune-constraints":
        report = tune_constraints_report(args)
    elif command == "evaluate-leave-one-fixture-out":
        report = leave_one_fixture_out_report(args)
    else:
        raise ValueError(f"unsupported native core command: {command}")
    report.setdefault("elapsed_seconds", round(time.perf_counter() - started, 6))
    report.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    report.setdefault("fingerprint", data_sha256(report.get("summary", report)))
    write_core_outputs(args, report)
    return {
        "method": report.get("method", "retrieval_lab_native_core_experiment"),
        "output": str(output_path(args)),
        "summary": {**report.get("summary", {}), "output": str(output_path(args)), "fingerprint": report["fingerprint"]},
    }


def workflow_evaluation_report(
    args: Any,
    *,
    command: str,
    cases: list[dict[str, Any]],
    forced_ranking_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    retrieval = run_retrieval_cases(args, cases=cases, run_name="baseline::native_retrieval")
    keys = list(forced_ranking_keys or command_ranking_keys(args, command))
    run_rows = workflow_runs_from_retrieval(retrieval, ranking_keys=keys, top_k=int(getattr(args, "top_k", 10)))
    qrels = qrels_for_report(args, run_rows=run_rows)
    run_metrics = evaluate_runs_if_possible(run_rows, qrels=qrels, top_k=int(getattr(args, "top_k", 10)))
    workflows = {
        key: {
            "summary": summarize_rows(first_rows_for_key(run_rows, key), qrels=qrels),
            "metrics": {"overall": run_metrics.get(first_run_name_for_key(run_rows, key), {})},
        }
        for key in keys
    }
    best_workflow = best_run_by_metrics(run_metrics) if run_metrics else best_workflow_by_summary(workflows)
    cases_payload = cases_from_run_rows(run_rows)
    summary = {
        **run_artifact_summary(run_rows, cases_payload),
        **summarize_rows(first_rows(run_rows), qrels=qrels),
        "command": command,
        "ranking_keys": keys,
        "best_workflow": best_workflow,
        "qrels_count": len(qrels),
        "qrels_trust_level": qrels_trust_level(qrels) if qrels else "none",
        "compat_backend_used": False,
    }
    return {
        "method": f"retrieval_lab_native_{command.replace('-', '_')}",
        "summary": summary,
        "run_rows": run_rows,
        "cases": cases_payload,
        "workflows": workflows,
        "run_metrics": run_metrics,
        "qrels_source": qrels_source(args, qrels),
        "experiment": experiment_record(args, command, summary),
    }


def search_report(args: Any) -> dict[str, Any]:
    cases = load_command_cases(args)
    items = index_items_from_cases(cases)
    prepared = prepare_retrieval_index(items)
    queries = list(getattr(args, "query", []) or [])
    if not queries:
        queries = [str(cases[0].get("user_input", ""))] if cases else [""]
    plans = plan_many(
        queries,
        planner=planner_name(args),
        cache_path=planner_cache(args),
        config={"command": "search"},
    )["plans"]
    rows = []
    for index, (query, plan) in enumerate(zip(queries, plans, strict=False), start=1):
        fake_case = {
            "case_id": f"search_{index}",
            "case_type": "search",
            "user_input": query,
            "target": {},
        }
        rows.append(
            retrieve_case(
                fake_case,
                plan=plan,
                items=items,
                prepared_index=prepared,
                top_k=int(getattr(args, "top_k", 10)),
                candidate_depth=int(getattr(args, "candidate_depth", 100)),
                ranking_key=str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature")),
            )
        )
    run_rows = {"search::native": rows}
    summary = {
        **run_artifact_summary(run_rows, cases_from_run_rows(run_rows)),
        "query_count": len(queries),
        "index_item_count": len(items),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_search",
        "summary": summary,
        "run_rows": run_rows,
        "cases": rows,
        "experiment": experiment_record(args, "search", summary),
    }


def query_understanding_report(args: Any) -> dict[str, Any]:
    cases = load_command_cases(args)
    planners = csv_values(str(getattr(args, "query_planners", "") or getattr(args, "planners", "") or "rule,multi_query,hyde_card"))
    planner_compare = compare_planners(
        [str(case.get("user_input", "")) for case in cases],
        planners=planners,
        cache_path=planner_cache(args),
        use_cache=planner_cache(args) is not None,
    )
    planner_reports = {}
    for planner in planners:
        retrieval = run_retrieval_cases(args, cases=cases, planner=planner, run_name=f"{planner}::native")
        rows = first_rows(retrieval["run_rows"])
        planner_reports[planner] = {
            "planner_summary": {
                **summarize_rows(rows),
                "negative_leak_rate": planner_compare["planner_metrics"][planner].get("negative_leak_rate", 0.0),
                "invalid_plan_count": planner_compare["planner_metrics"][planner].get("invalid_plan_count", 0),
            },
            "run_rows": retrieval["run_rows"],
        }
    baseline = planners[0] if planners else "rule"
    baseline_metrics = planner_reports.get(baseline, {}).get("planner_summary", {})
    delta = {
        planner: metric_delta(report["planner_summary"], baseline_metrics)
        for planner, report in planner_reports.items()
    }
    best = max(
        planner_reports,
        key=lambda name: (
            float(planner_reports[name]["planner_summary"].get("target_recall_at_10", 0.0)),
            float(planner_reports[name]["planner_summary"].get("stage_hit_at_3", 0.0)),
            -float(planner_reports[name]["planner_summary"].get("negative_leak_rate", 0.0)),
        ),
    ) if planner_reports else ""
    summary = {
        "planner_count": len(planners),
        "case_count": len(cases),
        "baseline_planner": baseline,
        "best_planner": best,
        "negative_leak_rate": planner_reports.get(best, {}).get("planner_summary", {}).get("negative_leak_rate", 0.0),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_query_understanding_comparison",
        "summary": summary,
        "planners": planner_reports,
        "planner_delta_vs_baseline": delta,
        "planner_compare": planner_compare,
        "experiment": experiment_record(args, "compare-query-understanding", summary),
    }


def fuzzy_understanding_report(args: Any) -> dict[str, Any]:
    cases = fuzzy_cases_for_command(args)
    retrieval = run_retrieval_cases(args, cases=cases, run_name=f"{planner_name(args)}::fuzzy")
    rows = first_rows(retrieval["run_rows"])
    leakage = metadata_leakage_summary(rows)
    summary = {
        "count": len(rows),
        "scene_level_recall_at_1": recall_at(rows, 1),
        "scene_level_recall_at_3": recall_at(rows, 3),
        "scene_level_recall_at_10": recall_at(rows, 10),
        "stage_level_hit_at_1": stage_hit_at(rows, 1),
        "stage_level_hit_at_3": stage_hit_at(rows, 3),
        "purpose_level_hit_at_3": purpose_hit_at(rows, 3),
        "style_violation_at_3": style_violation_at(rows, 3),
        "low_confidence_rate": low_confidence_rate(rows),
        "mean_top1_top2_margin": mean_margin(rows),
        "negative_leak_rate": retrieval.get("planner_summary", {}).get("negative_leak_rate", 0.0),
        "metadata_leak_rate": leakage["metadata_leak_rate"],
        "metadata_leak_count": leakage["metadata_leak_count"],
        "llm_call_count": retrieval.get("planner_summary", {}).get("llm_call_count", 0),
        "estimated_llm_cost_cny": retrieval.get("planner_summary", {}).get("estimated_llm_cost_cny", 0.0),
        **planner_llm_timing_summary(retrieval),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_fuzzy_understanding_validation",
        "summary": summary,
        "run_rows": retrieval["run_rows"],
        "cases": cases_from_run_rows(retrieval["run_rows"]),
        "by_variant_type": summarize_by_variant(rows),
        "by_fuzzy_set_type": summarize_by_fuzzy_set(rows),
        "metadata_leakage": leakage,
        "experiment": experiment_record(args, "validate-fuzzy-understanding", summary),
    }


def paraphrase_report(args: Any) -> dict[str, Any]:
    cases = paraphrase_variant_cases(load_command_cases(args))
    retrieval = run_retrieval_cases(args, cases=cases, run_name=f"{planner_name(args)}::paraphrase")
    rows = first_rows(retrieval["run_rows"])
    summary = {
        "count": len(rows),
        "target_recall_at_1": recall_at(rows, 1),
        "target_recall_at_3": recall_at(rows, 3),
        "target_recall_at_10": recall_at(rows, 10),
        "stage_hit_at_3": stage_hit_at(rows, 3),
        "purpose_hit_at_3": purpose_hit_at(rows, 3),
        "style_violation_at_3": style_violation_at(rows, 3),
        "low_confidence_rate": low_confidence_rate(rows),
        "negative_leak_rate": retrieval.get("planner_summary", {}).get("negative_leak_rate", 0.0),
        "llm_call_count": retrieval.get("planner_summary", {}).get("llm_call_count", 0),
        "estimated_llm_cost_cny": retrieval.get("planner_summary", {}).get("estimated_llm_cost_cny", 0.0),
        **planner_llm_timing_summary(retrieval),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_paraphrase_stress_validation",
        "summary": summary,
        "run_rows": retrieval["run_rows"],
        "cases": cases_from_run_rows(retrieval["run_rows"]),
        "by_variant_type": summarize_by_variant(rows),
        "experiment": experiment_record(args, "validate-paraphrase-stress", summary),
    }


def fuzzy_multirelevance_report(args: Any) -> dict[str, Any]:
    cases = fuzzy_cases_for_command(args)
    retrieval = run_retrieval_cases(args, cases=cases, run_name=f"{planner_name(args)}::fuzzy_multirelevance")
    run_rows = retrieval["run_rows"]
    qrels = qrels_for_report(args, run_rows=run_rows)
    rows = first_rows(run_rows)
    leakage = metadata_leakage_summary(rows)
    graded = graded_metrics(rows, qrels, top_k=int(getattr(args, "top_k", 10))) if qrels else {}
    summary = {
        **graded,
        "count": len(rows),
        "scene_level_recall_at_10": recall_at(rows, 10),
        "stage_level_hit_at_3": stage_hit_at(rows, 3),
        "purpose_level_hit_at_3": purpose_hit_at(rows, 3),
        "style_violation_at_3": style_violation_at(rows, 3),
        "metadata_leak_rate": leakage["metadata_leak_rate"],
        "metadata_leak_count": leakage["metadata_leak_count"],
        "qrels_count": len(qrels),
        "qrels_source": qrels_source(args, qrels),
        "qrels_trust_level": qrels_trust_level(qrels) if qrels else "none",
        "llm_call_count": retrieval.get("planner_summary", {}).get("llm_call_count", 0),
        "estimated_llm_cost_cny": retrieval.get("planner_summary", {}).get("estimated_llm_cost_cny", 0.0),
        **planner_llm_timing_summary(retrieval),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_fuzzy_multirelevance_evaluation",
        "summary": summary,
        "graded_metrics": graded,
        "run_rows": run_rows,
        "cases": cases_from_run_rows(run_rows),
        "by_variant_type": summarize_by_variant(rows, qrels=qrels),
        "by_fuzzy_set_type": summarize_by_fuzzy_set(rows, qrels=qrels),
        "metadata_leakage": leakage,
        "experiment": experiment_record(args, "evaluate-fuzzy-multirelevance", summary),
    }


def anti_overfit_fuzzy_report(args: Any) -> dict[str, Any]:
    base_cases = load_command_cases(args)
    top_k = int(getattr(args, "top_k", 10))
    scenarios: dict[str, dict[str, Any]] = {}
    for scenario_name, variant_mode, ranking_key in ANTI_OVERFIT_WORKFLOWS:
        scenario_args = clone_args(
            args,
            case_variants=variant_mode,
            ranking_key=ranking_key,
            planner_cache=planner_cache(args),
        )
        cases = fuzzy_cases_for_mode(base_cases, variant_mode)
        retrieval = run_retrieval_cases(
            scenario_args,
            cases=cases,
            run_name=f"{scenario_name}::{planner_name(args)}::{ranking_key}",
        )
        rows = first_rows(retrieval["run_rows"])
        qrels = qrels_for_report(scenario_args, run_rows=retrieval["run_rows"])
        leakage = metadata_leakage_summary(rows)
        scenarios[scenario_name] = {
            "variant_mode": variant_mode,
            "ranking_key": ranking_key,
            "summary": {
                **summarize_rows(rows, qrels=qrels),
                "count": len(rows),
                "metadata_leak_rate": leakage["metadata_leak_rate"],
                "metadata_leak_count": leakage["metadata_leak_count"],
                "negative_leak_rate": retrieval.get("planner_summary", {}).get("negative_leak_rate", 0.0),
                "qrels_count": len(qrels),
                "qrels_trust_level": qrels_trust_level(qrels) if qrels else "none",
                "llm_call_count": retrieval.get("planner_summary", {}).get("llm_call_count", 0),
            },
            "by_variant_type": summarize_by_variant(rows, qrels=qrels),
            "by_fuzzy_set_type": summarize_by_fuzzy_set(rows, qrels=qrels),
            "metadata_leakage": leakage,
        }
    summary = anti_overfit_summary(scenarios)
    return {
        "method": "retrieval_lab_native_anti_overfit_fuzzy_evaluation",
        "summary": summary,
        "scenarios": scenarios,
        "recommendation": anti_overfit_recommendation(scenarios),
        "experiment": experiment_record(args, "evaluate-anti-overfit-fuzzy", summary),
    }


def graded_or_pooled_report(args: Any, *, command: str) -> dict[str, Any]:
    workflow_report = workflow_evaluation_report(args, command=command, cases=load_command_cases(args))
    workflow_report["method"] = f"retrieval_lab_native_{command.replace('-', '_')}"
    return workflow_report


def build_qrels_report(args: Any, *, command: str) -> dict[str, Any]:
    workflow_report = workflow_evaluation_report(args, command=command, cases=variant_cases_for_qrels(args))
    run_rows = workflow_report["run_rows"]
    qrels = pooled_qrels_from_run_rows(run_rows)
    qrels_output = Path(getattr(args, "qrels_output", None) or DEFAULT_CORE_QRELS_PATH)
    write_qrels(qrels_output, qrels)
    summary = {
        **pooled_qrels_summary(qrels, cases_from_run_rows(run_rows), run_rows),
        **qrels_audit_summary(qrels),
        "qrels_output": str(qrels_output),
        "compat_backend_used": False,
    }
    return {
        "method": f"retrieval_lab_native_{command.replace('-', '_')}",
        "summary": summary,
        "qrels_output": str(qrels_output),
        "qrels_preview": qrels[:30],
        "run_rows": run_rows,
        "experiment": experiment_record(args, command, summary),
    }


def sample_active_qrels_report(args: Any) -> dict[str, Any]:
    workflow_report = workflow_evaluation_report(args, command="sample-active-qrels", cases=variant_cases_for_qrels(args))
    run_rows = workflow_report["run_rows"]
    qrels_path = Path(getattr(args, "qrels", "") or "")
    existing_qrels = load_qrels(qrels_path) if str(qrels_path) and qrels_path.exists() else []
    samples = active_qrels_samples(
        run_rows,
        existing_qrels=existing_qrels,
        sample_size=int(getattr(args, "sample_size", 80)),
        include_judged=bool(getattr(args, "include_judged", False)),
    )
    output = output_path(args, default=Path(".tmp") / "active_qrels_next.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples), encoding="utf-8")
    summary = {
        "sample_count": len(samples),
        "existing_qrels_count": len(existing_qrels),
        "reason_counts": dict(Counter(reason for row in samples for reason in row.get("reasons", []))),
        "output": str(output),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_active_qrels_sampler",
        "summary": summary,
        "examples": samples[:20],
        "experiment": experiment_record(args, "sample-active-qrels", summary),
    }


def rerank_upper_bound_report(args: Any, *, command: str) -> dict[str, Any]:
    retrieval = run_retrieval_cases(args, cases=variant_cases_for_qrels(args), run_name="baseline")
    baseline_rows = retrieval["run_rows"]
    qrels = qrels_for_report(args, run_rows=baseline_rows)
    rerank_depth = int(getattr(args, "rerank_depth", 20))
    top_k = int(getattr(args, "top_k", 10))
    combined = dict(baseline_rows)
    combined.update(rerank_run_rows(baseline_rows, method="rule", rerank_depth=rerank_depth, top_k=top_k))
    if qrels:
        combined.update(rerank_run_rows(baseline_rows, method="qrels_oracle", qrels=qrels, rerank_depth=rerank_depth, top_k=top_k))
    metrics = evaluate_runs_if_possible(combined, qrels=qrels, top_k=top_k)
    baseline_name = next(iter(baseline_rows), "")
    oracle_name = next((name for name in combined if "qrels_oracle" in name), "")
    opportunity = round(
        float(metrics.get(oracle_name, {}).get("nDCG@10", 0.0)) - float(metrics.get(baseline_name, {}).get("nDCG@10", 0.0)),
        6,
    )
    summary = {
        **run_artifact_summary(combined, cases_from_run_rows(combined)),
        "baseline_run": baseline_name,
        "oracle_run": oracle_name,
        "best_run": best_run_by_metrics(metrics),
        "rerank_depth": rerank_depth,
        "qrels_count": len(qrels),
        "baseline_nDCG@10": metrics.get(baseline_name, {}).get("nDCG@10", 0.0),
        "oracle_rerank_nDCG@10": metrics.get(oracle_name, {}).get("nDCG@10", 0.0),
        "rerank_opportunity_nDCG@10": opportunity,
        "llm_call_count": 0,
        "fallback_count": 0,
        "bottleneck": rerank_bottleneck(opportunity, baseline_rows),
        "compat_backend_used": False,
    }
    return {
        "method": f"retrieval_lab_native_{command.replace('-', '_')}",
        "summary": summary,
        "run_rows": combined,
        "run_metrics": metrics,
        "qrels_source": qrels_source(args, qrels),
        "experiment": experiment_record(args, command, summary),
    }


def rerank_gate_report(args: Any, *, command: str) -> dict[str, Any]:
    retrieval = run_retrieval_cases(args, cases=variant_cases_for_qrels(args), run_name="baseline")
    rows = first_rows(retrieval["run_rows"])
    selected = [gate_candidate(row) for row in rows if rerank_gate_reason(row)]
    qrels = qrels_for_report(args, run_rows=retrieval["run_rows"])
    rerank_sample = gated_rerank_sample_report(args, selected_rows(rows), qrels=qrels)
    summary = {
        "case_count": len(rows),
        "selected_count": len(selected),
        "selected_rate": round(len(selected) / max(1, len(rows)), 6),
        "reason_counts": dict(Counter(reason for row in selected for reason in row.get("gate_reasons", []))),
        "llm_call_count": rerank_sample.get("llm_call_count", 0),
        "fallback_count": rerank_sample.get("fallback_count", 0),
        "gated_sample_size": rerank_sample.get("sample_size", 0),
        "gated_sample_baseline_nDCG@10": rerank_sample.get("baseline_nDCG@10"),
        "gated_sample_llm_nDCG@10": rerank_sample.get("llm_rerank_nDCG@10"),
        "gated_sample_oracle_nDCG@10": rerank_sample.get("oracle_rerank_nDCG@10"),
        "gated_sample_llm_delta_nDCG@10": rerank_sample.get("llm_delta_nDCG@10"),
        "estimated_llm_cost_cny": rerank_sample.get("estimated_llm_cost_cny", 0.0),
        "compat_backend_used": False,
    }
    return {
        "method": f"retrieval_lab_native_{command.replace('-', '_')}",
        "summary": summary,
        "selected_cases": selected[:100],
        "gated_rerank_sample": rerank_sample,
        "run_rows": retrieval["run_rows"],
        "experiment": experiment_record(args, command, summary),
    }


def style_risk_report(args: Any, *, command: str) -> dict[str, Any]:
    cases = style_negative_cases(load_command_cases(args))
    retrieval = run_retrieval_cases(args, cases=cases, run_name=f"{planner_name(args)}::style_risk")
    rows = first_rows(retrieval["run_rows"])
    hard_negatives = hard_negative_rows(rows)
    summary = {
        "count": len(rows),
        "style_negative_violation_at_1": style_violation_at(rows, 1),
        "style_negative_violation_at_3": style_violation_at(rows, 3),
        "style_negative_violation_at_10": style_violation_at(rows, 10),
        "style_negative_expected_prefer_accuracy": 1.0 - style_violation_at(rows, 1),
        "hard_negative_candidate_count": len(hard_negatives),
        "compat_backend_used": False,
    }
    return {
        "method": f"retrieval_lab_native_{command.replace('-', '_')}",
        "summary": summary,
        "hard_negative_candidates": hard_negatives[:100],
        "run_rows": retrieval["run_rows"],
        "experiment": experiment_record(args, command, summary),
    }


def selected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if rerank_gate_reason(row)]


def gated_rerank_sample_report(args: Any, rows: list[dict[str, Any]], *, qrels: list[dict[str, Any]]) -> dict[str, Any]:
    sample_size = int(getattr(args, "llm_sample_size", 0) or 0)
    if sample_size <= 0 or not rows:
        return {"sample_size": 0, "llm_call_count": 0, "fallback_count": 0, "estimated_llm_cost_cny": 0.0}
    sample = select_gated_rerank_rows(rows, sample_size=sample_size)
    top_k = int(getattr(args, "top_k", 10))
    rerank_depth = int(getattr(args, "llm_rerank_top_n", getattr(args, "rerank_depth", 20)) or 20)
    llm_rows, llm_stats = llm_rerank_rows_for_sample(args, sample, rerank_depth=rerank_depth, top_k=top_k)
    run_rows = {
        "gated_baseline": sample,
        "gated_rule_rerank": [rerank_row_by_rule(row, rerank_depth=rerank_depth, top_k=top_k) for row in sample],
        "gated_oracle_rerank": [rerank_row_by_qrels(row, qrels, rerank_depth=rerank_depth, top_k=top_k) for row in sample] if qrels else [],
        "gated_llm_rerank": llm_rows,
    }
    run_rows = {name: rows for name, rows in run_rows.items() if rows}
    metrics = evaluate_runs_if_possible(run_rows, qrels=qrels, top_k=top_k)
    baseline = metrics.get("gated_baseline", {})
    llm_metrics = metrics.get("gated_llm_rerank", {})
    oracle_metrics = metrics.get("gated_oracle_rerank", {})
    attribution = rerank_sample_attribution(sample, llm_rows, run_rows.get("gated_oracle_rerank", []), qrels=qrels, top_k=top_k)
    return {
        "sample_size": len(sample),
        "selection_reason_counts": dict(Counter(reason for row in sample for reason in rerank_gate_reason(row))),
        "variant_counts": dict(Counter(str(row.get("variant_type", "default") or "default") for row in sample)),
        "baseline_nDCG@10": baseline.get("nDCG@10", 0.0),
        "rule_rerank_nDCG@10": metrics.get("gated_rule_rerank", {}).get("nDCG@10", 0.0),
        "oracle_rerank_nDCG@10": oracle_metrics.get("nDCG@10", 0.0),
        "llm_rerank_nDCG@10": llm_metrics.get("nDCG@10", 0.0),
        "llm_delta_nDCG@10": round(float(llm_metrics.get("nDCG@10", 0.0)) - float(baseline.get("nDCG@10", 0.0)), 6),
        "oracle_delta_nDCG@10": round(float(oracle_metrics.get("nDCG@10", 0.0)) - float(baseline.get("nDCG@10", 0.0)), 6),
        "attribution_summary": attribution["summary"],
        "attribution_examples": attribution["examples"],
        "run_metrics": metrics,
        "run_rows": run_rows if bool(getattr(args, "include_run_rows", False)) else {},
        "selected_cases": [gate_candidate(row) for row in sample],
        **llm_stats,
    }


def select_gated_rerank_rows(rows: list[dict[str, Any]], *, sample_size: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=gated_rerank_priority, reverse=True)
    return ranked[:sample_size]


def gated_rerank_priority(row: dict[str, Any]) -> tuple[float, str]:
    reasons = rerank_gate_reason(row)
    score = 0.0
    if str(row.get("variant_type", "")) == "natural_fuzzy_style":
        score += 3.0
    if "target_miss" in reasons:
        score += 2.0
    if "style_negative_present" in reasons:
        score += 1.5
    if "low_planner_confidence" in reasons:
        score += 1.0
    if "low_margin" in reasons:
        score += 0.75
    top = row.get("top_results", [])
    if len(top) >= 2:
        margin = float(top[0].get("score", 0.0)) - float(top[1].get("score", 0.0))
        score += max(0.0, 0.05 - margin)
    return (round(score, 6), str(row.get("case_id", "")))


def llm_rerank_rows_for_sample(args: Any, rows: list[dict[str, Any]], *, rerank_depth: int, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not bool(getattr(args, "require_llm", False)):
        reranked = [llm_fake_rerank_row(row, rerank_depth=rerank_depth, top_k=top_k) for row in rows]
        return reranked, {"llm_call_count": 0, "fallback_count": 0, "estimated_llm_cost_cny": 0.0, "rerank_client": "fake"}
    reranked = []
    llm_call_count = 0
    fallback_count = 0
    estimated_tokens = 0
    spent_cny = 0.0
    budget_cny = float(getattr(args, "budget_cny", 20.0) or 20.0)
    cny_per_1k = float(getattr(args, "cny_per_1k_tokens", 0.01) or 0.01)
    for row in rows:
        prompt = llm_rerank_prompt(row, rerank_depth=rerank_depth)
        tokens = max(1, int(len(prompt) / 3.5) + 256)
        cost = tokens / 1000.0 * cny_per_1k
        if spent_cny + cost > budget_cny:
            fallback_count += 1
            reranked.append(llm_fake_rerank_row(row, rerank_depth=rerank_depth, top_k=top_k))
            continue
        try:
            order = real_llm_rerank_order(row, prompt=prompt, args=args)
            llm_call_count += 1
            estimated_tokens += tokens
            spent_cny += cost
            reranked.append(row_with_reranked_results(row, ordered_results(row, order, rerank_depth=rerank_depth)[:top_k], ranking_key=f"llm_rerank@{rerank_depth}"))
        except Exception:
            fallback_count += 1
            reranked.append(llm_fake_rerank_row(row, rerank_depth=rerank_depth, top_k=top_k))
    return reranked, {
        "llm_call_count": llm_call_count,
        "fallback_count": fallback_count,
        "estimated_tokens": estimated_tokens,
        "estimated_llm_cost_cny": round(spent_cny, 6),
        "rerank_client": "real",
    }


def llm_fake_rerank_row(row: dict[str, Any], *, rerank_depth: int, top_k: int) -> dict[str, Any]:
    candidates = row.get("top_results", [])[:rerank_depth]
    reranked = sorted(candidates, key=llm_fake_rerank_score, reverse=True)
    return row_with_reranked_results(row, reranked[:top_k], ranking_key=f"llm_fake_rerank@{rerank_depth}")


def llm_fake_rerank_score(result: dict[str, Any]) -> float:
    return rule_rerank_score(result) + 0.25 * float(result.get("purpose_score", 0.0)) + 0.35 * float(result.get("signature_score", 0.0))


def llm_rerank_prompt(row: dict[str, Any], *, rerank_depth: int) -> str:
    payload = {
        "query_id": row.get("case_id"),
        "query": row.get("user_input", ""),
        "query_plan": compact_query_plan_for_rerank(row.get("query_plan", {})),
        "constraints": row.get("query_constraints", {}),
        "ranking_instruction": "Use the original rank and final score as a strong prior. Reorder only when candidate evidence clearly improves query intent, scene signature, stage/purpose fit, or style safety. Penalize explicit style risks and forbidden constraints. Keep stable ordering for ties. Judge only the candidate summaries.",
        "candidates": [rerank_prompt_candidate_summary(result, index) for index, result in enumerate(row.get("top_results", [])[:rerank_depth], start=1)],
        "response_schema": {"ordered_item_ids": ["best item_id first"], "reason": "short rationale"},
    }
    return json.dumps(payload, ensure_ascii=False)


def compact_query_plan_for_rerank(query_plan: Any) -> dict[str, Any]:
    if not isinstance(query_plan, dict):
        return {}
    keys = (
        "desired_stage",
        "forbidden_stage",
        "positive_purposes",
        "positive_style",
        "negative_style",
        "scene_signature",
        "ambiguity",
        "confidence",
    )
    return {key: query_plan[key] for key in keys if query_plan.get(key) not in (None, [], {}, "")}


def rerank_prompt_candidate_summary(result: dict[str, Any], rank: int) -> dict[str, Any]:
    summary = reranker_candidate_summary(result, rank=rank)
    components = result.get("workflow_score_components", {}) if isinstance(result.get("workflow_score_components"), dict) else {}
    scores = {
        "final": result.get("score"),
        "semantic": result.get("embedding_score", components.get("semantic")),
        "lexical": result.get("lexical_score", components.get("lexical")),
        "rrf": result.get("rrf_score", components.get("rrf")),
        "constraint": result.get("constraint_score", components.get("constraint")),
        "signature": result.get("signature_score", components.get("signature")),
        "purpose": result.get("purpose_score"),
        "style": result.get("style_score", components.get("style")),
    }
    summary["score_components"] = {key: value for key, value in scores.items() if value not in (None, "")}
    if result.get("style_guardrail_action"):
        summary["style_guardrail_action"] = result.get("style_guardrail_action")
    if result.get("risk_evidence"):
        summary["risk_evidence"] = result.get("risk_evidence")
    return summary


def real_llm_rerank_order(row: dict[str, Any], *, prompt: str, args: Any) -> list[str]:
    from sceneweaver.llm.client import VisionLLMClient

    response = VisionLLMClient().analyze_text_json(
        system_prompt="You are a strict retrieval reranker. Return JSON only with ordered_item_ids.",
        user_prompt=prompt,
        max_tokens=int(getattr(args, "max_tokens", 800) or 800),
        timeout_seconds=float(getattr(args, "timeout_seconds", 45.0) or 45.0),
        retries=int(getattr(args, "retries", 0) or 0),
        enable_thinking=bool(getattr(args, "llm_enable_thinking", False)),
        thinking_budget=int(getattr(args, "thinking_budget", 0) or 0),
    )
    raw = response.get("ordered_item_ids") or response.get("ranking") or response.get("item_ids") or []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def ordered_results(row: dict[str, Any], ordered_item_ids: list[str], *, rerank_depth: int) -> list[dict[str, Any]]:
    candidates = list(row.get("top_results", [])[:rerank_depth])
    by_id = {str(result.get("item_id", "")): result for result in candidates}
    used = set()
    ordered = []
    for item_id in ordered_item_ids:
        result = by_id.get(item_id)
        if result is None or item_id in used:
            continue
        ordered.append(result)
        used.add(item_id)
    ordered.extend(result for result in candidates if str(result.get("item_id", "")) not in used)
    return ordered


def rerank_sample_attribution(
    baseline_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    *,
    qrels: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    qrel_map = {(str(row["query_id"]), str(row["item_id"])): int(row["grade"]) for row in qrels}
    llm_by_case = {str(row.get("case_id")): row for row in llm_rows}
    oracle_by_case = {str(row.get("case_id")): row for row in oracle_rows}
    examples = []
    reason_counts: Counter[str] = Counter()
    improved = 0
    regressed = 0
    tied = 0
    for baseline in baseline_rows:
        case_id = str(baseline.get("case_id", ""))
        llm = llm_by_case.get(case_id)
        oracle = oracle_by_case.get(case_id)
        if llm is None:
            continue
        baseline_gain = graded_top_gain(baseline, qrel_map=qrel_map, top_k=top_k)
        llm_gain = graded_top_gain(llm, qrel_map=qrel_map, top_k=top_k)
        oracle_gain = graded_top_gain(oracle or {}, qrel_map=qrel_map, top_k=top_k)
        if llm_gain > baseline_gain:
            improved += 1
        elif llm_gain < baseline_gain:
            regressed += 1
        else:
            tied += 1
        reasons = rerank_attribution_reasons(baseline, llm or {}, oracle or {}, qrel_map=qrel_map, top_k=top_k)
        reason_counts.update(reasons)
        if reasons:
            examples.append(
                {
                    "case_id": case_id,
                    "user_input": baseline.get("user_input", ""),
                    "reasons": reasons,
                    "baseline_gain": baseline_gain,
                    "llm_gain": llm_gain,
                    "oracle_gain": oracle_gain,
                    "baseline_top": ranked_grade_preview(baseline, qrel_map=qrel_map, top_k=5),
                    "llm_top": ranked_grade_preview(llm, qrel_map=qrel_map, top_k=5),
                    "oracle_top": ranked_grade_preview(oracle or {}, qrel_map=qrel_map, top_k=5),
                    "best_relevant_missing_from_llm_top3": best_relevant_missing_from_top(llm, qrel_map=qrel_map, top_k=3),
                }
            )
    total = max(1, len(llm_rows))
    return {
        "summary": {
            "sample_size": len(llm_rows),
            "improved_count": improved,
            "regressed_count": regressed,
            "tied_count": tied,
            "improved_rate": round(improved / total, 6),
            "regressed_rate": round(regressed / total, 6),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "examples": examples[:20],
    }


def graded_top_gain(row: dict[str, Any], *, qrel_map: dict[tuple[str, str], int], top_k: int) -> float:
    query_id = str(row.get("case_id", ""))
    gain = 0.0
    for rank, result in enumerate(row.get("top_results", [])[:top_k], start=1):
        grade = qrel_map.get((query_id, str(result.get("item_id"))), 0)
        gain += (2**grade - 1) / rank
    return round(gain, 6)


def rerank_attribution_reasons(
    baseline: dict[str, Any],
    llm: dict[str, Any],
    oracle: dict[str, Any],
    *,
    qrel_map: dict[tuple[str, str], int],
    top_k: int,
) -> list[str]:
    query_id = str(baseline.get("case_id", ""))
    reasons = []
    baseline_top1 = grade_at_rank(baseline, query_id=query_id, qrel_map=qrel_map, rank=1)
    llm_top1 = grade_at_rank(llm, query_id=query_id, qrel_map=qrel_map, rank=1)
    oracle_top1 = grade_at_rank(oracle, query_id=query_id, qrel_map=qrel_map, rank=1)
    if llm_top1 < baseline_top1:
        reasons.append("llm_demoted_baseline_top1_grade")
    if oracle_top1 > llm_top1:
        reasons.append("llm_missed_oracle_top_grade")
    if max_grade_in_top(llm, query_id=query_id, qrel_map=qrel_map, top_k=3) < max_grade_in_top(oracle, query_id=query_id, qrel_map=qrel_map, top_k=3):
        reasons.append("relevant_candidate_not_promoted_to_top3")
    if judged_count(llm, query_id=query_id, qrel_map=qrel_map, top_k=top_k) < top_k:
        reasons.append("qrels_unjudged_candidates_in_llm_topk")
    if top_result_has_style_risk(llm):
        reasons.append("llm_promoted_style_risk")
    if target_rank_change(baseline, llm) > 0:
        reasons.append("llm_demoted_target")
    if not reasons and graded_top_gain(llm, qrel_map=qrel_map, top_k=top_k) <= graded_top_gain(baseline, qrel_map=qrel_map, top_k=top_k):
        reasons.append("no_measurable_gain")
    return reasons


def grade_at_rank(row: dict[str, Any], *, query_id: str, qrel_map: dict[tuple[str, str], int], rank: int) -> int:
    results = row.get("top_results", [])
    if len(results) < rank:
        return 0
    return qrel_map.get((query_id, str(results[rank - 1].get("item_id"))), 0)


def max_grade_in_top(row: dict[str, Any], *, query_id: str, qrel_map: dict[tuple[str, str], int], top_k: int) -> int:
    return max((qrel_map.get((query_id, str(result.get("item_id"))), 0) for result in row.get("top_results", [])[:top_k]), default=0)


def judged_count(row: dict[str, Any], *, query_id: str, qrel_map: dict[tuple[str, str], int], top_k: int) -> int:
    return sum(1 for result in row.get("top_results", [])[:top_k] if (query_id, str(result.get("item_id"))) in qrel_map)


def top_result_has_style_risk(row: dict[str, Any]) -> bool:
    top = row.get("top_results", [])[:1]
    if not top:
        return False
    hits = top[0].get("constraint_hits", {}) if isinstance(top[0].get("constraint_hits"), dict) else {}
    return bool(hits.get("negative_style"))


def target_rank_change(baseline: dict[str, Any], llm: dict[str, Any]) -> int:
    before = baseline.get("target_rank")
    after = llm.get("target_rank")
    if before is None or after is None:
        return 0
    return int(after) - int(before)


def ranked_grade_preview(row: dict[str, Any], *, qrel_map: dict[tuple[str, str], int], top_k: int) -> list[dict[str, Any]]:
    query_id = str(row.get("case_id", ""))
    preview = []
    for rank, result in enumerate(row.get("top_results", [])[:top_k], start=1):
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        preview.append(
            {
                "rank": rank,
                "item_id": result.get("item_id"),
                "grade": qrel_map.get((query_id, str(result.get("item_id"))), 0),
                "score": result.get("score"),
                "stage": metadata.get("script_stage"),
                "purposes": metadata.get("creative_purpose", []),
                "signature_score": result.get("signature_score"),
            }
        )
    return preview


def best_relevant_missing_from_top(row: dict[str, Any], *, qrel_map: dict[tuple[str, str], int], top_k: int) -> dict[str, Any] | None:
    query_id = str(row.get("case_id", ""))
    top_ids = {str(result.get("item_id")) for result in row.get("top_results", [])[:top_k]}
    candidates = []
    for rank, result in enumerate(row.get("top_results", []), start=1):
        item_id = str(result.get("item_id"))
        grade = qrel_map.get((query_id, item_id), 0)
        if grade >= 2 and item_id not in top_ids:
            candidates.append((grade, rank, result))
    if not candidates:
        return None
    grade, rank, result = sorted(candidates, key=lambda row: (-row[0], row[1]))[0]
    return {"rank": rank, "item_id": result.get("item_id"), "grade": grade, "summary": candidate_summary(result, rank)}


def mine_hard_negatives_report(args: Any) -> dict[str, Any]:
    mining_cases, mining_sources = variant_cases_for_mining(args)
    retrieval = run_retrieval_cases(args, cases=mining_cases, run_name="baseline")
    rows = first_rows(retrieval["run_rows"])
    hard_negatives = hard_negative_rows(rows)
    hard_output = Path(getattr(args, "hard_negatives_output", "") or DEFAULT_HARD_NEGATIVE_PATH)
    hard_output.parent.mkdir(parents=True, exist_ok=True)
    hard_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in hard_negatives),
        encoding="utf-8",
    )
    summary = {
        "case_count": len(rows),
        "mining_sources": mining_sources,
        "hard_negative_count": len(hard_negatives),
        "hard_negatives_output": str(hard_output),
        "reason_counts": dict(Counter(reason for row in hard_negatives for reason in row.get("reasons", []))),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_hard_negative_mining",
        "summary": summary,
        "hard_negatives": hard_negatives[:100],
        "run_rows": retrieval["run_rows"],
        "experiment": experiment_record(args, "mine-hard-negatives", summary),
    }


def native_failure_report(args: Any) -> dict[str, Any]:
    workflow_report = workflow_evaluation_report(args, command="analyze-failures", cases=load_command_cases(args))
    run_rows = workflow_report["run_rows"]
    qrels = qrels_for_report(args, run_rows=run_rows)
    failures = analyze_failure_rows(
        run_rows,
        qrels=qrels,
        top_k=int(getattr(args, "top_k", 10)),
        candidate_depth=int(getattr(args, "candidate_depth", 100)),
    )
    counts = Counter(row["failure_type"] for row in failures)
    summary = {
        **run_artifact_summary(run_rows, cases_from_run_rows(run_rows)),
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / max(1, sum(len(rows) for rows in run_rows.values())), 6),
        "failure_type_counts": dict(sorted(counts.items())),
        "top_failure_type": counts.most_common(1)[0][0] if counts else None,
        "qrels_count": len(qrels),
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_failure_analysis",
        "summary": summary,
        "failures": failures[: int(getattr(args, "max_failures", 200))],
        "run_rows": run_rows,
        "experiment": experiment_record(args, "analyze-failures", summary),
    }


def native_recall_bound_report(args: Any) -> dict[str, Any]:
    workflow_report = workflow_evaluation_report(args, command="analyze-recall-bound", cases=load_command_cases(args))
    run_rows = workflow_report["run_rows"]
    baseline = next(iter(run_rows), "")
    rows = recall_bound_rows(
        run_rows,
        baseline_key=baseline,
        candidate_depth=int(getattr(args, "candidate_depth", 100)),
        top_k=int(getattr(args, "top_k", 10)),
    )
    summary = {
        **recall_bound_summary(rows, top_k=int(getattr(args, "top_k", 10)), candidate_depth=int(getattr(args, "candidate_depth", 100))),
        "baseline_run": baseline,
        "compat_backend_used": False,
    }
    return {
        "method": "retrieval_lab_native_recall_bound_analysis",
        "summary": summary,
        "rows": rows,
        "run_rows": run_rows,
        "experiment": experiment_record(args, "analyze-recall-bound", summary),
    }


def native_build_index_report(args: Any) -> dict[str, Any]:
    manifest = build_index_manifest(
        dataset_path=Path(getattr(args, "dataset", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "test.md")),
        limit=int(getattr(args, "limit", 0)),
        index_id=str(getattr(args, "index_id", "") or ""),
    )
    index_output = output_path(args)
    write_index_manifest(index_output, manifest)
    summary = {
        "index_id": manifest.get("index_id", ""),
        "item_count": manifest.get("item_count", 0),
        "channels": manifest.get("channels", []),
        "output": str(index_output),
        "compat_backend_used": False,
    }
    return {"method": "retrieval_lab_native_build_index", "summary": summary, "manifest": manifest}


def native_compact_cache_report(args: Any) -> dict[str, Any]:
    summary = {
        "compacted": False,
        "reason": "Retrieval Lab native runtime uses compact in-memory lexical/hash-dense indexes by default.",
        "compat_backend_used": False,
    }
    return {"method": "retrieval_lab_native_compact_embedding_cache", "summary": summary}


def run_retrieval_cases(
    args: Any,
    *,
    cases: list[dict[str, Any]],
    planner: str | None = None,
    run_name: str = "",
) -> dict[str, Any]:
    return retrieval_run_from_cases(
        cases,
        dataset=str(getattr(args, "dataset", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "test.md")),
        limit=int(getattr(args, "limit", 0)),
        planner=planner or planner_name(args),
        planner_cache=planner_cache(args),
        top_k=int(getattr(args, "top_k", 10)),
        candidate_depth=int(getattr(args, "candidate_depth", 100)),
        run_name=run_name,
        ranking_key=str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature")),
        planner_config={
            "require_llm": bool(getattr(args, "require_llm", False)),
            "llm_sample_size": int(getattr(args, "llm_sample_size", 0) or 0),
            "budget_cny": float(getattr(args, "budget_cny", 20.0) or 20.0),
            "cny_per_1k_tokens": float(getattr(args, "cny_per_1k_tokens", 0.01) or 0.01),
            "max_tokens": int(getattr(args, "max_tokens", 800) or 800),
            "timeout_seconds": float(getattr(args, "timeout_seconds", 45.0) or 45.0),
            "retries": int(getattr(args, "retries", 0) or 0),
            "llm_enable_thinking": bool(getattr(args, "llm_enable_thinking", False)),
            "thinking_budget": int(getattr(args, "thinking_budget", 0) or 0),
            "natural_language_rewrites": any(
                str(case.get("fuzzy_set_type", "")) == "natural_language" for case in cases
            ),
        },
    )


def clone_args(args: Any, **overrides: Any) -> Any:
    values = vars(args).copy()
    values.update(overrides)
    return SimpleNamespace(**values)


def workflow_runs_from_retrieval(retrieval: dict[str, Any], *, ranking_keys: list[str], top_k: int) -> dict[str, list[dict[str, Any]]]:
    run_rows = retrieval.get("run_rows", {})
    if not isinstance(run_rows, dict):
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for key in ranking_keys:
        output.update(rerank_run_rows_by_workflow({str(name): list(rows) for name, rows in run_rows.items()}, ranking_key=key, top_k=top_k))
    return output


def load_command_cases(args: Any) -> list[dict[str, Any]]:
    return read_cases(
        Path(getattr(args, "dataset", DEFAULT_DATASET_PATH)),
        split=str(getattr(args, "split", "test.md")),
        limit=int(getattr(args, "limit", 0)),
    )


def variant_cases_for_qrels(args: Any) -> list[dict[str, Any]]:
    cases = load_command_cases(args)
    mode = str(getattr(args, "case_variants", "base") or "base")
    if mode in {"fuzzy", "natural_fuzzy", "natural_fuzzy_style_negative", "all_fuzzy"}:
        return fuzzy_cases_for_mode(cases, mode)
    if mode == "paraphrase":
        return paraphrase_variant_cases(cases)
    if mode == "style_negative":
        return style_negative_cases(cases)
    if mode == "all":
        return [*cases, *fuzzy_cases_for_mode(cases, "all_fuzzy"), *paraphrase_variant_cases(cases), *style_negative_cases(cases)]
    return cases


def variant_cases_for_mining(args: Any) -> tuple[list[dict[str, Any]], list[str]]:
    mode = str(getattr(args, "case_variants", "base") or "base")
    cases = load_command_cases(args)
    if mode != "base":
        return variant_cases_for_qrels(args), [mode]
    return [*cases, *fuzzy_cases_for_mode(cases, "all_fuzzy"), *style_negative_cases(cases)], ["base", "all_fuzzy", "style_negative"]


def fuzzy_cases_for_command(args: Any) -> list[dict[str, Any]]:
    cases = load_command_cases(args)
    query_file = getattr(args, "fuzzy_query_file", None)
    if query_file:
        path = Path(query_file)
        if path.exists():
            return fuzzy_cases_from_query_file(cases, path)
    mode = str(getattr(args, "case_variants", "fuzzy") or "fuzzy")
    if mode == "base":
        mode = "fuzzy"
    return fuzzy_cases_for_mode(cases, mode)


def fuzzy_cases_for_mode(cases: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "fuzzy":
        return fuzzy_variant_cases(cases)
    if mode == "natural_fuzzy":
        return natural_fuzzy_variant_cases(cases)
    if mode == "natural_fuzzy_style_negative":
        return natural_fuzzy_style_negative_cases(cases)
    if mode == "all_fuzzy":
        return [*fuzzy_variant_cases(cases), *natural_fuzzy_variant_cases(cases), *natural_fuzzy_style_negative_cases(cases)]
    return cases


def fuzzy_variant_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for case in cases:
        stage = target_field(case, "script_stage") or "scene"
        purposes = " ".join(target_field(case, "creative_purpose") or [])
        intent = purposes or stage
        templates = [
            ("implicit_stage", f"先让观众进入这一段的真实处境，服务 {intent}"),
            ("fuzzy_style", f"高级但别端着，要有人味、像纪录片，服务 {intent}"),
            ("underspecified_tone", f"自然一点，别像汇报片，适合 {purposes or '建立语境'}"),
            ("negative_style", f"不要广告感和大厂味，要真实克制，服务 {intent}"),
            ("director_brief", f"导演笔记：找一个{stage}阶段的可复用经验，重点是{purposes or '叙事功能'}"),
        ]
        variants.extend(case_variant(case, name, query, fuzzy_set_type="metadata_assisted") for name, query in templates)
    return variants


def natural_fuzzy_variant_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for case in cases:
        intent = natural_intent_text(case)
        stage_text = natural_stage_text(target_field(case, "script_stage") or "")
        templates = [
            ("natural_implicit_stage", f"先让观众自然进入这一段，重点抓住{intent}。"),
            ("natural_fuzzy_style", f"高级一点但别端着，要有人味，像人在现场观察，重点是{intent}。"),
            ("natural_underspecified_tone", f"自然克制一点，别像汇报片，帮这一段完成{intent}。"),
            ("natural_scene_signature", f"找一个{stage_text}，画面不必完全一样，但要承担{intent}。"),
            ("natural_director_brief", f"导演笔记：需要一个可复用的真实段落，让观众感到{intent}。"),
        ]
        variants.extend(case_variant(case, name, query, fuzzy_set_type="natural_language") for name, query in templates)
    return variants


def natural_fuzzy_style_negative_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        case_variant(
            case,
            "natural_negative_style",
            f"不要广告感、大厂味、口号感或卖点堆叠，保持真实克制，重点是{natural_intent_text(case)}。",
            fuzzy_set_type="natural_language",
        )
        for case in cases
    ]


def fuzzy_cases_from_query_file(cases: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    by_id = {str(case.get("case_id", "")): case for case in cases}
    rows = read_jsonl(path)
    variants = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        base_id = str(row.get("base_case_id", "") or str(row.get("case_id", "")).split("::")[0])
        base = by_id.get(base_id)
        if base is None and index < len(cases):
            base = cases[index]
        if base is None:
            continue
        variants.append(
            case_variant(
                base,
                str(row.get("variant_type", "llm_natural_fuzzy")),
                str(row.get("user_input", "") or row.get("query", "")),
                fuzzy_set_type=str(row.get("fuzzy_set_type", "natural_language") or "natural_language"),
            )
        )
    return variants


def paraphrase_variant_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = []
    for case in cases:
        original = str(case.get("user_input", ""))
        stage = target_field(case, "script_stage") or "scene"
        purposes = " ".join(target_field(case, "creative_purpose") or [])
        templates = [
            ("paraphrase_direct", original),
            ("paraphrase_brief", f"找一个能承担{purposes or stage}的段落"),
            ("paraphrase_director", f"导演想要一个{stage}功能的经验卡，不要只看画面像不像"),
            ("paraphrase_natural", f"有没有更适合这个叙事位置的真实场景，重点是{purposes or '观众进入'}"),
        ]
        variants.extend(case_variant(case, name, query) for name, query in templates)
    return variants


def style_negative_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        case_variant(
            case,
            "style_negative",
            f"{case.get('user_input', '')} 不要广告感、大厂味、炫技或产品卖点堆叠",
            fuzzy_set_type="metadata_assisted",
        )
        for case in cases
    ]


def case_variant(case: dict[str, Any], variant_type: str, user_input: str, *, fuzzy_set_type: str = "") -> dict[str, Any]:
    copied = dict(case)
    copied["case_id"] = f"{case.get('case_id', '')}::{variant_type}"
    copied["case_type"] = variant_type
    copied["variant_type"] = variant_type
    if fuzzy_set_type:
        copied["fuzzy_set_type"] = fuzzy_set_type
    copied["user_input"] = user_input
    return copied


def natural_intent_text(case: dict[str, Any]) -> str:
    purpose_values = [str(value) for value in target_field(case, "creative_purpose") or []]
    purpose_texts = [NATURAL_PURPOSE_TEXT.get(value, value.replace("_", " ")) for value in purpose_values]
    if purpose_texts:
        return "、".join(purpose_texts)
    return natural_stage_text(str(target_field(case, "script_stage") or ""))


def natural_stage_text(stage: str) -> str:
    return NATURAL_STAGE_TEXT.get(str(stage), str(stage).replace("_", " ") or "这一段的叙事功能")


def qrels_for_report(args: Any, *, run_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    raw_qrels = getattr(args, "qrels", "") or ""
    if raw_qrels:
        qrels_path = Path(raw_qrels)
    else:
        qrels_path = None
    if qrels_path is not None and qrels_path.exists() and qrels_path.is_file():
        return load_qrels(qrels_path)
    return pooled_qrels_from_run_rows(run_rows)


def qrels_source(args: Any, qrels: list[dict[str, Any]]) -> str:
    raw_qrels = getattr(args, "qrels", "") or ""
    qrels_path = Path(raw_qrels) if raw_qrels else None
    if qrels_path is not None and qrels_path.exists() and qrels_path.is_file():
        return str(qrels_path)
    return "bootstrap_from_native_run_pool" if qrels else "none"


def evaluate_runs_if_possible(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    qrels: list[dict[str, Any]],
    top_k: int,
) -> dict[str, dict[str, float]]:
    if not qrels:
        return {}
    return {name: graded_metrics(rows, qrels, top_k=top_k) for name, rows in run_rows.items()}


def summarize_rows(rows: list[dict[str, Any]], *, qrels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    summary = {
        "case_count": len(rows),
        "target_recall_at_1": recall_at(rows, 1),
        "target_recall_at_3": recall_at(rows, 3),
        "target_recall_at_10": recall_at(rows, 10),
        "stage_hit_at_1": stage_hit_at(rows, 1),
        "stage_hit_at_3": stage_hit_at(rows, 3),
        "purpose_hit_at_3": purpose_hit_at(rows, 3),
        "style_violation_at_3": style_violation_at(rows, 3),
        "low_confidence_rate": low_confidence_rate(rows),
        "mean_top1_top2_margin": mean_margin(rows),
    }
    if qrels:
        summary.update(graded_metrics(rows, qrels, top_k=10))
    return summary


def summarize_by_variant(rows: list[dict[str, Any]], *, qrels: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    variants = sorted({str(row.get("variant_type", "default") or "default") for row in rows})
    return {
        variant: summarize_rows([row for row in rows if str(row.get("variant_type", "default") or "default") == variant], qrels=qrels)
        for variant in variants
    }


def summarize_by_fuzzy_set(rows: list[dict[str, Any]], *, qrels: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    sets = sorted({str(row.get("fuzzy_set_type", "default") or "default") for row in rows})
    return {
        fuzzy_set: {
            **summarize_rows([row for row in rows if str(row.get("fuzzy_set_type", "default") or "default") == fuzzy_set], qrels=qrels),
            **metadata_leakage_summary([row for row in rows if str(row.get("fuzzy_set_type", "default") or "default") == fuzzy_set]),
        }
        for fuzzy_set in sets
    }


def metadata_leakage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = []
    leak_count = 0
    for row in rows:
        labels = metadata_label_leaks_for_row(row)
        if not labels:
            continue
        leak_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "case_id": row.get("case_id", ""),
                    "variant_type": row.get("variant_type", ""),
                    "fuzzy_set_type": row.get("fuzzy_set_type", ""),
                    "leaked_labels": labels,
                    "user_input": row.get("user_input", ""),
                }
            )
    return {
        "metadata_leak_count": leak_count,
        "metadata_leak_rate": round(leak_count / max(1, len(rows)), 6),
        "leaked_label_examples": examples,
    }


def metadata_label_leaks_for_row(row: dict[str, Any]) -> list[str]:
    plan = row.get("query_plan", {}) if isinstance(row.get("query_plan"), dict) else {}
    texts = [str(row.get("user_input", "")), str(plan.get("positive_query", "")), str(plan.get("hyde_text", ""))]
    texts.extend(str(rewrite.get("text", "")) for rewrite in plan.get("rewrites", []) if isinstance(rewrite, dict))
    return metadata_label_leaks(" ".join(texts))


def metadata_label_leaks(text: str) -> list[str]:
    lower = str(text or "").lower()
    return sorted(label for label in INTERNAL_METADATA_LABELS if label in lower)


def anti_overfit_summary(scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata_safe = scenarios.get("metadata_assisted_style_safe", {}).get("summary", {})
    natural_safe = scenarios.get("natural_fuzzy_style_safe", {}).get("summary", {})
    unsafe_natural = scenarios.get("natural_fuzzy_hybrid", {}).get("summary", {})
    summary = {
        "scenario_count": len(scenarios),
        "metadata_assisted_style_safe_stage_hit_at_3": metadata_safe.get("stage_hit_at_3", 0.0),
        "natural_style_safe_stage_hit_at_3": natural_safe.get("stage_hit_at_3", 0.0),
        "metadata_assisted_style_safe_purpose_hit_at_3": metadata_safe.get("purpose_hit_at_3", 0.0),
        "natural_style_safe_purpose_hit_at_3": natural_safe.get("purpose_hit_at_3", 0.0),
        "natural_style_safe_scene_recall_at_10": natural_safe.get("target_recall_at_10", 0.0),
        "natural_style_safe_style_violation_at_3": natural_safe.get("style_violation_at_3", 0.0),
        "natural_style_safe_metadata_leak_rate": natural_safe.get("metadata_leak_rate", 0.0),
        "natural_hybrid_style_violation_at_3": unsafe_natural.get("style_violation_at_3", 0.0),
        "stage_hit_delta_natural_vs_metadata": round(
            float(natural_safe.get("stage_hit_at_3", 0.0)) - float(metadata_safe.get("stage_hit_at_3", 0.0)),
            6,
        ),
        "purpose_hit_delta_natural_vs_metadata": round(
            float(natural_safe.get("purpose_hit_at_3", 0.0)) - float(metadata_safe.get("purpose_hit_at_3", 0.0)),
            6,
        ),
        "style_guardrail_delta": round(
            float(natural_safe.get("style_violation_at_3", 0.0)) - float(unsafe_natural.get("style_violation_at_3", 0.0)),
            6,
        ),
        "llm_call_count": sum(int(row.get("summary", {}).get("llm_call_count", 0) or 0) for row in scenarios.values()),
        "compat_backend_used": False,
    }
    summary["bottleneck"] = anti_overfit_bottleneck(summary)
    return summary


def anti_overfit_bottleneck(summary: dict[str, Any]) -> str:
    if float(summary.get("natural_style_safe_metadata_leak_rate", 0.0)) > 0.0:
        return "metadata_leakage"
    if float(summary.get("natural_style_safe_style_violation_at_3", 0.0)) > 0.05:
        return "style_guardrail"
    if float(summary.get("natural_style_safe_stage_hit_at_3", 0.0)) < 0.8:
        return "natural_query_understanding"
    if float(summary.get("natural_style_safe_scene_recall_at_10", 0.0)) < 0.7:
        return "candidate_recall"
    return "validated_with_natural_fuzzy"


def anti_overfit_recommendation(scenarios: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = anti_overfit_summary(scenarios)
    bottleneck = summary["bottleneck"]
    if bottleneck == "validated_with_natural_fuzzy":
        action = "Keep style_safe_signature as the fuzzy-safe workflow and move next effort to LLM qrels coverage."
    elif bottleneck == "metadata_leakage":
        action = "Block internal labels from natural fuzzy planner positive text before trusting the metric."
    elif bottleneck == "natural_query_understanding":
        action = "Use a small LLM planner sample to improve natural fuzzy stage and purpose parsing."
    elif bottleneck == "style_guardrail":
        action = "Mine remaining natural style-negative failures and strengthen style risk aliases."
    else:
        action = "Inspect candidate recall for natural fuzzy misses before adding rerank complexity."
    return {"bottleneck": bottleneck, "next_action": action}


def first_rows(run_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return next(iter(run_rows.values()), [])


def first_run_name_for_key(run_rows: dict[str, list[dict[str, Any]]], key: str) -> str:
    return next((name for name in run_rows if name.endswith(f"::{key}") or key in name), "")


def first_rows_for_key(run_rows: dict[str, list[dict[str, Any]]], key: str) -> list[dict[str, Any]]:
    name = first_run_name_for_key(run_rows, key)
    return run_rows.get(name, [])


def command_ranking_keys(args: Any, command: str) -> list[str]:
    if getattr(args, "ranking_keys", ""):
        return csv_values(str(getattr(args, "ranking_keys")))
    if command == "evaluate":
        return [str(getattr(args, "ranking_key", "hybrid_rrf_constraints_signature"))]
    if command == "validate-ranking-keys":
        return ["semantic_only", "lexical_only", "hybrid_rrf", "hybrid_rrf_constraints", "hybrid_rrf_constraints_signature", "adaptive_signature"]
    return list(DEFAULT_WORKFLOW_KEYS)


def planner_name(args: Any) -> str:
    return str(getattr(args, "query_planner", "") or getattr(args, "planner", "") or "multi_query")


def planner_cache(args: Any) -> Path | None:
    if bool(getattr(args, "no_cache", False)):
        return None
    value = getattr(args, "planner_cache", DEFAULT_PLANNER_CACHE_PATH)
    return Path(value) if value else None


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def target_field(case: dict[str, Any], field: str) -> Any:
    target = case.get("expected_prefer") if isinstance(case.get("expected_prefer"), dict) else case.get("target", {})
    if not isinstance(target, dict):
        return [] if field == "creative_purpose" else ""
    return target.get(field, [] if field == "creative_purpose" else "")


def recall_at(rows: list[dict[str, Any]], k: int) -> float:
    hits = sum(1 for row in rows if any(result.get("item_id") == row.get("target_item_id") for result in row.get("top_results", [])[:k]))
    return round(hits / max(1, len(rows)), 6)


def stage_hit_at(rows: list[dict[str, Any]], k: int) -> float:
    hits = 0
    for row in rows:
        stage = row.get("target_stage")
        if stage and any(result.get("metadata", {}).get("script_stage") == stage for result in row.get("top_results", [])[:k]):
            hits += 1
    return round(hits / max(1, len(rows)), 6)


def purpose_hit_at(rows: list[dict[str, Any]], k: int) -> float:
    hits = 0
    for row in rows:
        target_purposes = set(row.get("target_purposes", []) or [])
        if not target_purposes:
            continue
        if any(target_purposes & set(result.get("metadata", {}).get("creative_purpose", []) or []) for result in row.get("top_results", [])[:k]):
            hits += 1
    return round(hits / max(1, len(rows)), 6)


def style_violation_at(rows: list[dict[str, Any]], k: int) -> float:
    violations = sum(
        1
        for row in rows
        if any(result.get("constraint_hits", {}).get("negative_style") for result in row.get("top_results", [])[:k])
    )
    return round(violations / max(1, len(rows)), 6)


def low_confidence_rate(rows: list[dict[str, Any]]) -> float:
    return round(sum(1 for row in rows if rerank_gate_reason(row)) / max(1, len(rows)), 6)


def mean_margin(rows: list[dict[str, Any]]) -> float:
    margins = []
    for row in rows:
        top = row.get("top_results", [])
        if len(top) >= 2:
            margins.append(float(top[0].get("score", 0.0)) - float(top[1].get("score", 0.0)))
    return round(sum(margins) / max(1, len(margins)), 6)


def metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("target_recall_at_1", "target_recall_at_3", "target_recall_at_10", "stage_hit_at_3", "purpose_hit_at_3", "style_violation_at_3", "negative_leak_rate")
    return {key: round(float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6) for key in keys}


def best_run_by_metrics(metrics: dict[str, dict[str, float]]) -> str:
    if not metrics:
        return ""
    return max(metrics, key=lambda name: (metrics[name].get("nDCG@10", 0.0), metrics[name].get("MRR@10", 0.0), metrics[name].get("Recall@10", 0.0)))


def best_workflow_by_summary(workflows: dict[str, dict[str, Any]]) -> str:
    if not workflows:
        return ""
    return max(workflows, key=lambda name: workflows[name]["summary"].get("target_recall_at_10", 0.0))


def rerank_bottleneck(opportunity: float, run_rows: dict[str, list[dict[str, Any]]]) -> str:
    rows = first_rows(run_rows)
    recall10 = recall_at(rows, 10)
    if recall10 < 0.7:
        return "candidate_recall_or_query_understanding"
    if opportunity >= 0.15:
        return "reranking"
    return "qrels_or_minor_ranking"


def rerank_gate_reason(row: dict[str, Any]) -> list[str]:
    reasons = []
    top = row.get("top_results", [])
    if len(top) >= 2 and float(top[0].get("score", 0.0)) - float(top[1].get("score", 0.0)) < 0.05:
        reasons.append("low_margin")
    if float(row.get("planner_confidence", 1.0) or 1.0) < 0.6:
        reasons.append("low_planner_confidence")
    if any(result.get("constraint_hits", {}).get("negative_style") for result in top[:3]):
        reasons.append("style_negative_present")
    if not any(result.get("item_id") == row.get("target_item_id") for result in top[:10]):
        reasons.append("target_miss")
    return reasons


def gate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "user_input": row.get("user_input", ""),
        "target_item_id": row.get("target_item_id"),
        "target_rank": row.get("target_rank"),
        "gate_reasons": rerank_gate_reason(row),
        "top_candidates": [candidate_summary(result, index) for index, result in enumerate(row.get("top_results", [])[:5], start=1)],
    }


def candidate_summary(result: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    return {
        "rank": rank,
        "item_id": result.get("item_id"),
        "score": result.get("score"),
        "script_stage": metadata.get("script_stage"),
        "creative_purpose": metadata.get("creative_purpose", []),
        "style_traits": metadata.get("style_traits", []),
        "style_risks": metadata.get("style_risks", []),
        "constraint_hits": result.get("constraint_hits", {}),
    }


def hard_negative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mined = []
    for row in rows:
        for rank, result in enumerate(row.get("top_results", [])[:10], start=1):
            reasons = []
            if result.get("constraint_hits", {}).get("negative_style"):
                reasons.append("style_violation")
                reasons.append("style_risk_candidate")
                if rank <= 3:
                    reasons.append("top3_style_risk")
            if result.get("constraint_hits", {}).get("forbidden_stage"):
                reasons.append("forbidden_stage_violation")
            if rank == 1 and result.get("item_id") != row.get("target_item_id"):
                reasons.append("top1_false_positive")
            if result.get("item_id") == row.get("target_item_id"):
                continue
            if not reasons:
                continue
            mined.append(
                {
                    "query_id": row.get("case_id"),
                    "user_input": row.get("user_input", ""),
                    "item_id": result.get("item_id"),
                    "rank": rank,
                    "score": result.get("score"),
                    "target_item_id": row.get("target_item_id"),
                    "metadata": result.get("metadata", {}),
                    "constraint_hits": result.get("constraint_hits", {}),
                    "style_risk_score": result.get("style_risk_score", 0),
                    "style_guardrail_action": result.get("style_guardrail_action", ""),
                    "risk_evidence": result.get("risk_evidence", []),
                    "reasons": reasons,
                    "source": "retrieval_lab_native_mining",
                }
            )
    return mined


def planner_llm_timing_summary(retrieval: dict[str, Any]) -> dict[str, Any]:
    planner_summary = retrieval.get("planner_summary", {})
    if not isinstance(planner_summary, dict):
        return {}
    keys = (
        "llm_fallback_count",
        "llm_timing_count",
        "llm_request_seconds_total",
        "llm_request_seconds_avg",
        "llm_total_seconds_total",
        "llm_total_seconds_avg",
        "llm_max_request_seconds",
        "llm_prompt_chars_total",
        "llm_response_chars_total",
    )
    return {key: planner_summary[key] for key in keys if key in planner_summary}


def write_core_outputs(args: Any, report: dict[str, Any]) -> None:
    output = output_path(args)
    if report.get("method") == "retrieval_lab_native_active_qrels_sampler":
        return
    if report.get("method") == "retrieval_lab_native_build_index":
        return
    write_json(output, compact_report(report) if bool(getattr(args, "compact_output", False)) else report)
    markdown_output = getattr(args, "markdown_output", None)
    if markdown_output:
        path = Path(markdown_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown_report(report), encoding="utf-8")


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    compacted = dict(report)
    compacted.pop("run_rows", None)
    compacted.pop("cases", None)
    for key in ("workflows", "planners", "by_variant_type"):
        value = compacted.get(key)
        if isinstance(value, dict):
            compacted[key] = {
                name: {k: v for k, v in row.items() if k != "run_rows"} if isinstance(row, dict) else row
                for name, row in value.items()
            }
    compacted["compact_output"] = True
    return compacted


def output_path(args: Any, default: Path = DEFAULT_CORE_REPORT_PATH) -> Path:
    return Path(getattr(args, "output", None) or default)


def experiment_record(args: Any, command: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": command,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "summary": summary,
    }


__all__ = [
    "CORE_EXPERIMENT_COMMANDS",
    "DEFAULT_CORE_MARKDOWN_PATH",
    "DEFAULT_CORE_QRELS_PATH",
    "DEFAULT_CORE_REPORT_PATH",
    "DEFAULT_HARD_NEGATIVE_PATH",
    "native_core_experiment_command",
]
