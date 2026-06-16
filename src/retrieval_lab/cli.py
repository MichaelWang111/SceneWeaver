from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from retrieval_lab.architecture import layer_manifest
from retrieval_lab.artifacts import DEFAULT_ARTIFACT_MANIFEST_PATH, write_artifact_manifest_command
from retrieval_lab.capability import generate_capability_report_command, record_capability_cycle_command
from retrieval_lab.config import project_paths
from retrieval_lab.compat import COMMAND_ALIASES, LEGACY_COMMANDS, run_mocktesting_backend, translate_argv
from retrieval_lab.datasets import DEFAULT_DATASET_MANIFEST_PATH, DEFAULT_DATASET_PATH, VALID_SPLITS, inspect_dataset_command
from retrieval_lab.experiments import (
    DEFAULT_EXPERIMENT_COMPARISON_PATH,
    DEFAULT_INFRA_AUDIT_REPORT_PATH,
    DEFAULT_LEGACY_MANIFEST_PATH,
    DEFAULT_LEGACY_REPORT_PATH,
    DEFAULT_MIGRATION_AUDIT_REPORT_PATH,
    DEFAULT_MIGRATION_CERTIFICATION_REPORT_PATH,
    DEFAULT_RUN_ARTIFACT_PATH,
    audit_infra_coverage_command,
    compare_experiments_command,
    export_run_artifact_command,
    migration_audit_command,
    migration_certify_command,
    retrieval_lab_flywheel_guide,
    run_legacy_with_artifacts_command,
    write_flywheel_guide,
)
from retrieval_lab.experiments.constants import CORE_EXPERIMENT_COMMANDS
from retrieval_lab.experiments.core import native_core_experiment_command
from retrieval_lab.planners import (
    DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT,
    DEFAULT_PLANNER_CACHE_PATH,
    DEFAULT_PLANNER_COMPARE_OUTPUT,
    DEFAULT_PLANNER_PLAN_OUTPUT,
    PLANNER_NAMES,
    planner_audit_cache_command,
    planner_compare_command,
    planner_plan_command,
)
from retrieval_lab.evaluators import (
    DEFAULT_FAILURE_REPORT_PATH,
    DEFAULT_ROUND2_OUTPUT_DIR,
    DEFAULT_RUN_EVAL_REPORT_PATH,
    analyze_failures_from_runs_command,
    evaluate_run_artifact_command,
    round2_taxonomy_gate_report_command,
)
from retrieval_lab.indexes import DEFAULT_INDEX_MANIFEST_PATH, index_inspect_command, index_manifest_command
from retrieval_lab.graph import DEFAULT_SCENE_GRAPH_MANIFEST, DEFAULT_SCENE_GRAPH_REPORT, build_scene_graph_manifest_command
from retrieval_lab.llm import (
    DEFAULT_LLM_ADJUDICATION_OUTPUT,
    DEFAULT_LLM_ADJUDICATION_REPORT,
    DEFAULT_LLM_JUDGE_CACHE,
    DEFAULT_LLM_NATURAL_FUZZY_OUTPUT,
    DEFAULT_LLM_NATURAL_FUZZY_REPORT,
    llm_adjudicate_qrels_command,
    llm_generate_natural_fuzzy_command,
    llm_status_command,
)
from retrieval_lab.llm.budget_guard import DEFAULT_LLM_USAGE_LEDGER
from retrieval_lab.qrels.commands import (
    DEFAULT_ADJUDICATED_QRELS_PATH,
    DEFAULT_ACTIVE_QRELS_SAMPLE_PATH,
    DEFAULT_COVERAGE_QRELS_SAMPLE_PATH,
    DEFAULT_POOLED_QRELS_PATH,
    DEFAULT_POOLED_QRELS_REPORT_PATH,
    DEFAULT_QRELS_AUDIT_REPORT_PATH,
    audit_qrels_command,
    merge_adjudicated_qrels_command,
    pool_qrels_from_runs_command,
    sample_active_qrels_from_runs_command,
    sample_coverage_qrels_from_runs_command,
)
from retrieval_lab.qrels import DEFAULT_JUDGE_CALIBRATION_REPORT, judge_calibration_command
from retrieval_lab.ranking import (
    DEFAULT_CALIBRATED_MODEL_PATH,
    DEFAULT_CALIBRATED_REPORT_PATH,
    DEFAULT_CALIBRATED_RUNS_PATH,
    DEFAULT_RERANK_ATTRIBUTION_PATH,
    DEFAULT_RERANK_FEATURE_REPORT_PATH,
    DEFAULT_RERANK_FEATURES_PATH,
    DEFAULT_WORKFLOW_COMPARISON_PATH,
    DEFAULT_RERANKED_RUN_ARTIFACT_PATH,
    DEFAULT_WORKFLOW_RUN_ARTIFACT_PATH,
    WORKFLOW_RANKING_KEYS,
    apply_calibrated_rerank_command,
    attribute_rerank_command,
    calibrate_rerank_command,
    compare_workflow_runs_command,
    export_rerank_features_command,
    rerank_run_artifact_command,
    workflow_run_artifact_command,
)
from retrieval_lab.reports import (
    DEFAULT_EVAL_REPORT_INPUT,
    DEFAULT_EVAL_REPORT_OUTPUT,
    generate_eval_report_command,
    markdown_report,
)
from retrieval_lab.retrieval import (
    DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT,
    DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT,
    DEFAULT_RETRIEVAL_RUN_OUTPUT,
    retrieval_benchmark_command,
    retrieval_compare_legacy_command,
    retrieval_run_command,
)
from retrieval_lab.schemas import (
    DEFAULT_SCHEMA_CATALOG_PATH,
    DEFAULT_SCHEMA_REPORT_PATH,
    schema_catalog_command,
    schema_show_command,
    schema_validate_command,
)


HELP_TEXT = """Retrieval Lab - SceneWeaver retrieval experiment platform
Usage:
  python -m retrieval_lab <group> <command> [options]
  python -m retrieval_lab <legacy-mocktesting-command> [options]

Modern command groups:
  flywheel guide
  dataset inspect
  qrels build-pooled | pool-from-runs | audit | sample-active | sample-active-from-runs | merge-adjudicated
  eval fuzzy | graded | pooled | rerank-upper-bound | failures | recall-bound
  eval round2-taxonomy-gate
  cycle record
  report capability | eval
  artifact manifest
  run export | legacy | rerank | workflow-rerank | analyze-failures | evaluate
  infra audit
  migration audit | certify
  schema catalog | show | validate
  index inspect | manifest
  retrieval run | compare-legacy
  benchmark retrieval
  llm status | adjudicate-qrels | generate-natural-fuzzy
  workflow compare | compare-runs
  rerank export-features | calibrate | apply-calibrated | attribute
  qrels judge-calibration
  graph build-manifest
  planner plan | compare | audit-cache
  index build | compact-cache
  retrieval search

Examples:
  python -m retrieval_lab flywheel guide
  python -m retrieval_lab qrels audit --qrels .tmp\\pooled_qrels_next.jsonl
  python -m retrieval_lab migration audit --round-id round_001
  python -m retrieval_lab planner plan --planner multi_query --query "need grounded setup without product pitch"
  python -m retrieval_lab index manifest --split test --limit 60
  python -m retrieval_lab retrieval run --split test --limit 60 --planner multi_query
  python -m retrieval_lab workflow compare-runs --runs .tmp\\retrieval_lab\\retrieval_run_latest.json
  python -m retrieval_lab rerank export-features --runs .tmp\\retrieval_lab\\retrieval_run_latest.json --qrels .tmp\\retrieval_lab\\pooled_qrels.jsonl
  python -m retrieval_lab graph build-manifest --split test --limit 60
  python -m retrieval_lab schema catalog
  python -m retrieval_lab eval fuzzy --split test --limit 60
  python -m retrieval_lab cycle record --cycle-id cycle_002 --reports .tmp\\qrels_audit_next.json

Compatibility:
  Existing mocktesting commands are still accepted unchanged.
"""


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP_TEXT)
        return
    if args[0] == "architecture":
        print(json.dumps({"layers": layer_manifest(), "paths": project_paths()}, ensure_ascii=False, indent=2))
        return
    if args[0] == "commands":
        print(
            json.dumps(
                {
                    "modern_aliases": {" ".join(key): value for key, value in sorted(COMMAND_ALIASES.items())},
                    "legacy_commands": sorted(LEGACY_COMMANDS),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args[:2] == ["flywheel", "guide"]:
        guide = retrieval_lab_flywheel_guide()
        output_path = option_value(args[2:], "--output")
        if output_path:
            write_flywheel_guide(Path(output_path), guide)
        print(json.dumps(guide["summary"], ensure_ascii=False, indent=2))
        return
    translated = translate_argv(args)
    if run_native_infra_command(translated):
        return
    run_mocktesting_backend(translated)


def run_native_infra_command(args: Sequence[str]) -> bool:
    if not args:
        return False
    command = args[0]
    if command == "audit-qrels":
        namespace = audit_qrels_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = audit_qrels_command(namespace)
        write_json(namespace.output, result)
        if namespace.markdown_output is not None:
            namespace.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            namespace.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "inspect-dataset":
        namespace = dataset_inspect_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = inspect_dataset_command(namespace)
        if namespace.markdown_output is not None:
            report = json.loads(namespace.output.read_text(encoding="utf-8"))
            namespace.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            namespace.markdown_output.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "benchmark-retrieval":
        namespace = retrieval_benchmark_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = retrieval_benchmark_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "pool-qrels-from-runs":
        namespace = pool_qrels_from_runs_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = pool_qrels_from_runs_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "sample-active-qrels-from-runs":
        namespace = sample_active_qrels_from_runs_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = sample_active_qrels_from_runs_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "sample-coverage-qrels-from-runs":
        namespace = sample_coverage_qrels_from_runs_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = sample_coverage_qrels_from_runs_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "merge-adjudicated-qrels":
        namespace = merge_qrels_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = merge_adjudicated_qrels_command(namespace)
        write_json(namespace.report_output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "llm-adjudicate-qrels":
        namespace = llm_adjudicate_qrels_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = llm_adjudicate_qrels_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "llm-generate-natural-fuzzy":
        namespace = llm_generate_natural_fuzzy_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = llm_generate_natural_fuzzy_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "llm-status":
        namespace = llm_status_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = llm_status_command(namespace)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return True
    if command == "record-capability-cycle":
        namespace = capability_record_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = record_capability_cycle_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "generate-capability-report":
        namespace = capability_report_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = generate_capability_report_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "write-artifact-manifest":
        namespace = artifact_manifest_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = write_artifact_manifest_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "export-run-artifact":
        namespace = run_export_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = export_run_artifact_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "run-legacy-with-artifacts":
        namespace = run_legacy_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = run_legacy_with_artifacts_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "rerank-run-artifact":
        namespace = run_rerank_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = rerank_run_artifact_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "workflow-run-artifact":
        namespace = run_workflow_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = workflow_run_artifact_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "workflow-compare-runs":
        namespace = workflow_compare_runs_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = compare_workflow_runs_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "rerank-export-features":
        namespace = rerank_export_features_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = export_rerank_features_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "rerank-calibrate":
        namespace = rerank_calibrate_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = calibrate_rerank_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "rerank-apply-calibrated":
        namespace = rerank_apply_calibrated_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = apply_calibrated_rerank_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "rerank-attribute":
        namespace = rerank_attribute_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = attribute_rerank_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "qrels-judge-calibration":
        namespace = judge_calibration_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = judge_calibration_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "graph-build-manifest":
        namespace = graph_build_manifest_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = build_scene_graph_manifest_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "analyze-failures-from-runs":
        namespace = run_failure_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = analyze_failures_from_runs_command(namespace)
        if namespace.markdown_output is not None:
            report = json.loads(namespace.output.read_text(encoding="utf-8"))
            namespace.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            namespace.markdown_output.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "evaluate-runs":
        namespace = run_evaluate_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = evaluate_run_artifact_command(namespace)
        if namespace.markdown_output is not None:
            report = json.loads(namespace.output.read_text(encoding="utf-8"))
            namespace.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            namespace.markdown_output.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "round2-taxonomy-gate":
        namespace = round2_taxonomy_gate_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = round2_taxonomy_gate_report_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "compare-experiments":
        namespace = compare_experiments_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = compare_experiments_command(namespace)
        if namespace.markdown_output is not None:
            report = json.loads(namespace.output.read_text(encoding="utf-8"))
            namespace.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            namespace.markdown_output.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "generate-eval-report":
        namespace = eval_report_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = generate_eval_report_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "audit-infra-coverage":
        namespace = infra_audit_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = audit_infra_coverage_command(namespace)
        if namespace.markdown_output is not None:
            report = json.loads(namespace.output.read_text(encoding="utf-8"))
            namespace.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            namespace.markdown_output.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "migration-audit":
        namespace = migration_audit_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = migration_audit_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "migration-certify":
        namespace = migration_certify_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = migration_certify_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command in CORE_EXPERIMENT_COMMANDS:
        namespace = native_core_parser(command).parse_args(list(args[1:]))
        namespace.command = command
        result = native_core_experiment_command(namespace, command)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "planner-plan":
        namespace = planner_plan_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = planner_plan_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "planner-compare":
        namespace = planner_compare_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = planner_compare_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "planner-audit-cache":
        namespace = planner_audit_cache_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = planner_audit_cache_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "index-inspect":
        namespace = index_manifest_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = index_inspect_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "index-manifest":
        namespace = index_manifest_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = index_manifest_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "retrieval-run":
        namespace = retrieval_run_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = retrieval_run_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "retrieval-compare-legacy":
        namespace = retrieval_compare_legacy_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = retrieval_compare_legacy_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "schema-catalog":
        namespace = schema_catalog_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = schema_catalog_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "schema-show":
        namespace = schema_show_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = schema_show_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    if command == "schema-validate":
        namespace = schema_validate_parser().parse_args(list(args[1:]))
        namespace.command = command
        result = schema_validate_command(namespace)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return True
    return False


def audit_qrels_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab qrels audit")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_QRELS_AUDIT_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def dataset_inspect_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab dataset inspect")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_MANIFEST_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def pool_qrels_from_runs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab qrels pool-from-runs")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels-output", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_POOLED_QRELS_REPORT_PATH)
    parser.add_argument("--baseline-run", default="")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def sample_active_qrels_from_runs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab qrels sample-active-from-runs")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTIVE_QRELS_SAMPLE_PATH)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--case-variants", choices=["base", "fuzzy", "paraphrase", "style_negative", "all"], default="base")
    parser.add_argument("--include-judged", action="store_true")
    return parser


def sample_coverage_qrels_from_runs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab qrels sample-coverage-from-runs")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_COVERAGE_QRELS_SAMPLE_PATH)
    parser.add_argument("--sample-size", type=int, default=0, help="0 samples exactly the reviewed-count gap needed for high trust.")
    return parser


def merge_qrels_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab qrels merge-adjudicated")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ADJUDICATED_QRELS_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_QRELS_AUDIT_REPORT_PATH)
    parser.add_argument("--default-judge-type", choices=["human", "llm"], default="human")
    parser.add_argument("--default-judge-id", default="adjudicator")
    parser.add_argument("--judge-version", default="v1")
    return parser


def llm_adjudicate_qrels_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab llm adjudicate-qrels")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_LLM_ADJUDICATION_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_LLM_ADJUDICATION_REPORT)
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--merged-qrels-output", type=Path, default=None)
    parser.add_argument("--judge-cache", type=Path, default=DEFAULT_LLM_JUDGE_CACHE)
    parser.add_argument("--llm-sample-size", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=0, help="Manual batch size override; 0 enables auto batching.")
    parser.add_argument("--max-batch-size", type=int, default=10)
    parser.add_argument("--max-batch-tokens", type=int, default=6000)
    parser.add_argument("--min-batch-size-for-concurrency", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--budget-cny", type=float, default=20.0)
    parser.add_argument("--cny-per-1k-tokens", type=float, default=0.01)
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--fake-client", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=0)
    add_hard_budget_args(parser)
    return parser


def llm_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab llm status")
    parser.add_argument("--provider", choices=["auto", "deepseek", "dashscope", "bailian", "aliyun"], default="auto")
    parser.add_argument("--live-models", action="store_true", help="Query the provider /models endpoint instead of using the local registry.")
    parser.add_argument("--check-balance", action="store_true", help="Query provider account balance when supported and credentials are configured.")
    parser.add_argument("--include-models", action="store_true", help="Include model rows in the JSON output.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser


def llm_generate_natural_fuzzy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab llm generate-natural-fuzzy")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_LLM_NATURAL_FUZZY_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_LLM_NATURAL_FUZZY_REPORT)
    parser.add_argument("--llm-sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--budget-cny", type=float, default=20.0)
    parser.add_argument("--cny-per-1k-tokens", type=float, default=0.01)
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=0)
    add_hard_budget_args(parser)
    return parser


def add_hard_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hard-budget-cny", type=float, default=0.0, help="Enable provider-enforced hard budget mode when > 0.")
    parser.add_argument("--budget-safety-cny", type=float, default=0.05)
    parser.add_argument("--provider", choices=["auto", "deepseek", "dashscope", "bailian", "aliyun"], default="auto")
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_LLM_USAGE_LEDGER)
    parser.add_argument("--balance-check-interval-seconds", type=float, default=10.0)


def capability_record_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab cycle record")
    parser.add_argument("--cycle-id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--reports", type=Path, nargs="*", default=None)
    parser.add_argument("--registry", type=Path, default=Path(".tmp") / "capability_cycles.jsonl")
    parser.add_argument("--output", type=Path, default=Path(".tmp") / "capability_cycle_latest.json")
    parser.add_argument("--as-origin", action="store_true")
    return parser


def capability_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab report capability")
    parser.add_argument("--registry", type=Path, default=Path(".tmp") / "capability_cycles.jsonl")
    parser.add_argument("--output", type=Path, default=Path(".tmp") / "capability_report.md")
    parser.add_argument("--chart-dir", type=Path, default=Path(".tmp") / "capability_charts")
    return parser


def artifact_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab artifact manifest")
    parser.add_argument("--manifest-id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--outputs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_MANIFEST_PATH)
    return parser


def run_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab run export")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_ARTIFACT_PATH)
    parser.add_argument("--run-name", default="")
    return parser


def run_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab run legacy")
    parser.add_argument("--legacy-command", required=True)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_LEGACY_REPORT_PATH)
    parser.add_argument("--run-output", type=Path, default=DEFAULT_RUN_ARTIFACT_PATH)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_LEGACY_MANIFEST_PATH)
    parser.add_argument("--manifest-id", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("legacy_args", nargs=argparse.REMAINDER)
    return parser


def run_rerank_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab run rerank")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--method", choices=["rule", "qrels_oracle"], default="rule")
    parser.add_argument("--qrels", type=Path, default=Path(""))
    parser.add_argument("--output", type=Path, default=DEFAULT_RERANKED_RUN_ARTIFACT_PATH)
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def run_workflow_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab run workflow-rerank")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--ranking-key", choices=sorted(WORKFLOW_RANKING_KEYS), default="hybrid_rrf_constraints")
    parser.add_argument("--output", type=Path, default=DEFAULT_WORKFLOW_RUN_ARTIFACT_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def workflow_compare_runs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab workflow compare-runs")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--ranking-keys", default="hybrid_rrf_constraints,hybrid_rrf_constraints_signature,adaptive_signature")
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_WORKFLOW_COMPARISON_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def rerank_export_features_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab rerank export-features")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=Path(""))
    parser.add_argument("--feature-profile", choices=["diagnostic", "production_safe"], default="diagnostic")
    parser.add_argument("--output", type=Path, default=DEFAULT_RERANK_FEATURES_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_RERANK_FEATURE_REPORT_PATH)
    return parser


def rerank_calibrate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab rerank calibrate")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--method", choices=["linear", "coordinate_search", "qrels_oracle_upper"], default="linear")
    parser.add_argument("--split-strategy", choices=["none", "query_hash", "fixture_holdout"], default="none")
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--exclude-features", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATED_MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_CALIBRATED_REPORT_PATH)
    return parser


def rerank_apply_calibrated_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab rerank apply-calibrated")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATED_RUNS_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def rerank_attribute_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab rerank attribute")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=Path(""))
    parser.add_argument("--features", type=Path, default=Path(""))
    parser.add_argument("--output", type=Path, default=DEFAULT_RERANK_ATTRIBUTION_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--max-failures", type=int, default=200)
    return parser


def judge_calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab qrels judge-calibration")
    parser.add_argument("--samples", type=Path, default=Path(""))
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_JUDGE_CALIBRATION_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--repeat-count", type=int, default=0)
    parser.add_argument("--shuffle-candidate-order", action="store_true")
    parser.add_argument("--budget-cny", type=float, default=20.0)
    return parser


def graph_build_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab graph build-manifest")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCENE_GRAPH_MANIFEST)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_SCENE_GRAPH_REPORT)
    return parser


def run_failure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab run analyze-failures")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=Path(""))
    parser.add_argument("--output", type=Path, default=DEFAULT_FAILURE_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--max-failures", type=int, default=200)
    return parser


def run_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab run evaluate")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_EVAL_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--baseline-run", default="")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def round2_taxonomy_gate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab eval round2-taxonomy-gate")
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--blind-analysis", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, default=Path(""))
    parser.add_argument("--round1-gate", type=Path, default=Path(""))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROUND2_OUTPUT_DIR)
    return parser


def compare_experiments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab experiment compare")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPERIMENT_COMPARISON_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def eval_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab report eval")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--inputs", type=Path, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_REPORT_OUTPUT)
    return parser


def infra_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab infra audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_INFRA_AUDIT_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def migration_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab migration audit")
    parser.add_argument("--round-id", default="round")
    parser.add_argument("--label", default="")
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_MIGRATION_AUDIT_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser


def migration_certify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab migration certify")
    parser.add_argument("--round-id", default="certification")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_MIGRATION_CERTIFICATION_REPORT_PATH)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--parity-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--test-reports", type=Path, nargs="*", default=[])
    return parser


def native_core_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"retrieval_lab {command}")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--planner", choices=PLANNER_NAMES, default="multi_query")
    parser.add_argument("--query-planner", choices=PLANNER_NAMES, default="")
    parser.add_argument("--query-planners", default="")
    parser.add_argument("--planners", default="")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_PLANNER_CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--ranking-key", choices=sorted(WORKFLOW_RANKING_KEYS), default="hybrid_rrf_constraints_signature")
    parser.add_argument("--ranking-keys", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument(
        "--case-variants",
        choices=[
            "base",
            "fuzzy",
            "natural_fuzzy",
            "natural_fuzzy_style_negative",
            "all_fuzzy",
            "paraphrase",
            "style_negative",
            "all",
        ],
        default="base",
    )
    parser.add_argument("--fuzzy-query-file", type=Path, default=None)
    parser.add_argument("--include-judged", action="store_true")
    parser.add_argument("--qrels", type=Path, default=None)
    parser.add_argument("--qrels-output", type=Path, default=None)
    parser.add_argument("--hard-negatives-output", type=Path, default=None)
    parser.add_argument("--index-id", default="")
    parser.add_argument("--llm-sample-size", type=int, default=0)
    parser.add_argument("--llm-rerank-top-n", type=int, default=20)
    parser.add_argument("--judge-cache", type=Path, default=None)
    parser.add_argument("--budget-cny", type=float, default=20.0)
    parser.add_argument("--cny-per-1k-tokens", type=float, default=0.01)
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--llm-enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--include-run-rows", action="store_true")
    parser.add_argument("--include-debug-text", action="store_true")
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".tmp") / "retrieval_lab" / f"{command.replace('-', '_')}_latest.json")
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--max-failures", type=int, default=200)
    return parser


def retrieval_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab benchmark retrieval")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat-to", type=int, default=1000)
    parser.add_argument("--planner", choices=PLANNER_NAMES, default="multi_query")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_PLANNER_CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ranking-key", choices=sorted(WORKFLOW_RANKING_KEYS), default="hybrid_rrf_constraints_signature")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=10)
    parser.add_argument("--compact-output", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT)
    return parser


def planner_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab planner plan")
    add_planner_query_args(parser)
    parser.add_argument("--planner", choices=PLANNER_NAMES, default="rule")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_PLANNER_CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLANNER_PLAN_OUTPUT)
    parser.add_argument("--jsonl-output", type=Path, default=None)
    return parser


def planner_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab planner compare")
    add_planner_query_args(parser)
    parser.add_argument("--planners", default="rule,multi_query,hyde_card")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_PLANNER_CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_PLANNER_COMPARE_OUTPUT)
    return parser


def planner_audit_cache_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab planner audit-cache")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_PLANNER_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLANNER_CACHE_AUDIT_OUTPUT)
    return parser


def index_manifest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab index manifest")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index-id", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_MANIFEST_PATH)
    return parser


def retrieval_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab retrieval run")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--planner", choices=PLANNER_NAMES, default="multi_query")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_PLANNER_CACHE_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_RETRIEVAL_RUN_OUTPUT)
    return parser


def retrieval_compare_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab retrieval compare-legacy")
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT)
    return parser


def add_planner_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--limit", type=int, default=0)


def schema_catalog_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab schema catalog")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_CATALOG_PATH)
    parser.add_argument("--include-json-schema", action="store_true")
    return parser


def schema_show_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab schema show")
    parser.add_argument("name")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def schema_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_lab schema validate")
    parser.add_argument("name")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_REPORT_PATH)
    parser.add_argument("--max-issues", type=int, default=50)
    return parser


def option_value(args: Sequence[str], option: str) -> str | None:
    for index, value in enumerate(args):
        if value == option and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(f"{option}="):
            return value.split("=", 1)[1]
    return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
