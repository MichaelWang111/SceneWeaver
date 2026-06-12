from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

from retreieval_lab.architecture import layer_manifest
from retreieval_lab.artifacts import artifact_manifest, data_sha256
from retreieval_lab.config import project_paths
from retreieval_lab.compat import translate_argv
from retreieval_lab.evaluators import (
    analyze_failure_rows,
    classify_failure_from_artifact,
    evaluate_run_rows,
    graded_metrics,
    recall_bound_rows,
    recall_bound_summary,
    run_metric_selection_score,
)
from retreieval_lab.experiments import extract_run_rows_from_report
from retreieval_lab.experiments.compare import extract_report_metrics
from retreieval_lab.experiments.legacy import with_output_option
from retreieval_lab.planners import compare_planners, plan_many, planner_cache_key
from retreieval_lab.qrels import active_qrels_samples, pooled_qrels_from_run_rows, pooled_qrels_summary, qrels_audit_summary
from retreieval_lab.ranking import (
    rerank_row_by_qrels,
    rerank_row_by_rule,
    rerank_run_rows,
    rerank_run_rows_by_workflow,
    workflow_score,
)
from retreieval_lab.indexes import build_index_manifest, index_items_from_cases
from retreieval_lab.retrieval import retrieval_run, score_item
from retreieval_lab.schemas import schema_catalog, validate_record, validate_records


def test_retreieval_lab_command_aliases_are_decision_complete():
    assert translate_argv(["flywheel", "guide"]) == ["retrieval-flywheel-guide"]
    assert translate_argv(["qrels", "audit", "--qrels", "x.jsonl"]) == ["audit-qrels", "--qrels", "x.jsonl"]
    assert translate_argv(["qrels", "pool-from-runs", "--runs", "r.json"]) == ["pool-qrels-from-runs", "--runs", "r.json"]
    assert translate_argv(["qrels", "sample-active-from-runs", "--runs", "r.json"]) == [
        "sample-active-qrels-from-runs",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["artifact", "manifest", "--inputs", "x.json"]) == [
        "write-artifact-manifest",
        "--inputs",
        "x.json",
    ]
    assert translate_argv(["run", "export", "--reports", "r.json"]) == ["export-run-artifact", "--reports", "r.json"]
    assert translate_argv(["run", "legacy", "--legacy-command", "retrieval-flywheel-guide"]) == [
        "run-legacy-with-artifacts",
        "--legacy-command",
        "retrieval-flywheel-guide",
    ]
    assert translate_argv(["run", "rerank", "--runs", "r.json"]) == ["rerank-run-artifact", "--runs", "r.json"]
    assert translate_argv(["run", "workflow-rerank", "--runs", "r.json"]) == [
        "workflow-run-artifact",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["workflow", "compare-runs", "--runs", "r.json"]) == [
        "workflow-compare-runs",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["run", "analyze-failures", "--runs", "r.json"]) == [
        "analyze-failures-from-runs",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["run", "evaluate", "--runs", "r.json"]) == ["evaluate-runs", "--runs", "r.json"]
    assert translate_argv(["eval", "runs", "--runs", "r.json"]) == ["evaluate-runs", "--runs", "r.json"]
    assert translate_argv(["experiment", "compare", "--reports", "a.json"]) == ["compare-experiments", "--reports", "a.json"]
    assert translate_argv(["infra", "audit"]) == ["audit-infra-coverage"]
    assert translate_argv(["schema", "catalog"]) == ["schema-catalog"]
    assert translate_argv(["schema", "show", "query_plan"]) == ["schema-show", "query_plan"]
    assert translate_argv(["schema", "validate", "qrel", "--input", "q.jsonl"]) == [
        "schema-validate",
        "qrel",
        "--input",
        "q.jsonl",
    ]
    assert translate_argv(["migration", "audit"]) == ["migration-audit"]
    assert translate_argv(["planner", "plan", "--query", "x"]) == ["planner-plan", "--query", "x"]
    assert translate_argv(["planner", "compare", "--query", "x"]) == ["planner-compare", "--query", "x"]
    assert translate_argv(["planner", "audit-cache"]) == ["planner-audit-cache"]
    assert translate_argv(["index", "inspect"]) == ["index-inspect"]
    assert translate_argv(["index", "manifest"]) == ["index-manifest"]
    assert translate_argv(["retrieval", "run"]) == ["retrieval-run"]
    assert translate_argv(["retrieval", "compare-legacy"]) == ["retrieval-compare-legacy"]
    assert translate_argv(["eval", "fuzzy", "--limit", "6"]) == ["evaluate-fuzzy-multirelevance", "--limit", "6"]
    assert translate_argv(["cycle", "record", "--cycle-id", "c1"]) == ["record-capability-cycle", "--cycle-id", "c1"]
    assert translate_argv(["report", "eval", "--input", "r.json"]) == ["generate-eval-report", "--input", "r.json"]
    assert translate_argv(["retrieval-flywheel-guide"]) == ["retrieval-flywheel-guide"]


def test_retreieval_lab_architecture_manifest_names_core_layers():
    layers = layer_manifest()
    names = {row["name"] for row in layers}

    assert {
        "datasets",
        "planners",
        "indexes",
        "retrieval",
        "ranking",
        "qrels",
        "evaluators",
        "experiments",
        "schemas",
        "reports",
        "capability",
        "artifacts",
        "config",
    } <= names

    paths = project_paths()
    assert paths["package_name"] == "retreieval_lab"
    assert paths["legacy_baseline_package"] == "mocktesting"


def run_module(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "retreieval_lab", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_retreieval_lab_flywheel_guide_uses_modern_commands(tmp_path):
    output = tmp_path / "guide.json"
    result = subprocess.run(
        [sys.executable, "-m", "retreieval_lab", "flywheel", "guide", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    guide = json.loads(output.read_text(encoding="utf-8"))

    assert summary["step_count"] == len(guide["steps"])
    assert all("python -m retreieval_lab" in row["command"] for row in guide["steps"])
    assert any(row["name"] == "cycle record" for row in guide["steps"])


def test_retreieval_lab_legacy_flywheel_passthrough_matches_mocktesting_summary():
    modern = run_module("flywheel", "guide")
    legacy_passthrough = run_module("retrieval-flywheel-guide")
    legacy_result = subprocess.run(
        [sys.executable, "-m", "mocktesting.mock_retriever", "retrieval-flywheel-guide"],
        check=True,
        capture_output=True,
        text=True,
    )
    legacy = json.loads(legacy_result.stdout)

    assert legacy_passthrough == legacy
    assert modern["step_count"] < legacy["step_count"] or modern["recommended_first_loop"] != legacy["recommended_first_loop"]
    assert legacy["step_count"] >= 10


def test_retreieval_lab_facade_exports_qrels_services():
    rows = [
        {
            "query_id": "q1",
            "item_id": "i1",
            "grade": 3,
            "reason": "manual",
            "source": "manual_adjudicated",
            "grade_votes": [{"grade": 3, "judge_type": "human", "confidence": 0.98}],
        }
    ]
    summary = qrels_audit_summary(rows)

    assert summary["manual_count"] == 1
    assert summary["qrels_trust_level"] in {"medium", "high"}


def test_retreieval_lab_qrels_audit_matches_legacy_summary_for_core_fields():
    rows = [
        {
            "query_id": "q1",
            "item_id": "i1",
            "grade": 3,
            "reason": "target item",
            "source": "pooled_bootstrap",
            "pooled_from": ["a", "b", "c"],
            "grade_votes": [{"grade": 3, "judge_type": "bootstrap", "confidence": 0.8}],
        },
        {
            "query_id": "q1",
            "item_id": "i2",
            "grade": 0,
            "reason": "bad",
            "source": "llm_adjudicated",
            "grade_votes": [{"grade": 0, "judge_type": "llm", "confidence": 0.82}],
        },
    ]
    from mocktesting.mock_retriever import qrels_audit_summary as legacy_qrels_audit_summary

    modern = qrels_audit_summary(rows)
    legacy = legacy_qrels_audit_summary(rows)

    for key in (
        "qrels_count",
        "manual_count",
        "llm_count",
        "bootstrap_only_count",
        "needs_adjudication_count",
        "qrels_trust_level",
    ):
        assert modern[key] == legacy[key]


def test_retreieval_lab_evaluator_metrics_match_legacy_for_representative_rows():
    rows = [
        {
            "case_id": "q1",
            "top_results": [{"item_id": "i2"}, {"item_id": "i1"}, {"item_id": "i3"}],
        },
        {
            "case_id": "q2",
            "top_results": [{"item_id": "i5"}, {"item_id": "i4"}],
        },
    ]
    qrels = [
        {"query_id": "q1", "item_id": "i1", "grade": 3},
        {"query_id": "q1", "item_id": "i2", "grade": 1},
        {"query_id": "q1", "item_id": "i3", "grade": 2},
        {"query_id": "q2", "item_id": "i4", "grade": 2},
    ]
    from mocktesting.mock_retriever import graded_metrics as legacy_graded_metrics

    assert graded_metrics(rows, qrels, top_k=10) == legacy_graded_metrics(rows, qrels, top_k=10)


def test_retreieval_lab_recall_bound_metrics_match_legacy():
    rows_by_key = {
        "baseline": [
            {
                "case_id": "q1",
                "user_input": "find setup",
                "target_item_id": "i1",
                "target_rank": 12,
                "top_results": [{"item_id": "i2"}],
            }
        ],
        "hybrid": [
            {
                "case_id": "q1",
                "target_rank": 3,
                "top_results": [{"item_id": "i3"}],
            }
        ],
    }
    from mocktesting.mock_retriever import recall_bound_rows as legacy_rows
    from mocktesting.mock_retriever import recall_bound_summary as legacy_summary

    modern_rows = recall_bound_rows(rows_by_key, baseline_key="baseline", candidate_depth=20, top_k=10)
    legacy_bound_rows = legacy_rows(rows_by_key, baseline_key="baseline", candidate_depth=20, top_k=10)

    assert modern_rows == legacy_bound_rows
    assert recall_bound_summary(modern_rows, top_k=10, candidate_depth=20) == legacy_summary(
        legacy_bound_rows,
        top_k=10,
        candidate_depth=20,
    )


def representative_run_rows() -> dict[str, list[dict]]:
    return {
        "rule::hybrid": [
            {
                "case_id": "q1",
                "user_input": "need a grounded setup",
                "target_item_id": "i-target",
                "target_stage": "setup",
                "target_purposes": ["build_trust"],
                "target_rank": 12,
                "top_results": [
                    {
                        "item_id": "i-style-risk",
                        "score": 0.91,
                        "metadata": {
                            "script_stage": "setup",
                            "creative_purpose": ["build_trust"],
                            "style_risks": ["ad_like"],
                        },
                        "constraint_hits": {"negative_style": ["ad_like"]},
                    },
                    {
                        "item_id": "i-partial",
                        "score": 0.83,
                        "metadata": {"script_stage": "setup", "creative_purpose": ["establish_context"]},
                    },
                ],
            },
        ],
        "hyde::hybrid": [
            {
                "case_id": "q1",
                "user_input": "need a grounded setup",
                "target_item_id": "i-target",
                "target_stage": "setup",
                "target_purposes": ["build_trust"],
                "target_rank": 1,
                "top_results": [
                    {
                        "item_id": "i-target",
                        "score": 0.95,
                        "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"]},
                    },
                    {
                        "item_id": "i-style-risk",
                        "score": 0.6,
                        "metadata": {
                            "script_stage": "setup",
                            "creative_purpose": ["build_trust"],
                            "style_risks": ["ad_like"],
                        },
                        "constraint_hits": {"negative_style": ["ad_like"]},
                    },
                    {
                        "item_id": "i-partial",
                        "score": 0.51,
                        "metadata": {"script_stage": "setup", "creative_purpose": ["establish_context"]},
                    },
                ],
            }
        ],
    }


def representative_rerank_row() -> dict:
    return {
        "case_id": "q-rerank",
        "user_input": "need setup without ad polish",
        "target_item_id": "i-safe",
        "target_stage": "setup",
        "target_purposes": ["build_trust"],
        "target_rank": 2,
        "top_results": [
            {
                "item_id": "i-risk",
                "score": 0.92,
                "constraint_score": 0.0,
                "signature_score": 0.0,
                "constraint_hits": {"negative_style": ["ad_like"]},
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"]},
            },
            {
                "item_id": "i-safe",
                "score": 0.82,
                "constraint_score": 0.2,
                "signature_score": 0.4,
                "constraint_hits": {"desired_stage": ["setup"]},
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"]},
            },
            {
                "item_id": "i-other",
                "score": 0.4,
                "constraint_score": 0.0,
                "signature_score": 0.0,
                "constraint_hits": {},
                "metadata": {"script_stage": "outcome", "creative_purpose": ["show_outcome"]},
            },
        ],
    }


def representative_workflow_row() -> dict:
    return {
        "case_id": "q-workflow",
        "user_input": "need setup without opening",
        "target_item_id": "i-target",
        "target_stage": "setup",
        "target_purposes": ["build_trust"],
        "query_constraints": {"forbidden_stage": ["opening"]},
        "query_plan": {"ambiguity": {"level": "medium"}},
        "top_results": [
            {
                "item_id": "i-veto",
                "score": 0.99,
                "embedding_score": 0.99,
                "lexical_score": 0.2,
                "rrf_score": 4.0,
                "constraint_score": 0.0,
                "signature_score": 0.7,
                "metadata": {"script_stage": "opening", "creative_purpose": ["build_trust"]},
                "constraint_hits": {},
                "channel_scores": {"script_use": 0.4, "visual_tags": 0.9},
            },
            {
                "item_id": "i-target",
                "score": 0.7,
                "embedding_score": 0.7,
                "lexical_score": 0.8,
                "rrf_score": 3.0,
                "constraint_score": 0.2,
                "signature_score": 0.9,
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"]},
                "constraint_hits": {"desired_stage": ["setup"]},
                "channel_scores": {"script_use": 0.95, "visual_tags": 0.2},
            },
            {
                "item_id": "i-semantic",
                "score": 0.85,
                "embedding_score": 0.85,
                "lexical_score": 0.1,
                "rrf_score": 2.0,
                "constraint_score": 0.0,
                "signature_score": 0.1,
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"]},
                "constraint_hits": {},
                "channel_scores": {"script_use": 0.2, "visual_tags": 0.1},
            },
        ],
    }


def representative_failure_rows() -> dict[str, list[dict]]:
    base = representative_workflow_row()
    return {
        "base": [
            {
                **base,
                "case_id": "q-style",
                "target_rank": 4,
                "top_results": [
                    {
                        **base["top_results"][1],
                        "item_id": "i-risk",
                        "constraint_hits": {"negative_style": ["ad_like"]},
                    },
                    base["top_results"][1],
                ],
            },
            {
                **base,
                "case_id": "q-fusion",
                "target_rank": 4,
                "top_results": [
                    {
                        **base["top_results"][2],
                        "metadata": {"script_stage": "setup", "creative_purpose": ["show_outcome"]},
                    },
                    base["top_results"][1],
                ],
                "all_results": [
                    {
                        **base["top_results"][2],
                        "metadata": {"script_stage": "setup", "creative_purpose": ["show_outcome"]},
                    },
                    {"item_id": "filler-a", "score": 0.6, "metadata": {"script_stage": "setup"}},
                    {"item_id": "filler-b", "score": 0.5, "metadata": {"script_stage": "setup"}},
                    base["top_results"][1],
                ],
            },
            {
                **base,
                "case_id": "q-recall",
                "target_rank": None,
                "top_results": [base["top_results"][2]],
                "all_results": [base["top_results"][2]],
            },
            {
                **base,
                "case_id": "q-ambiguous",
                "target_rank": 5,
                "top_results": [
                    {
                        **base["top_results"][2],
                        "item_id": "i-alt",
                        "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"]},
                    },
                    base["top_results"][1],
                ],
            },
        ]
    }


def test_retreieval_lab_pooled_qrels_match_legacy_for_representative_runs():
    run_rows = representative_run_rows()
    cases = [{"case_id": "q1"}]
    from mocktesting.mock_retriever import pooled_qrels_from_run_rows as legacy_pooled_qrels_from_run_rows
    from mocktesting.mock_retriever import pooled_qrels_summary as legacy_pooled_qrels_summary

    modern = pooled_qrels_from_run_rows(run_rows)
    legacy = legacy_pooled_qrels_from_run_rows(run_rows)

    assert modern == legacy
    assert pooled_qrels_summary(modern, cases, run_rows) == legacy_pooled_qrels_summary(legacy, cases, run_rows)


def test_retreieval_lab_active_qrels_samples_match_legacy_for_representative_runs():
    run_rows = representative_run_rows()
    existing_qrels = [
        {
            "query_id": "q1",
            "item_id": "i-partial",
            "grade": 1,
            "reason": "weak bootstrap",
            "source": "pooled_bootstrap",
            "grade_votes": [{"grade": 1, "judge_type": "bootstrap", "confidence": 0.45}],
        }
    ]
    from mocktesting.mock_retriever import active_qrels_samples as legacy_active_qrels_samples

    modern = active_qrels_samples(run_rows, existing_qrels=existing_qrels, sample_size=20, include_judged=False)
    legacy = legacy_active_qrels_samples(run_rows, existing_qrels=existing_qrels, sample_size=20, include_judged=False)

    assert modern == legacy
    assert any("style_risk_candidate" in row["reasons"] for row in modern)
    assert any(row.get("existing_qrel", {}).get("needs_adjudication") for row in modern if row["item_id"] == "i-partial")


def test_retreieval_lab_pool_from_runs_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    qrels_output = tmp_path / "pooled.jsonl"
    report_output = tmp_path / "pooled_report.json"
    runs.write_text(json.dumps({"run_rows": representative_run_rows()}), encoding="utf-8")

    summary = run_module(
        "qrels",
        "pool-from-runs",
        "--runs",
        str(runs),
        "--qrels-output",
        str(qrels_output),
        "--report-output",
        str(report_output),
        "--baseline-run",
        "rule::hybrid",
    )
    report = json.loads(report_output.read_text(encoding="utf-8"))
    qrels = [json.loads(line) for line in qrels_output.read_text(encoding="utf-8").splitlines()]

    assert summary["qrels_count"] == len(qrels)
    assert summary["run_count"] == 2
    assert report["method"] == "retrieval_lab_pool_qrels_from_runs"
    assert any(row["item_id"] == "i-target" and row["grade"] == 3 for row in qrels)


def test_retreieval_lab_sample_active_from_runs_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "active.jsonl"
    runs.write_text(json.dumps({"run_rows": representative_run_rows()}), encoding="utf-8")
    qrels.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "item_id": "i-partial",
                "grade": 1,
                "reason": "weak bootstrap",
                "source": "pooled_bootstrap",
                "grade_votes": [{"grade": 1, "judge_type": "bootstrap", "confidence": 0.45}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module(
        "qrels",
        "sample-active-from-runs",
        "--runs",
        str(runs),
        "--qrels",
        str(qrels),
        "--output",
        str(output),
        "--sample-size",
        "20",
    )
    samples = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["sample_count"] == len(samples)
    assert summary["reason_counts"]["style_risk_candidate"] >= 1
    assert any(row["item_id"] == "i-partial" and row["existing_qrel"]["needs_adjudication"] for row in samples)


def test_retreieval_lab_extracts_run_rows_from_report_cases():
    run_rows = representative_run_rows()
    report = {
        "method": "mock_fuzzy_understanding",
        "query_planner": "multi_query",
        "ranking_key": "adaptive_signature",
        "cases": run_rows["rule::hybrid"],
    }

    result = extract_run_rows_from_report(report)

    assert result["source"]["skipped"] is False
    assert list(result["run_rows"]) == ["multi_query::adaptive_signature"]
    assert result["run_rows"]["multi_query::adaptive_signature"] == run_rows["rule::hybrid"]


def test_retreieval_lab_extracts_run_rows_from_workflow_cases():
    report = {
        "method": "mock_workflow_comparison_debug",
        "workflows": {
            "semantic_only": {"cases": representative_run_rows()["rule::hybrid"]},
            "hybrid_rrf": {"metrics": {"recall_at_10": 1.0}},
        },
    }

    result = extract_run_rows_from_report(report)

    assert result["source"]["extracted_run_count"] == 1
    assert list(result["run_rows"]) == ["semantic_only"]


def test_retreieval_lab_run_export_records_metrics_only_reports_as_skipped(tmp_path):
    report = tmp_path / "workflow_summary.json"
    output = tmp_path / "runs.json"
    report.write_text(
        json.dumps(
            {
                "method": "mock_workflow_comparison",
                "workflows": {"semantic_only": {"metrics": {"recall_at_10": 0.5}}},
            }
        ),
        encoding="utf-8",
    )

    summary = run_module("run", "export", "--reports", str(report), "--output", str(output))
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert summary["run_count"] == 0
    assert summary["skipped_report_count"] == 1
    assert artifact["source_reports"][0]["skip_reason"] == "no ranked rows found"


def test_retreieval_lab_run_export_cli_feeds_pool_from_runs(tmp_path):
    report = tmp_path / "fuzzy_report.json"
    runs_output = tmp_path / "runs.json"
    qrels_output = tmp_path / "qrels.jsonl"
    qrels_report = tmp_path / "qrels_report.json"
    report.write_text(
        json.dumps(
            {
                "method": "mock_fuzzy_understanding",
                "query_planner": "multi_query",
                "ranking_key": "adaptive_signature",
                "cases": representative_run_rows()["rule::hybrid"],
            }
        ),
        encoding="utf-8",
    )

    run_summary = run_module("run", "export", "--reports", str(report), "--output", str(runs_output))
    qrels_summary = run_module(
        "qrels",
        "pool-from-runs",
        "--runs",
        str(runs_output),
        "--qrels-output",
        str(qrels_output),
        "--report-output",
        str(qrels_report),
    )

    assert run_summary["run_count"] == 1
    assert run_summary["row_count"] == 1
    assert qrels_summary["qrels_count"] >= 2
    assert qrels_output.exists()


def test_retreieval_lab_legacy_bridge_replaces_output_option():
    assert with_output_option(["--split", "test"], "x.json") == ["--split", "test", "--output", "x.json"]
    assert with_output_option(["--output", "old.json", "--split", "test"], "new.json") == [
        "--output",
        "new.json",
        "--split",
        "test",
    ]
    assert with_output_option(["--output=old.json"], "new.json") == ["--output=new.json"]


def test_retreieval_lab_run_legacy_cli_writes_report_run_and_manifest(tmp_path):
    report = tmp_path / "legacy_report.json"
    run_output = tmp_path / "run_rows.json"
    manifest = tmp_path / "manifest.json"

    summary = run_module(
        "run",
        "legacy",
        "--legacy-command",
        "retrieval-flywheel-guide",
        "--report-output",
        str(report),
        "--run-output",
        str(run_output),
        "--manifest-output",
        str(manifest),
    )
    report_data = json.loads(report.read_text(encoding="utf-8"))
    run_data = json.loads(run_output.read_text(encoding="utf-8"))
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))

    assert summary["legacy_command"] == "retrieval-flywheel-guide"
    assert summary["run_count"] == 0
    assert summary["skipped_report_count"] == 1
    assert "flywheel" in report_data["method"]
    assert run_data["method"] == "retrieval_lab_run_artifact"
    assert manifest_data["metadata"]["legacy_command"] == "retrieval-flywheel-guide"


def test_retreieval_lab_rule_rerank_matches_legacy():
    row = representative_rerank_row()
    from mocktesting.mock_retriever import rerank_row_by_rule as legacy_rerank_row_by_rule

    assert rerank_row_by_rule(row, rerank_depth=3, top_k=2) == legacy_rerank_row_by_rule(
        row,
        rerank_depth=3,
        top_k=2,
    )


def test_retreieval_lab_qrels_oracle_rerank_matches_legacy():
    row = representative_rerank_row()
    qrels = [
        {"query_id": "q-rerank", "item_id": "i-risk", "grade": 0},
        {"query_id": "q-rerank", "item_id": "i-safe", "grade": 3},
    ]
    from mocktesting.mock_retriever import rerank_row_by_qrels as legacy_rerank_row_by_qrels

    assert rerank_row_by_qrels(row, qrels, rerank_depth=3, top_k=2) == legacy_rerank_row_by_qrels(
        row,
        qrels,
        rerank_depth=3,
        top_k=2,
    )


def test_retreieval_lab_rerank_run_rows_outputs_standard_run_mapping():
    reranked = rerank_run_rows(
        {"base": [representative_rerank_row()]},
        method="rule",
        rerank_depth=3,
        top_k=2,
    )
    run_name = next(iter(reranked))

    assert run_name == "base::rule_rerank@3"
    assert reranked[run_name][0]["top_results"][0]["item_id"] == "i-safe"
    assert reranked[run_name][0]["ranking_key"] == "rule_rerank@3"


def test_retreieval_lab_run_rerank_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    output = tmp_path / "reranked.json"
    runs.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_run_artifact",
                "run_rows": {"base": [representative_rerank_row()]},
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_module(
        "run",
        "rerank",
        "--runs",
        str(runs),
        "--method",
        "rule",
        "--rerank-depth",
        "3",
        "--top-k",
        "2",
        "--output",
        str(output),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert summary["run_count"] == 1
    assert artifact["method"] == "retrieval_lab_reranked_run_artifact"
    assert artifact["run_rows"]["base::rule_rerank@3"][0]["top_results"][0]["item_id"] == "i-safe"


def test_retreieval_lab_workflow_score_uses_existing_signals_and_forbidden_veto():
    row = representative_workflow_row()
    target = row["top_results"][1]
    veto = row["top_results"][0]

    assert workflow_score(target, ranking_key="hybrid_rrf_constraints_signature", row=row) == 3.515
    assert workflow_score(target, ranking_key="adaptive_signature", row=row) == 3.605
    assert workflow_score(veto, ranking_key="hybrid_rrf_constraints", row=row) == -996.0
    assert workflow_score(target, ranking_key="script_use_only", row=row) == 0.95


def test_retreieval_lab_workflow_rerank_run_rows_outputs_standard_mapping():
    reranked = rerank_run_rows_by_workflow(
        {"base": [representative_workflow_row()]},
        ranking_key="hybrid_rrf_constraints",
        top_k=2,
    )
    run_name = next(iter(reranked))
    row = reranked[run_name][0]

    assert run_name == "base::hybrid_rrf_constraints"
    assert row["ranking_key"] == "hybrid_rrf_constraints"
    assert row["top_results"][0]["item_id"] == "i-target"
    assert row["target_rank"] == 1
    assert row["top_results"][0]["workflow_score_components"]["constraint"] == 0.2


def test_retreieval_lab_workflow_rerank_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    output = tmp_path / "workflow.json"
    runs.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_run_artifact",
                "run_rows": {"base": [representative_workflow_row()]},
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    summary = run_module(
        "run",
        "workflow-rerank",
        "--runs",
        str(runs),
        "--ranking-key",
        "hybrid_rrf_constraints",
        "--top-k",
        "2",
        "--output",
        str(output),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert summary["run_count"] == 1
    assert artifact["method"] == "retrieval_lab_workflow_run_artifact"
    assert artifact["run_rows"]["base::hybrid_rrf_constraints"][0]["top_results"][0]["item_id"] == "i-target"


def test_retreieval_lab_workflow_compare_runs_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    output = tmp_path / "workflow_compare.json"
    runs.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_run_artifact",
                "run_rows": {"base": [representative_workflow_row()]},
                "cases": [{"case_id": "q-workflow"}],
            }
        ),
        encoding="utf-8",
    )

    summary = run_module(
        "workflow",
        "compare-runs",
        "--runs",
        str(runs),
        "--ranking-keys",
        "hybrid_rrf_constraints,hybrid_rrf_constraints_signature,adaptive_signature",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["run_count"] == 3
    assert summary["case_count"] == 1
    assert report["method"] == "retrieval_lab_workflow_comparison"
    assert set(report["workflow_summaries"]) == {
        "hybrid_rrf_constraints",
        "hybrid_rrf_constraints_signature",
        "adaptive_signature",
    }


def test_retreieval_lab_evaluate_run_rows_matches_legacy_graded_metrics():
    run_rows = representative_run_rows()
    qrels = [
        {"query_id": "q1", "item_id": "i-target", "grade": 3},
        {"query_id": "q1", "item_id": "i-partial", "grade": 1},
        {"query_id": "q1", "item_id": "i-style-risk", "grade": 0},
    ]
    from mocktesting.mock_retriever import graded_metrics as legacy_graded_metrics

    modern = evaluate_run_rows(run_rows, qrels=qrels, top_k=10)

    assert modern["rule::hybrid"] == legacy_graded_metrics(run_rows["rule::hybrid"], qrels, top_k=10)
    assert modern["hyde::hybrid"] == legacy_graded_metrics(run_rows["hyde::hybrid"], qrels, top_k=10)
    assert run_metric_selection_score(modern["hyde::hybrid"]) > run_metric_selection_score(modern["rule::hybrid"])


def test_retreieval_lab_run_evaluate_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "run_eval.json"
    markdown = tmp_path / "run_eval.md"
    runs.write_text(
        json.dumps({"method": "retrieval_lab_run_artifact", "run_rows": representative_run_rows(), "cases": []}),
        encoding="utf-8",
    )
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "q1", "item_id": "i-target", "grade": 3}),
                json.dumps({"query_id": "q1", "item_id": "i-partial", "grade": 1}),
                json.dumps({"query_id": "q1", "item_id": "i-style-risk", "grade": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module(
        "run",
        "evaluate",
        "--runs",
        str(runs),
        "--qrels",
        str(qrels),
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["best_run"] == "hyde::hybrid"
    assert report["method"] == "retrieval_lab_run_evaluation"
    assert report["delta_vs_baseline"]["hyde::hybrid"]["nDCG@10"] > 0
    assert markdown.read_text(encoding="utf-8").startswith("# Run Evaluation Report")


def test_retreieval_lab_failure_classifier_matches_legacy_for_core_paths():
    from mocktesting.mock_retriever import classify_failure as legacy_classify_failure

    signal = SimpleNamespace(query_plan=SimpleNamespace(ambiguity={}, desired_stage=["setup"]))
    style_row = representative_failure_rows()["base"][0]
    recall_row = representative_failure_rows()["base"][2]
    fusion_row = representative_failure_rows()["base"][1]

    assert classify_failure_from_artifact(
        style_row,
        top1=style_row["top_results"][0],
        qrel_map={},
        top_k=3,
        candidate_depth=100,
        target_result=style_row["top_results"][1],
    ) == legacy_classify_failure(style_row, signal, {"semantic": 0.7}, style_row["top_results"][0])
    assert classify_failure_from_artifact(
        recall_row,
        top1=recall_row["top_results"][0],
        qrel_map={},
        top_k=3,
        candidate_depth=100,
        target_result=None,
    ) == legacy_classify_failure(recall_row, signal, {}, recall_row["top_results"][0])
    assert classify_failure_from_artifact(
        fusion_row,
        top1=fusion_row["top_results"][0],
        qrel_map={},
        top_k=3,
        candidate_depth=100,
        target_result=fusion_row["all_results"][-1],
    ) == legacy_classify_failure(fusion_row, signal, {"semantic": 0.7}, fusion_row["top_results"][0])


def test_retreieval_lab_failure_analysis_uses_qrels_for_ambiguous_multi_answer():
    run_rows = representative_failure_rows()
    qrels = [
        {"query_id": "q-ambiguous", "item_id": "i-alt", "grade": 2},
        {"query_id": "q-ambiguous", "item_id": "i-target", "grade": 3},
    ]

    failures = analyze_failure_rows(run_rows, qrels=qrels, top_k=3, candidate_depth=100)
    by_case = {row["case_id"]: row for row in failures}

    assert by_case["q-style"]["failure_type"] == "style_risk_miss"
    assert by_case["q-fusion"]["failure_type"] == "fusion_ranking_failure"
    assert by_case["q-recall"]["failure_type"] == "candidate_recall_failure"
    assert by_case["q-ambiguous"]["failure_type"] == "ambiguous_multi_valid_answer"
    assert by_case["q-ambiguous"]["suggested_next_action"] == "use graded qrels and avoid single-target-only scoring"


def test_retreieval_lab_failure_analysis_cli_is_native(tmp_path):
    runs = tmp_path / "runs.json"
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "failures.json"
    markdown = tmp_path / "failures.md"
    runs.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_run_artifact",
                "run_rows": representative_failure_rows(),
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "q-ambiguous", "item_id": "i-alt", "grade": 2}),
                json.dumps({"query_id": "q-ambiguous", "item_id": "i-target", "grade": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module(
        "run",
        "analyze-failures",
        "--runs",
        str(runs),
        "--qrels",
        str(qrels),
        "--top-k",
        "3",
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["failure_count"] == 4
    assert report["method"] == "retrieval_lab_failure_analysis_from_runs"
    assert report["summary"]["failure_type_counts"]["ambiguous_multi_valid_answer"] == 1
    assert markdown.read_text(encoding="utf-8").startswith("# Failure Analysis Report")


def test_retreieval_lab_extract_report_metrics_handles_nested_summaries():
    report = {
        "method": "mock_fuzzy_multirelevance_evaluation",
        "summary": {"scene_level_recall_at_10": 0.6, "style_violation_at_3": 0.1},
        "graded_metrics": {"nDCG@10": 0.72, "MRR@10": 0.8},
        "metrics": {"overall": {"recall_at_10": 0.9}},
    }
    metrics = extract_report_metrics(report)

    assert metrics["nDCG@10"] == 0.72
    assert metrics["MRR@10"] == 0.8
    assert metrics["scene_level_recall_at_10"] == 0.6
    assert metrics["recall_at_10"] == 0.9


def test_retreieval_lab_compare_experiments_cli_is_native_and_compatible(tmp_path):
    baseline = tmp_path / "baseline.json"
    improved = tmp_path / "improved.json"
    output = tmp_path / "comparison.json"
    markdown = tmp_path / "comparison.md"
    baseline.write_text(
        json.dumps(
            {
                "method": "baseline_eval",
                "summary": {"scene_level_recall_at_10": 0.6, "style_violation_at_3": 0.08},
                "graded_metrics": {"nDCG@10": 0.5, "MRR@10": 0.7},
            }
        ),
        encoding="utf-8",
    )
    improved.write_text(
        json.dumps(
            {
                "method": "improved_eval",
                "summary": {"scene_level_recall_at_10": 0.72, "style_violation_at_3": 0.02},
                "graded_metrics": {"nDCG@10": 0.68, "MRR@10": 0.8},
            }
        ),
        encoding="utf-8",
    )

    summary = run_module(
        "experiment",
        "compare",
        "--reports",
        str(baseline),
        str(improved),
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    from mocktesting.mock_retriever import compare_experiments_command as legacy_compare_experiments_command

    legacy = legacy_compare_experiments_command(SimpleNamespace(reports=[baseline, improved]))

    assert summary["report_count"] == legacy["summary"]["report_count"]
    assert summary["methods"] == legacy["summary"]["methods"]
    assert summary["best_method"] == "improved_eval"
    assert report["method"] == "retrieval_lab_experiment_comparison"
    assert report["reports"][1]["delta_vs_baseline"]["nDCG@10"] == 0.18
    assert markdown.read_text(encoding="utf-8").startswith("# Experiment Comparison Report")


def test_retreieval_lab_generate_eval_report_cli_summarizes_workflow_report_and_matches_legacy_section(tmp_path):
    source = tmp_path / "workflow.json"
    output = tmp_path / "eval.md"
    report = {
        "method": "mock_workflow_comparison",
        "summary": {"best_workflow": "hybrid_rrf_constraints"},
        "workflows": {
            "hybrid_rrf_constraints": {
                "summary": {
                    "target_recall_at_10": 0.7,
                    "stage_hit_at_3": 0.95,
                    "purpose_hit_at_3": 0.91,
                    "style_violation_at_3": 0.02,
                },
                "metrics": {"overall": {"recall_at_10": 0.8}},
            }
        },
    }
    source.write_text(json.dumps(report), encoding="utf-8")
    from mocktesting.mock_retriever import markdown_report as legacy_markdown_report

    summary = run_module("report", "eval", "--input", str(source), "--output", str(output))
    markdown = output.read_text(encoding="utf-8")
    legacy_markdown = legacy_markdown_report(report)

    assert summary["source_report_count"] == 1
    assert summary["methods"] == ["mock_workflow_comparison"]
    assert markdown.startswith("# Retrieval Lab Evaluation Report")
    assert "## Workflow Metrics" in markdown
    assert "## Workflow Metrics" in legacy_markdown


def test_retreieval_lab_generate_eval_report_cli_aggregates_multiple_reports(tmp_path):
    qrels = tmp_path / "qrels.json"
    failures = tmp_path / "failures.json"
    output = tmp_path / "eval.md"
    qrels.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_qrels_audit",
                "summary": {"qrels_trust_level": "low", "qrels_count": 12, "needs_adjudication_count": 3},
            }
        ),
        encoding="utf-8",
    )
    failures.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_failure_analysis_from_runs",
                "summary": {"failure_count": 2, "failure_rate": 0.25, "top_failure_type": "fusion_ranking_failure"},
                "failures": [
                    {
                        "case_id": "q1",
                        "failure_type": "fusion_ranking_failure",
                        "target_rank": 4,
                        "top1_item_id": "i1",
                        "suggested_next_action": "tune fusion",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = run_module("report", "eval", "--inputs", str(qrels), str(failures), "--output", str(output))
    markdown = output.read_text(encoding="utf-8")

    assert summary["source_report_count"] == 2
    assert "## Qrels Trust" in markdown
    assert "## Failure Analysis" in markdown
    assert "Prioritize active qrels sampling" in markdown


def test_retreieval_lab_qrels_audit_cli_is_native(tmp_path):
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    qrels.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q1",
                        "item_id": "i1",
                        "grade": 3,
                        "reason": "target item",
                        "source": "pooled_bootstrap",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module(
        "qrels",
        "audit",
        "--qrels",
        str(qrels),
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["qrels_count"] == 1
    assert report["method"] == "retrieval_lab_qrels_audit"
    assert markdown.read_text(encoding="utf-8").startswith("# Qrels Audit Report")


def test_retreieval_lab_merge_adjudicated_qrels_cli_is_native(tmp_path):
    qrels = tmp_path / "qrels.jsonl"
    adjudications = tmp_path / "adjudications.jsonl"
    output = tmp_path / "merged.jsonl"
    report_output = tmp_path / "merge_report.json"
    qrels.write_text(
        json.dumps({"query_id": "q1", "item_id": "i1", "grade": 1, "reason": "weak", "source": "pooled_bootstrap"})
        + "\n",
        encoding="utf-8",
    )
    adjudications.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "item_id": "i1",
                "grade": 3,
                "reason": "human says ideal",
                "confidence": 0.97,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module(
        "qrels",
        "merge-adjudicated",
        "--qrels",
        str(qrels),
        "--adjudications",
        str(adjudications),
        "--output",
        str(output),
        "--report-output",
        str(report_output),
    )
    merged_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_output.read_text(encoding="utf-8"))

    assert summary["adjudication_vote_count"] == 1
    assert merged_rows[0]["grade"] == 3
    assert merged_rows[0]["source"] == "manual_adjudicated"
    assert report["method"] == "retrieval_lab_merge_adjudicated_qrels"


def test_retreieval_lab_can_write_capability_report_from_tmp_reports(tmp_path):
    audit = tmp_path / "audit.json"
    fuzzy = tmp_path / "fuzzy.json"
    rerank = tmp_path / "rerank.json"
    registry = tmp_path / "cycles.jsonl"
    report = tmp_path / "capability.md"
    chart_dir = tmp_path / "charts"
    audit.write_text(
        json.dumps(
            {
                "method": "mock_qrels_audit",
                "summary": {
                    "qrels_trust_level": "low",
                    "qrels_count": 10,
                    "manual_or_llm_count": 0,
                    "manual_count": 0,
                    "llm_count": 0,
                    "bootstrap_only_count": 10,
                    "needs_adjudication_count": 2,
                    "vote_conflict_rate": 0.1,
                },
                "elapsed_seconds": 0.1,
            }
        ),
        encoding="utf-8",
    )
    fuzzy.write_text(
        json.dumps(
            {
                "method": "mock_fuzzy_multirelevance_evaluation",
                "summary": {
                    "nDCG@10": 0.62,
                    "MRR@10": 0.86,
                    "scene_level_recall_at_10": 0.59,
                    "stage_level_hit_at_3": 0.91,
                    "purpose_level_hit_at_3": 0.91,
                    "style_violation_at_3": 0.06,
                },
                "elapsed_seconds": 0.3,
            }
        ),
        encoding="utf-8",
    )
    rerank.write_text(
        json.dumps(
            {
                "method": "mock_rerank_upper_bound_comparison",
                "summary": {
                    "rerank_opportunity_nDCG@10": 0.3,
                    "oracle_rerank_nDCG@10": 0.74,
                    "baseline_nDCG@10": 0.44,
                },
                "elapsed_seconds": 0.2,
            }
        ),
        encoding="utf-8",
    )

    cycle = run_module(
        "cycle",
        "record",
        "--cycle-id",
        "test_origin",
        "--as-origin",
        "--reports",
        str(audit),
        str(fuzzy),
        str(rerank),
        "--registry",
        str(registry),
        "--output",
        str(tmp_path / "latest.json"),
    )
    generated = run_module(
        "report",
        "capability",
        "--registry",
        str(registry),
        "--output",
        str(report),
        "--chart-dir",
        str(chart_dir),
    )

    assert cycle["cycle_id"] == "test_origin"
    assert cycle["is_origin"] is True
    assert generated["cycle_count"] == 1
    assert report.exists()
    assert (chart_dir / "capability_bar_latest.svg").exists()


def test_retreieval_lab_artifact_manifest_records_fingerprints(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")

    manifest = artifact_manifest([artifact], manifest_id="m1", label="unit")

    assert manifest["manifest_id"] == "m1"
    assert manifest["artifact_count"] == 1
    assert manifest["missing_count"] == 0
    assert manifest["artifacts"][0]["sha256"]
    assert data_sha256({"a": 1, "b": 2}) == data_sha256({"b": 2, "a": 1})


def test_retreieval_lab_artifact_manifest_cli_is_native(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"
    input_path.write_text(json.dumps({"run_rows": representative_run_rows()}), encoding="utf-8")
    output_path.write_text(json.dumps({"query_id": "q1", "item_id": "i1", "grade": 3}) + "\n", encoding="utf-8")

    summary = run_module(
        "artifact",
        "manifest",
        "--manifest-id",
        "cycle_smoke",
        "--inputs",
        str(input_path),
        "--outputs",
        str(output_path),
        "--output",
        str(manifest_path),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["manifest_id"] == "cycle_smoke"
    assert summary["artifact_count"] == 2
    assert manifest["method"] == "retrieval_lab_artifact_manifest"
    assert {row["role"] for row in manifest["artifacts"]} == {"input", "output"}


def test_retreieval_lab_schema_catalog_names_core_contracts():
    rows = schema_catalog()
    names = {row["name"] for row in rows}

    assert {
        "query_plan",
        "scene_signature",
        "run_row",
        "qrel",
        "index_manifest",
        "retrieval_run_config",
        "llm_judgement",
        "capability_cycle",
    } <= names


def test_retreieval_lab_query_plan_schema_blocks_negative_leak():
    valid = validate_record(
        "query_plan",
        {
            "planner": "rule",
            "original_text": "need a grounded setup without product pitch",
            "positive_query": "grounded setup",
            "desired_stage": ["setup"],
            "negative_style": ["product_pitch"],
            "scene_signature": {"raw_positive_query": "grounded setup"},
            "soft_constraints": [{"kind": "style", "polarity": "should_not", "values": ["product_pitch"]}],
        },
    )
    invalid = validate_record(
        "query_plan",
        {
            "planner": "rule",
            "original_text": "need a grounded setup without product pitch",
            "positive_query": "grounded setup product_pitch",
            "negative_style": ["product_pitch"],
        },
    )

    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert "negative term leaked" in invalid["errors"][0]["message"]


def test_retreieval_lab_schema_validate_cli_handles_jsonl(tmp_path):
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "schema_report.json"
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "q1", "item_id": "i1", "grade": 3, "source": "bootstrap"}),
                json.dumps({"query_id": "q1", "item_id": "i2", "grade": 0, "source": "bootstrap"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module("schema", "validate", "qrel", "--input", str(qrels), "--output", str(output))
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["schema_name"] == "qrel"
    assert summary["valid_count"] == 2
    assert report["validation"]["invalid_count"] == 0


def test_retreieval_lab_core_schema_facades_are_native():
    from retreieval_lab.indexes import validate_index_manifest
    from retreieval_lab.llm import validate_llm_judgement
    from retreieval_lab.planners import validate_query_plan
    from retreieval_lab.retrieval import validate_retrieval_run_config, validate_run_row

    assert validate_query_plan({"original_text": "setup", "positive_query": "setup"})["valid"] is True
    assert validate_index_manifest({"index_id": "idx", "item_count": 1})["valid"] is True
    assert validate_retrieval_run_config({"workflow": "hybrid_rrf", "top_k": 10})["valid"] is True
    assert validate_run_row({"case_id": "q1", "top_results": [{"item_id": "i1", "score": 1.0}]})["valid"] is True
    assert validate_llm_judgement({"query_id": "q1", "candidate_item_id": "i1", "grade": 2})["valid"] is True


def test_retreieval_lab_migration_audit_reports_mocktesting_clean(tmp_path):
    output = tmp_path / "migration.json"
    markdown = tmp_path / "migration.md"

    summary = run_module(
        "migration",
        "audit",
        "--round-id",
        "round_0",
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["mocktesting_clean"] is True
    assert summary["compat_only_command_count"] > 0
    assert report["self_check_contract"]["mocktesting_must_remain_clean"] is True
    assert markdown.read_text(encoding="utf-8").startswith("# Retrieval Lab Migration Audit")


def test_retreieval_lab_planner_plan_sanitizes_negative_text_and_uses_cache(tmp_path):
    cache = tmp_path / "planner_cache.jsonl"
    query = "need grounded setup without product pitch"

    first = plan_many([query], planner="multi_query", cache_path=cache)
    second = plan_many([query], planner="multi_query", cache_path=cache)
    plan = first["plans"][0]

    assert first["summary"]["cache_misses"] == 1
    assert second["summary"]["cache_hits"] == 1
    assert plan["planner"] == "multi_query"
    assert plan["desired_stage"] == ["setup"]
    assert "product_pitch" in plan["negative_style"]
    assert "product pitch" not in plan["positive_query"].lower()
    assert first["summary"]["negative_leak_rate"] == 0


def test_retreieval_lab_planner_compare_can_include_legacy_adapter(tmp_path):
    cache = tmp_path / "planner_cache.jsonl"
    report = compare_planners(
        ["need grounded setup without product pitch"],
        planners=["rule", "multi_query", "legacy_adapter"],
        cache_path=cache,
    )

    assert report["summary"]["planner_count"] == 3
    assert report["planner_metrics"]["rule"]["negative_leak_rate"] == 0
    assert "legacy_adapter" in report["planner_metrics"]
    assert report["comparisons"][0]["planner_diffs"]["multi_query"]["changed_field_count"] >= 0


def test_retreieval_lab_planner_cli_writes_plan_and_cache_audit(tmp_path):
    output = tmp_path / "planner_plan.json"
    cache = tmp_path / "planner_cache.jsonl"
    audit = tmp_path / "planner_cache_audit.json"

    summary = run_module(
        "planner",
        "plan",
        "--planner",
        "hyde_card",
        "--query",
        "need setup without advertising",
        "--planner-cache",
        str(cache),
        "--output",
        str(output),
    )
    audit_summary = run_module("planner", "audit-cache", "--planner-cache", str(cache), "--output", str(audit))
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["planner"] == "hyde_card"
    assert summary["input_count"] == 1
    assert report["plans"][0]["hyde_text"]
    assert audit_summary["row_count"] == 1


def test_retreieval_lab_planner_cache_key_includes_config_fingerprint():
    a = planner_cache_key("rule", "query", {"x": 1})
    b = planner_cache_key("rule", "query", {"x": 2})

    assert a != b


def small_retrieval_dataset(path):
    data = {
        "dataset_id": "small_retrieval",
        "cases": [
            {
                "case_id": "case_setup",
                "case_type": "positive",
                "user_input": "need grounded setup without product pitch",
                "expected_relation": "should_match",
                "target": {
                    "fixture_id": "f1",
                    "scene_id": "scene_setup",
                    "retrieval_id": "ret_setup",
                    "script_stage": "setup",
                    "creative_purpose": ["build_reality"],
                    "title": "Grounded setup",
                    "industry": "healthcare",
                    "style": "documentary",
                },
                "target_summary": "Grounded hospital setup with real location pressure.",
                "target_tags_text": "hospital doctor setup real location",
                "target_embedding_texts": {"script_usage": "setup build_reality hospital doctor"},
            },
            {
                "case_id": "case_opening",
                "case_type": "negative",
                "user_input": "need grounded setup without opening",
                "expected_relation": "should_not_match",
                "target": {
                    "fixture_id": "f1",
                    "scene_id": "scene_opening",
                    "retrieval_id": "ret_opening",
                    "script_stage": "opening",
                    "creative_purpose": ["establish_problem"],
                    "title": "Opening distance",
                    "industry": "healthcare",
                    "style": "brand_film",
                },
                "expected_prefer": {
                    "fixture_id": "f1",
                    "scene_id": "scene_setup",
                    "retrieval_id": "ret_setup",
                    "script_stage": "setup",
                    "creative_purpose": ["build_reality"],
                    "title": "Grounded setup",
                    "industry": "healthcare",
                    "style": "documentary",
                },
                "target_summary": "Opening with distance and problem establishment.",
                "target_tags_text": "opening mountain hospital",
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_retreieval_lab_index_manifest_builds_dataset_fingerprint(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    manifest = build_index_manifest(dataset_path=dataset, split="all", limit=0, index_id="idx")

    assert manifest["index_id"] == "idx"
    assert manifest["item_count"] == 2
    assert "lexical" in manifest["channels"]
    assert manifest["fingerprint"]


def test_retreieval_lab_native_retrieval_run_returns_expected_target(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    artifact = retrieval_run(dataset_path=dataset, split="all", limit=0, planner="multi_query", planner_cache=None, top_k=2)
    rows = next(iter(artifact["run_rows"].values()))

    assert artifact["summary"]["row_count"] == 2
    assert rows[0]["target_rank"] == 1
    assert rows[1]["target_rank"] == 1
    assert rows[1]["top_results"][0]["metadata"]["script_stage"] == "setup"


def test_retreieval_lab_retrieval_cli_writes_run_artifact(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "retrieval_run.json"
    index_output = tmp_path / "index_manifest.json"

    index_summary = run_module("index", "manifest", "--dataset", str(dataset), "--split", "all", "--output", str(index_output))
    summary = run_module(
        "retrieval",
        "run",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--planner",
        "multi_query",
        "--planner-cache",
        str(tmp_path / "planner_cache.jsonl"),
        "--output",
        str(output),
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert index_summary["item_count"] == 2
    assert summary["row_count"] == 2
    assert artifact["method"] == "retrieval_lab_native_retrieval_run"
    assert artifact["run_rows"]


def test_retreieval_lab_infra_audit_reports_native_and_compat_gaps(tmp_path):
    output = tmp_path / "infra.json"
    markdown = tmp_path / "infra.md"

    summary = run_module("infra", "audit", "--output", str(output), "--markdown-output", str(markdown))
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["legacy_command_count"] > 0
    assert summary["compat_only_legacy_command_count"] > 0
    assert summary["empty_layer_count"] == 0
    assert any(row["command"] == "compare-experiments" and row["status"] in {"native", "partial"} for row in report["command_coverage"])
    assert any(row["command"] == "schema-catalog" and row["status"] == "native_only" for row in report["command_coverage"])
    assert any(row["layer"] == "planners" and row["status"] == "implemented" for row in report["layer_coverage"])
    assert any("compat_only_commands" in gap for gap in summary["top_gaps"])
    assert markdown.read_text(encoding="utf-8").startswith("# Infra Coverage Audit")
