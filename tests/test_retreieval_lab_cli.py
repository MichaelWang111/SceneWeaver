from __future__ import annotations

from decimal import Decimal
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from retrieval_lab.architecture import layer_manifest
from retrieval_lab.artifacts import artifact_manifest, data_sha256
from retrieval_lab.config import project_paths
from retrieval_lab.compat import translate_argv
from retrieval_lab.evaluators import (
    analyze_failure_rows,
    classify_failure_from_artifact,
    evaluate_run_rows,
    graded_metrics,
    recall_bound_rows,
    recall_bound_summary,
    run_metric_selection_score,
)
from retrieval_lab.experiments import extract_run_rows_from_report
from retrieval_lab.experiments.compare import extract_report_metrics
from retrieval_lab.experiments.core import llm_rerank_prompt, metadata_label_leaks, natural_fuzzy_variant_cases
from retrieval_lab.capability import extract_capability_raw_metrics
from retrieval_lab.graph import scene_graph_from_cases
from retrieval_lab.qrels.calibration import judge_calibration_summary
from retrieval_lab.experiments.legacy import with_output_option
from retrieval_lab.planners import compare_planners, plan_many, planner_cache_key
from retrieval_lab.planners.service import llm_negative_style_values
from retrieval_lab.qrels import (
    active_qrels_samples,
    merge_adjudicated_qrels,
    pooled_qrels_from_run_rows,
    pooled_qrels_summary,
    qrel_confidence,
    qrels_audit_summary,
    qrels_trust_level,
)
from retrieval_lab.ranking import (
    WORKFLOW_RANKING_KEYS,
    rerank_row_by_qrels,
    rerank_row_by_rule,
    rerank_run_rows,
    rerank_run_rows_by_workflow,
    workflow_score,
)
from retrieval_lab.indexes import build_index_manifest, index_items_from_cases
from retrieval_lab.retrieval import retrieval_run, score_item
from retrieval_lab.retrieval.benchmark import retrieval_benchmark_command
from retrieval_lab.retrieval.service import prepare_retrieval_index, rows_with_rrf_scores, score_items
from retrieval_lab.schemas import schema_catalog, validate_record, validate_records
from retrieval_lab.llm.adjudication import (
    adaptive_judge_batch_sizes,
    llm_adjudicate_qrels_command,
    natural_fuzzy_prompt,
    natural_fuzzy_response_rows,
    split_batch_by_token_budget,
)
from retrieval_lab.llm.budget_guard import (
    HardBudgetError,
    ModelPricing,
    ProviderBudgetGuard,
    ProviderLimits,
    ProviderProfile,
    estimate_request_cost_cny,
)
import retrieval_lab.llm.adjudication as llm_adjudication


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
    assert translate_argv(["rerank", "export-features", "--runs", "r.json"]) == [
        "rerank-export-features",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["rerank", "calibrate", "--features", "f.jsonl"]) == [
        "rerank-calibrate",
        "--features",
        "f.jsonl",
    ]
    assert translate_argv(["rerank", "apply-calibrated", "--runs", "r.json", "--model", "m.json"]) == [
        "rerank-apply-calibrated",
        "--runs",
        "r.json",
        "--model",
        "m.json",
    ]
    assert translate_argv(["rerank", "attribute", "--runs", "r.json"]) == [
        "rerank-attribute",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["qrels", "judge-calibration", "--qrels", "q.jsonl"]) == [
        "qrels-judge-calibration",
        "--qrels",
        "q.jsonl",
    ]
    assert translate_argv(["qrels", "sample-coverage-from-runs", "--runs", "r.json"]) == [
        "sample-coverage-qrels-from-runs",
        "--runs",
        "r.json",
    ]
    assert translate_argv(["graph", "build-manifest"]) == ["graph-build-manifest"]
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
    assert translate_argv(["benchmark", "retrieval"]) == ["benchmark-retrieval"]
    assert translate_argv(["llm", "adjudicate-qrels"]) == ["llm-adjudicate-qrels"]
    assert translate_argv(["eval", "fuzzy", "--limit", "6"]) == ["evaluate-fuzzy-multirelevance", "--limit", "6"]
    assert translate_argv(["eval", "anti-overfit-fuzzy"]) == ["evaluate-anti-overfit-fuzzy"]
    assert translate_argv(["cycle", "record", "--cycle-id", "c1"]) == ["record-capability-cycle", "--cycle-id", "c1"]
    assert translate_argv(["llm", "generate-natural-fuzzy"]) == ["llm-generate-natural-fuzzy"]
    assert translate_argv(["llm", "status"]) == ["llm-status"]
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
    assert paths["package_name"] == "retrieval_lab"
    assert paths["legacy_baseline_package"] == "mocktesting"


def run_module(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "retrieval_lab", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_retreieval_lab_flywheel_guide_uses_modern_commands(tmp_path):
    output = tmp_path / "guide.json"
    result = subprocess.run(
        [sys.executable, "-m", "retrieval_lab", "flywheel", "guide", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    guide = json.loads(output.read_text(encoding="utf-8"))

    assert summary["step_count"] == len(guide["steps"])
    assert all("python -m retrieval_lab" in row["command"] for row in guide["steps"])
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


def test_retreieval_lab_adjudication_votes_override_bootstrap_conflicts():
    existing = [{
        "query_id": "q1",
        "item_id": "candidate",
        "grade": 1,
        "reason": "bootstrap partial",
        "source": "pooled_bootstrap",
        "grade_votes": [{"grade": 1, "judge_type": "bootstrap", "reason": "partial"}],
    }]
    votes = [{
        "query_id": "q1",
        "item_id": "candidate",
        "grade": 3,
        "reason": "human says this is ideal",
        "judge_type": "human",
        "judge_id": "reviewer",
        "judge_version": "v1",
        "confidence": 0.97,
    }]

    row = merge_adjudicated_qrels(existing, votes)[0]
    summary = qrels_audit_summary([row])

    assert row["grade"] == 3
    assert row["source"] == "manual_adjudicated"
    assert row["needs_adjudication"] is False
    assert qrel_confidence(row) == 0.97
    assert qrels_trust_level([row]) == "high"
    assert summary["conflict_count"] == 0
    assert summary["needs_adjudication_count"] == 0


def test_retreieval_lab_adjudication_keeps_same_tier_conflicts():
    existing = [{
        "query_id": "q1",
        "item_id": "candidate",
        "grade": 1,
        "reason": "bootstrap partial",
        "source": "pooled_bootstrap",
        "grade_votes": [{"grade": 1, "judge_type": "bootstrap", "reason": "partial"}],
    }]
    votes = [
        {
            "query_id": "q1",
            "item_id": "candidate",
            "grade": 2,
            "reason": "llm says usable",
            "judge_type": "llm",
            "judge_id": "judge_a",
            "judge_version": "v1",
            "confidence": 0.72,
        },
        {
            "query_id": "q1",
            "item_id": "candidate",
            "grade": 0,
            "reason": "llm says irrelevant",
            "judge_type": "llm",
            "judge_id": "judge_b",
            "judge_version": "v1",
            "confidence": 0.81,
        },
    ]

    row = merge_adjudicated_qrels(existing, votes)[0]
    summary = qrels_audit_summary([row])

    assert row["source"] == "llm_adjudicated"
    assert row["needs_adjudication"] is True
    assert summary["conflict_count"] == 1


def test_retreieval_lab_llm_qrels_fake_adjudication_merges_votes(tmp_path):
    samples = tmp_path / "active.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "adjudications.jsonl"
    report_output = tmp_path / "adjudication_report.json"
    merged = tmp_path / "merged.jsonl"
    samples.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "item_id": "i1",
                "user_input": "need setup",
                "target_item_id": "i1",
                "metadata": {"script_stage": "setup"},
                "constraint_hits": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    qrels.write_text(
        json.dumps({"query_id": "q1", "item_id": "i1", "grade": 3, "source": "pooled_bootstrap"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        samples=samples,
        output=output,
        report_output=report_output,
        qrels=qrels,
        merged_qrels_output=merged,
        judge_cache=tmp_path / "judge_cache.jsonl",
        llm_sample_size=1,
        batch_size=1,
        budget_cny=20.0,
        cny_per_1k_tokens=0.01,
        require_llm=False,
        fake_client=True,
        max_tokens=200,
        timeout_seconds=5.0,
        retries=0,
    )

    report = llm_adjudicate_qrels_command(args)
    merged_rows = [json.loads(line) for line in merged.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert report["summary"]["llm_count"] == 1
    assert report["summary"]["total_estimated_cost_cny"] <= 20
    assert report["summary"]["batching_strategy"] == "manual"
    assert merged_rows[0]["source"] == "llm_adjudicated"
    assert any(vote["judge_type"] == "llm" for vote in merged_rows[0]["grade_votes"])


def test_retreieval_lab_llm_qrels_adjudication_runs_batches_concurrently(tmp_path, monkeypatch):
    samples = tmp_path / "active.jsonl"
    rows = [
        {
            "query_id": f"q{index}",
            "item_id": f"i{index}",
            "user_input": "need setup",
            "target_item_id": f"i{index}",
            "metadata": {"script_stage": "setup"},
            "constraint_hits": {},
        }
        for index in range(6)
    ]
    samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_real(samples_batch, *, prompt, max_tokens, timeout_seconds, retries):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return [{"grade": 3, "reason": "ok", "confidence": 0.9} for _sample in samples_batch]

    monkeypatch.setattr(llm_adjudication, "real_llm_judgements", slow_real)
    args = SimpleNamespace(
        samples=samples,
        output=tmp_path / "adjudications.jsonl",
        report_output=tmp_path / "report.json",
        qrels=None,
        merged_qrels_output=None,
        judge_cache=tmp_path / "judge_cache.jsonl",
        llm_sample_size=6,
        batch_size=1,
        concurrency=3,
        budget_cny=20.0,
        cny_per_1k_tokens=0.01,
        require_llm=True,
        fake_client=False,
        max_tokens=200,
        timeout_seconds=5.0,
        retries=0,
    )

    report = llm_adjudicate_qrels_command(args)

    assert report["summary"]["llm_call_count"] == 6
    assert report["summary"]["successful_batch_count"] == 6
    assert report["summary"]["failed_batch_count"] == 0
    assert report["summary"]["batching_strategy"] == "manual"
    assert report["summary"]["concurrency"] == 3
    assert max_active > 1
    output_rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["query_id"] for row in output_rows] == [f"q{index}" for index in range(6)]


def test_retreieval_lab_llm_qrels_adjudication_is_resume_safe_after_batch_failure(tmp_path, monkeypatch):
    samples = tmp_path / "active.jsonl"
    rows = [
        {
            "query_id": f"q{index}",
            "item_id": f"i{index}",
            "user_input": "need setup",
            "target_item_id": f"i{index}",
            "metadata": {"script_stage": "setup"},
            "constraint_hits": {},
        }
        for index in range(4)
    ]
    samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    seen_batches: list[list[str]] = []

    def flaky_real(samples_batch, *, prompt, max_tokens, timeout_seconds, retries):
        ids = [str(sample["query_id"]) for sample in samples_batch]
        seen_batches.append(ids)
        if "q2" in ids:
            raise TimeoutError("simulated timeout")
        return [{"grade": 3, "reason": "ok", "confidence": 0.9} for _sample in samples_batch]

    monkeypatch.setattr(llm_adjudication, "real_llm_judgements", flaky_real)
    args = SimpleNamespace(
        samples=samples,
        output=tmp_path / "adjudications.jsonl",
        report_output=tmp_path / "report.json",
        qrels=None,
        merged_qrels_output=None,
        judge_cache=tmp_path / "judge_cache.jsonl",
        llm_sample_size=4,
        batch_size=2,
        concurrency=2,
        budget_cny=20.0,
        cny_per_1k_tokens=0.01,
        require_llm=True,
        fake_client=False,
        max_tokens=200,
        timeout_seconds=5.0,
        retries=0,
    )

    first = llm_adjudicate_qrels_command(args)

    assert first["summary"]["status"] == "partial_failed"
    assert first["summary"]["judgement_count"] == 2
    assert first["summary"]["failed_batch_count"] == 1
    assert [json.loads(line)["judgement"]["query_id"] for line in args.judge_cache.read_text(encoding="utf-8").splitlines()] == ["q0", "q1"]

    seen_batches.clear()

    def repaired_real(samples_batch, *, prompt, max_tokens, timeout_seconds, retries):
        ids = [str(sample["query_id"]) for sample in samples_batch]
        seen_batches.append(ids)
        return [{"grade": 2, "reason": "repaired", "confidence": 0.8} for _sample in samples_batch]

    monkeypatch.setattr(llm_adjudication, "real_llm_judgements", repaired_real)
    second = llm_adjudicate_qrels_command(args)

    assert second["summary"]["status"] == "ok"
    assert second["summary"]["cache_hit_count"] == 2
    assert second["summary"]["judgement_count"] == 4
    assert seen_batches == [["q2", "q3"]]


class FakeBudgetClient:
    provider = "deepseek"
    model = "deepseek-v4-flash"

    def __init__(self, balances: list[str] | None = None) -> None:
        self.balances = [Decimal(value) for value in (balances or ["10.00", "9.99"])]

    def profile(self) -> ProviderProfile:
        return ProviderProfile(
            provider=self.provider,
            model=self.model,
            pricing=ModelPricing(input_cny_per_million=Decimal("1"), output_cny_per_million=Decimal("2")),
            limits=ProviderLimits(concurrency=2500),
            balance_available=self.balance_cny(),
        )

    def balance_cny(self) -> Decimal:
        if len(self.balances) > 1:
            return self.balances.pop(0)
        return self.balances[0]


def test_hard_budget_guard_reserves_max_output_and_writes_ledger(tmp_path):
    guard = ProviderBudgetGuard(
        client=FakeBudgetClient(["10.00", "9.999"]),
        hard_budget_cny=1.0,
        safety_cny=0.0,
        ledger_path=tmp_path / "usage.jsonl",
        balance_check_interval_seconds=999,
    )
    guard.preflight()

    reservation = guard.reserve(batch_id=1, sample_count=2, prompt_tokens_upper_bound=1000, max_completion_tokens=500)
    guard.settle_success(reservation, usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, request_id="req-1")
    rows = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()]

    assert reservation.reserved_cny == estimate_request_cost_cny(
        ModelPricing(input_cny_per_million=Decimal("1"), output_cny_per_million=Decimal("2")),
        prompt_tokens=1000,
        max_completion_tokens=500,
    )
    assert rows[0]["status"] == "ok"
    assert rows[0]["request_id"] == "req-1"
    assert rows[0]["usage"]["total_tokens"] == 150
    assert guard.summary()["observed_balance_delta_cny"] == 0.001


def test_hard_budget_guard_marks_missing_usage_as_charged_failure(tmp_path):
    guard = ProviderBudgetGuard(
        client=FakeBudgetClient(["10.00", "10.00"]),
        hard_budget_cny=1.0,
        safety_cny=0.0,
        ledger_path=tmp_path / "usage.jsonl",
    )
    guard.preflight()
    reservation = guard.reserve(batch_id=1, sample_count=1, prompt_tokens_upper_bound=1000, max_completion_tokens=500)

    guard.settle_success(reservation, usage={}, request_id=None)
    row = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))

    assert row["status"] == "usage_missing"
    assert row["actual_cost_cny"] == row["reserved_cny"]
    assert guard.summary()["budget_stop_reason"] == "provider_usage_missing"


def test_dashscope_hard_budget_fails_closed_without_bss_credentials(monkeypatch):
    from retrieval_lab.llm.budget_guard import DashScopeBudgetClient

    monkeypatch.delenv("ALIBABA_CLOUD_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", raising=False)

    try:
        DashScopeBudgetClient(model="qwen3.6-plus").profile()
    except HardBudgetError as exc:
        assert "fail-closed" in str(exc)
    else:
        raise AssertionError("DashScope hard budget must fail closed without BSS credentials")


def test_retreieval_lab_llm_status_cli_reports_static_provider_state(monkeypatch):
    monkeypatch.delenv("SCENEWEAVER_API_KEY", raising=False)
    monkeypatch.delenv("SCENEWEAVER_BASE_URL", raising=False)
    monkeypatch.delenv("SCENEWEAVER_MODEL", raising=False)
    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    result = run_module("llm", "status", "--provider", "deepseek")

    assert result["summary"]["provider"] == "deepseek"
    assert result["summary"]["api_key_configured"] is True
    assert result["summary"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert result["summary"]["model_count"] >= 2
    assert result["summary"]["limits"]["concurrency"] == 2500
    assert "deepseek-secret" not in repr(result)


def test_retreieval_lab_llm_qrels_hard_budget_records_usage_ledger(tmp_path, monkeypatch):
    samples = tmp_path / "active.jsonl"
    samples.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "item_id": "i1",
                "user_input": "need setup",
                "target_item_id": "i1",
                "metadata": {"script_stage": "setup"},
                "constraint_hits": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.jsonl"

    monkeypatch.setattr(
        llm_adjudication,
        "build_budget_guard",
        lambda args: ProviderBudgetGuard(
            client=FakeBudgetClient(["10.00", "9.99"]),
            hard_budget_cny=1.0,
            safety_cny=0.0,
            ledger_path=ledger,
        ),
    )
    monkeypatch.setattr(
        llm_adjudication,
        "real_llm_judgement_response",
        lambda samples_batch, *, prompt, max_tokens, timeout_seconds, retries: llm_adjudication.LLMJudgementResponse(
            judgements=[{"grade": 3, "reason": "ok", "confidence": 0.9} for _sample in samples_batch],
            usage={"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            request_id="req-hard-budget",
        ),
    )
    args = SimpleNamespace(
        samples=samples,
        output=tmp_path / "adjudications.jsonl",
        report_output=tmp_path / "report.json",
        qrels=None,
        merged_qrels_output=None,
        judge_cache=tmp_path / "judge_cache.jsonl",
        llm_sample_size=1,
        batch_size=1,
        concurrency=1,
        budget_cny=20.0,
        cny_per_1k_tokens=0.01,
        require_llm=True,
        fake_client=False,
        max_tokens=200,
        timeout_seconds=5.0,
        retries=0,
        hard_budget_cny=1.0,
        budget_safety_cny=0.0,
        provider="deepseek",
        usage_ledger=ledger,
        balance_check_interval_seconds=999.0,
    )

    report = llm_adjudicate_qrels_command(args)
    ledger_rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]

    assert report["summary"]["hard_budget_enabled"] is True
    assert report["summary"]["provider"] == "deepseek"
    assert report["summary"]["provider_concurrency_limit"] == 2500
    assert ledger_rows[0]["request_id"] == "req-hard-budget"
    assert ledger_rows[0]["status"] == "ok"
    assert ledger_rows[0]["usage"]["total_tokens"] == 140


def test_retreieval_lab_llm_qrels_adaptive_batch_sizes_balance_sample_shape():
    kwargs = {"concurrency": 10, "max_batch_size": 10, "min_batch_size_for_concurrency": 5}

    assert adaptive_judge_batch_sizes(10, **kwargs) == [10]
    assert adaptive_judge_batch_sizes(50, **kwargs) == [5] * 10
    assert adaptive_judge_batch_sizes(89, **kwargs) == [9] * 9 + [8]
    assert adaptive_judge_batch_sizes(100, **kwargs) == [10] * 10
    assert adaptive_judge_batch_sizes(120, **kwargs) == [10] * 12
    assert adaptive_judge_batch_sizes(153, **kwargs) == [10] * 9 + [9] * 7
    assert adaptive_judge_batch_sizes(239, **kwargs) == [10] * 23 + [9]
    for sample_count in [1, 7, 11, 37, 89, 153, 239, 1000]:
        sizes = adaptive_judge_batch_sizes(sample_count, **kwargs)
        assert sum(sizes) == sample_count
        assert max(sizes) <= kwargs["max_batch_size"]
        assert max(sizes) - min(sizes) <= 1


def test_retreieval_lab_llm_qrels_auto_batching_avoids_single_sample_requests(tmp_path):
    samples = tmp_path / "active.jsonl"
    rows = [
        {
            "query_id": f"q{index}",
            "item_id": f"i{index}",
            "user_input": "need setup",
            "target_item_id": f"i{index}",
            "metadata": {"script_stage": "setup"},
            "constraint_hits": {},
        }
        for index in range(50)
    ]
    samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        samples=samples,
        output=tmp_path / "adjudications.jsonl",
        report_output=tmp_path / "report.json",
        qrels=None,
        merged_qrels_output=None,
        judge_cache=tmp_path / "judge_cache.jsonl",
        llm_sample_size=50,
        batch_size=0,
        max_batch_size=10,
        max_batch_tokens=6000,
        min_batch_size_for_concurrency=5,
        concurrency=10,
        budget_cny=20.0,
        cny_per_1k_tokens=0.01,
        require_llm=False,
        fake_client=True,
        max_tokens=200,
        timeout_seconds=5.0,
        retries=0,
    )

    report = llm_adjudicate_qrels_command(args)

    assert report["summary"]["batching_strategy"] == "auto_balanced"
    assert report["summary"]["batch_size"] == 5
    assert report["summary"]["submitted_batch_count"] == 10
    assert report["summary"]["judgement_count"] == 50


def test_retreieval_lab_llm_qrels_token_budget_splits_long_batches():
    rows = [
        (
            index,
            {
                "query_id": f"q{index}",
                "item_id": f"i{index}",
                "user_input": "need setup",
                "target_item_id": f"i{index}",
                "metadata": {"script_stage": "setup", "long_text": "x" * 2400},
                "constraint_hits": {},
            },
        )
        for index in range(4)
    ]

    batches = split_batch_by_token_budget(rows, max_batch_tokens=1500)

    assert len(batches) == 4
    assert [len(batch) for batch in batches] == [1, 1, 1, 1]


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


def representative_diagnostic_run_rows() -> dict[str, list[dict]]:
    row = representative_workflow_row()
    row = {**row, "all_results": list(row["top_results"])}
    return {"base": [row]}


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
    assert "style_safe_signature" in WORKFLOW_RANKING_KEYS


def test_retreieval_lab_style_safe_workflow_demotes_negative_style_candidate():
    row = {
        "case_id": "q-style-safe",
        "target_item_id": "i-safe",
        "target_stage": "setup",
        "target_purposes": ["build_trust"],
        "query_constraints": {"negative_style": ["ad_like"]},
        "query_plan": {"negative_style": ["ad_like"], "ambiguity": {"level": "high"}},
        "top_results": [
            {
                "item_id": "i-risk",
                "score": 10.0,
                "embedding_score": 10.0,
                "lexical_score": 1.0,
                "rrf_score": 10.0,
                "constraint_score": 0.0,
                "signature_score": 1.0,
                "style_score": -1.5,
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"], "style_risks": ["ad_like"]},
                "constraint_hits": {"negative_style": ["ad_like"]},
            },
            {
                "item_id": "i-safe",
                "score": 1.0,
                "embedding_score": 1.0,
                "lexical_score": 0.8,
                "rrf_score": 1.0,
                "constraint_score": 0.2,
                "signature_score": 0.6,
                "style_score": 0.0,
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_trust"], "style_risks": []},
                "constraint_hits": {"desired_stage": ["setup"]},
            },
        ],
    }

    reranked = rerank_run_rows_by_workflow({"base": [row]}, ranking_key="style_safe_signature", top_k=2)
    top = next(iter(reranked.values()))[0]["top_results"][0]
    risky = next(iter(reranked.values()))[0]["top_results"][1]

    assert top["item_id"] == "i-safe"
    assert risky["style_guardrail_action"] == "strong_style_penalty"
    assert risky["workflow_score_components"]["style_risk_score"] == 1


def test_retreieval_lab_feature_rerank_signature_is_available_and_style_safe():
    row = {
        "case_id": "q-feature-rerank",
        "target_item_id": "i-safe",
        "target_stage": "setup",
        "target_purposes": ["build_reality"],
        "query_constraints": {"negative_style": ["ad_like"]},
        "top_results": [
            {
                "item_id": "i-risk",
                "score": 5.0,
                "rrf_score": 2.0,
                "constraint_score": 1.0,
                "signature_score": 1.0,
                "purpose_score": 1.0,
                "style_score": -1.0,
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"], "style_risks": ["ad_like"]},
                "constraint_hits": {"desired_stage": ["setup"], "negative_style": ["ad_like"]},
            },
            {
                "item_id": "i-safe",
                "score": 1.0,
                "rrf_score": 1.6,
                "constraint_score": 1.0,
                "signature_score": 0.8,
                "purpose_score": 1.0,
                "style_score": 0.0,
                "metadata": {"script_stage": "setup", "creative_purpose": ["build_reality"], "style_risks": []},
                "constraint_hits": {"desired_stage": ["setup"]},
            },
        ],
    }

    reranked = rerank_run_rows_by_workflow({"base": [row]}, ranking_key="feature_rerank_signature", top_k=2)
    output_row = next(iter(reranked.values()))[0]

    assert "feature_rerank_signature" in WORKFLOW_RANKING_KEYS
    assert output_row["top_results"][0]["item_id"] == "i-safe"
    assert output_row["top_results"][0]["workflow_score_components"]["feature_rerank_signature"] > 0


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


def test_retreieval_lab_rerank_diagnostic_feature_ltr_cli_cycle(tmp_path):
    runs = tmp_path / "runs.json"
    qrels = tmp_path / "qrels.jsonl"
    features = tmp_path / "features.jsonl"
    feature_report = tmp_path / "feature_report.json"
    model = tmp_path / "model.json"
    calibrate_report = tmp_path / "calibrate_report.json"
    reranked = tmp_path / "reranked.json"
    attribution = tmp_path / "attribution.json"
    attribution_md = tmp_path / "attribution.md"
    runs.write_text(
        json.dumps(
            {
                "method": "retrieval_lab_run_artifact",
                "run_rows": representative_diagnostic_run_rows(),
                "cases": [{"case_id": "q-workflow"}],
            }
        ),
        encoding="utf-8",
    )
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "q-workflow", "item_id": "i-target", "grade": 3, "source": "llm_adjudicated", "grade_votes": [{"grade": 3, "judge_type": "llm", "confidence": 0.9}]}),
                json.dumps({"query_id": "q-workflow", "item_id": "i-veto", "grade": 0, "source": "llm_adjudicated", "grade_votes": [{"grade": 0, "judge_type": "llm", "confidence": 0.9}]}),
                json.dumps({"query_id": "q-workflow", "item_id": "i-semantic", "grade": 1, "source": "pooled_bootstrap", "grade_votes": [{"grade": 1, "judge_type": "bootstrap", "confidence": 0.7}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    export_summary = run_module(
        "rerank",
        "export-features",
        "--runs",
        str(runs),
        "--qrels",
        str(qrels),
        "--output",
        str(features),
        "--report-output",
        str(feature_report),
    )
    feature_rows = [json.loads(line) for line in features.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert export_summary["feature_row_count"] == 3
    assert export_summary["judged_row_count"] == 3
    assert feature_rows[0]["qrel_source"] == "llm_adjudicated"
    assert "graph_neighbor_score" in feature_rows[0]

    safe_features = tmp_path / "features_safe.jsonl"
    safe_feature_report = tmp_path / "feature_safe_report.json"
    safe_export_summary = run_module(
        "rerank",
        "export-features",
        "--runs",
        str(runs),
        "--qrels",
        str(qrels),
        "--feature-profile",
        "production_safe",
        "--output",
        str(safe_features),
        "--report-output",
        str(safe_feature_report),
    )
    safe_feature_rows = [json.loads(line) for line in safe_features.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert safe_export_summary["feature_profile"] == "production_safe"
    assert safe_export_summary["excluded_features"] == ["target_like_score"]
    assert {row["target_like_score"] for row in safe_feature_rows} == {0.0}
    assert {row["is_target"] for row in safe_feature_rows} == {False}
    assert safe_feature_rows[0]["feature_profile"] == "production_safe"

    calibrate_summary = run_module(
        "rerank",
        "calibrate",
        "--features",
        str(features),
        "--method",
        "coordinate_search",
        "--output",
        str(model),
        "--report-output",
        str(calibrate_report),
    )
    assert calibrate_summary["feature_row_count"] == 3
    assert "rerank_gap_closure_rate" in calibrate_summary
    assert json.loads(model.read_text(encoding="utf-8"))["weights"]

    holdout_model = tmp_path / "holdout_model.json"
    holdout_report = tmp_path / "holdout_report.json"
    holdout_summary = run_module(
        "rerank",
        "calibrate",
        "--features",
        str(safe_features),
        "--method",
        "coordinate_search",
        "--split-strategy",
        "query_hash",
        "--train-ratio",
        "0.5",
        "--output",
        str(holdout_model),
        "--report-output",
        str(holdout_report),
    )
    holdout_model_data = json.loads(holdout_model.read_text(encoding="utf-8"))
    assert holdout_summary["split_strategy"] == "query_hash"
    assert holdout_summary["train_row_count"] == 3
    assert holdout_summary["eval_row_count"] == 3
    assert "target_like_score" not in holdout_model_data["weights"]
    assert holdout_model_data["feature_profile"] == "production_safe"
    assert "eval_metrics" in json.loads(holdout_report.read_text(encoding="utf-8"))

    apply_summary = run_module(
        "rerank",
        "apply-calibrated",
        "--runs",
        str(runs),
        "--model",
        str(model),
        "--output",
        str(reranked),
        "--top-k",
        "2",
    )
    reranked_artifact = json.loads(reranked.read_text(encoding="utf-8"))
    assert apply_summary["run_count"] == 1
    assert reranked_artifact["method"] == "retrieval_lab_calibrated_reranked_runs"

    attribution_summary = run_module(
        "rerank",
        "attribute",
        "--runs",
        str(runs),
        "--qrels",
        str(qrels),
        "--features",
        str(features),
        "--output",
        str(attribution),
        "--markdown-output",
        str(attribution_md),
    )
    attribution_report = json.loads(attribution.read_text(encoding="utf-8"))
    assert attribution_summary["failure_count"] == 1
    assert attribution_report["failures"][0]["failure_category"] in {
        "feature_weight_misaligned",
        "stage_purpose_mismatch",
        "style_penalty_over_or_under_applied",
    }
    assert attribution_md.read_text(encoding="utf-8").startswith("# Rerank Attribution Report")


def test_retreieval_lab_judge_calibration_cli_is_offline_and_slice_aware(tmp_path):
    qrels = tmp_path / "qrels.jsonl"
    samples = tmp_path / "samples.jsonl"
    output = tmp_path / "judge_calibration.json"
    markdown = tmp_path / "judge_calibration.md"
    qrels.write_text(
        "\n".join(
            [
                json.dumps({"query_id": "natural_fuzzy_q1", "item_id": "i1", "grade": 3, "source": "llm_adjudicated", "grade_votes": [{"grade": 3, "judge_type": "llm", "judge_version": "v1", "confidence": 0.9}, {"grade": 2, "judge_type": "llm", "judge_version": "v1", "confidence": 0.8}]}),
                json.dumps({"query_id": "fuzzy_q2", "item_id": "i2", "grade": 0, "source": "pooled_bootstrap", "grade_votes": [{"grade": 0, "judge_type": "bootstrap", "confidence": 0.4}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    samples.write_text(json.dumps({"query_id": "fuzzy_q2", "item_id": "i2", "low_confidence_reason": "bootstrap"}) + "\n", encoding="utf-8")

    summary = run_module(
        "qrels",
        "judge-calibration",
        "--samples",
        str(samples),
        "--qrels",
        str(qrels),
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["status"] == "offline_analysis"
    assert summary["llm_call_count"] == 0
    assert "natural_fuzzy" in report["slice_metrics"]
    assert report["recommended_next_adjudication_queue"]
    assert markdown.read_text(encoding="utf-8").startswith("# Judge Calibration Report")


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
                    "Recall@10": 0.74,
                    "scene_level_recall_at_10": 0.59,
                    "target_recall_at_10": 0.6,
                    "stage_level_hit_at_3": 0.91,
                    "purpose_level_hit_at_3": 0.91,
                    "style_violation_at_3": 0.06,
                    "failure_rate": 0.05,
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
    html = report.with_name("core_metrics_trend.html")
    assert html.exists()
    assert "Recall@10" in html.read_text(encoding="utf-8")
    diagnostic_html = report.with_name("diagnostic_metrics_trend.html")
    assert diagnostic_html.exists()
    assert "Diagnostic Metrics Trend" in diagnostic_html.read_text(encoding="utf-8")
    diagnostic_text = diagnostic_html.read_text(encoding="utf-8")
    assert "data-metric=\"ndcg_headroom_at_10\"" in diagnostic_text
    assert "chart-higher-is-better" in diagnostic_text
    assert "chart-lower-is-better" in diagnostic_text
    assert "chart-opportunity" in diagnostic_text
    assert "Show lower-is-better" in diagnostic_text
    assert "data-chart-tab=\"lower_is_better\"" in diagnostic_text
    assert "setActiveChart('lower_is_better')" in diagnostic_text
    assert "chart-panel is-active" in diagnostic_text
    assert "data-toggle-metric-controls" in diagnostic_text
    assert "Hide metric controls" in diagnostic_text
    assert "Show metric controls" in diagnostic_text
    assert "metric-controls-body" in diagnostic_text
    assert 'data-metric="rerank_oracle_gap_ndcg_at_10" data-direction="opportunity" checked' in diagnostic_text
    assert 'data-metric="llm_seconds_per_judgement" data-direction="lower_is_better" checked' not in diagnostic_text
    assert "Latest Observed" in diagnostic_text
    assert "Source / Use / Meaning" in diagnostic_text
    assert "sparkline" in diagnostic_text
    assert "data-tooltip" in diagnostic_text
    assert "metric-hit" in diagnostic_text
    assert "metric-dot" not in diagnostic_text
    assert 'data-metric-row="llm_seconds_per_judgement" data-direction="lower_is_better"' in diagnostic_text
    assert "row.dataset.direction !== direction" in diagnostic_text
    assert "Latest observed=" in diagnostic_text
    assert "Use: Estimates the maximum nDCG@10 lift" in diagnostic_text
    diagnostic_data = diagnostic_text.split('id="diagnostic-metrics-data">', 1)[1].split("</script>", 1)[0]
    diagnostic_payload = json.loads(diagnostic_data)
    assert "&quot;" not in diagnostic_data
    assert "ndcg_headroom_at_10" in diagnostic_payload["series"]
    assert diagnostic_payload["groups"]["lower_is_better"]
    markdown_text = report.read_text(encoding="utf-8")
    assert "Diagnostic Metric Trends" in markdown_text
    assert "bootstrap-only qrels rate" in markdown_text


def test_retreieval_lab_capability_raw_metrics_include_core_trend_fields():
    raw = extract_capability_raw_metrics(
        [
            {
                "path": "fuzzy.json",
                "method": "retrieval_lab_native_compare_ranking_workflows",
                "summary": {
                    "nDCG@10": 0.62,
                    "MRR@10": 0.86,
                    "Recall@10": 0.74,
                    "target_recall_at_10": 0.6,
                    "stage_level_hit_at_3": 0.91,
                    "purpose_level_hit_at_3": 0.92,
                    "style_violation_at_3": 0.02,
                    "failure_rate": 0.05,
                },
                "graded_metrics": {},
                "elapsed_seconds": 0.2,
            }
        ],
        missing_reports=[],
    )

    assert raw["recall_at_10"] == 0.74
    assert raw["target_recall_at_10"] == 0.6
    assert raw["scene_level_recall_at_10"] == 0.6
    assert raw["failure_rate"] == 0.05
    assert raw["ndcg_headroom_at_10"] == 0.38
    assert raw["style_violation_gap"] == 0.0


def test_retreieval_lab_capability_raw_metrics_include_diagnostic_fields():
    raw = extract_capability_raw_metrics(
        [
            {
                "path": "qrels.json",
                "method": "retrieval_lab_qrels_judge_calibration",
                "summary": {
                    "qrels_count": 100,
                    "bootstrap_only_count": 70,
                    "needs_adjudication_count": 20,
                    "llm_count": 30,
                    "vote_conflict_rate": 0.08,
                },
                "graded_metrics": {},
                "elapsed_seconds": 0.1,
            },
            {
                "path": "llm.json",
                "method": "retrieval_lab_llm_qrels_adjudication",
                "summary": {"judgement_count": 10, "elapsed_seconds": 20.0},
                "graded_metrics": {},
                "elapsed_seconds": 20.0,
            },
            {
                "path": "rerank.json",
                "method": "retrieval_lab_calibrated_rerank",
                "summary": {
                    "baseline_nDCG@10": 0.7,
                    "calibrated_nDCG@10": 0.75,
                    "oracle_rerank_nDCG@10": 0.85,
                },
                "graded_metrics": {},
                "elapsed_seconds": 0.2,
            },
            {
                "path": "anti.json",
                "method": "retrieval_lab_native_anti_overfit_fuzzy_evaluation",
                "summary": {},
                "graded_metrics": {},
                "scenarios": {
                    "metadata_assisted_style_safe": {"summary": {"nDCG@10": 0.86, "style_violation_at_3": 0.04}},
                    "natural_style_safe": {"summary": {"nDCG@10": 0.79, "style_violation_at_3": 0.0}},
                },
                "elapsed_seconds": 0.3,
            },
        ],
        missing_reports=[],
    )

    assert raw["bootstrap_only_rate"] == 0.7
    assert raw["needs_adjudication_rate"] == 0.2
    assert raw["llm_coverage_rate"] == 0.3
    assert raw["llm_seconds_per_judgement"] == 2.0
    assert raw["rerank_oracle_gap_ndcg_at_10"] == 0.15
    assert raw["rerank_realized_gain_ndcg_at_10"] == 0.05
    assert raw["rerank_gap_closure_rate"] == 0.333333
    assert raw["all_fuzzy_ndcg_at_10"] == 0.86
    assert raw["natural_fuzzy_ndcg_at_10"] == 0.79
    assert raw["natural_fuzzy_style_violation_at_3"] == 0.0


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
    from retrieval_lab.indexes import validate_index_manifest
    from retrieval_lab.llm import validate_llm_judgement
    from retrieval_lab.planners import validate_query_plan
    from retrieval_lab.retrieval import validate_retrieval_run_config, validate_run_row

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


def test_retreieval_lab_planner_keeps_borrowed_objects_from_becoming_negative_leaks():
    query = (
        "画面可以借用养老金融顾问、长期资产配置工具和风险画像系统、账本、家庭合照、"
        "长期规划时间线、操作界面这类相似元素，但不要做成技术展示。我真正要的是开场，"
        "重点是opening、建立问题、建立真实感，不要像技术展示或功能说明，而要让画面承担别的叙事任务。"
    )

    report = plan_many([query], planner="multi_query", use_cache=False)
    plan = report["plans"][0]
    positive_texts = [plan["positive_query"], *(row["text"] for row in plan["rewrites"]), plan.get("hyde_text", "")]

    assert "tech_showoff" in plan["negative_style"]
    assert report["summary"]["negative_leak_rate"] == 0
    assert not any("技术展示" in text or "功能说明" in text for text in positive_texts)


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


def test_retreieval_lab_llm_structured_planner_fake_mode_keeps_negative_text_out():
    report = plan_many(
        ["need grounded setup without product pitch"],
        planner="llm_structured",
        cache_path=None,
        config={"require_llm": False, "budget_cny": 20},
    )
    plan = report["plans"][0]

    assert plan["planner"] == "llm_structured"
    assert report["summary"]["llm_call_count"] == 0
    assert report["summary"]["negative_leak_rate"] == 0
    assert "product pitch" not in plan["positive_query"].lower()
    assert plan["provenance"]["llm_used"] is False


def test_retreieval_lab_llm_negative_style_requires_explicit_negative_constraint():
    assert llm_negative_style_values(["ad_like"], {"negative_style": [], "negative_constraints": []}, "make it documentary") == []
    assert llm_negative_style_values(["ad_like"], {"negative_style": [], "negative_constraints": []}, "without product pitch") == [
        "ad_like"
    ]


def test_retreieval_lab_style_safe_llm_planner_keeps_risk_terms_out_of_positive_text():
    report = plan_many(
        ["need documentary campaign warmth without product pitch"],
        planner="style_safe_llm_structured",
        cache_path=None,
        config={"require_llm": False, "budget_cny": 20},
    )
    plan = report["plans"][0]
    positive_texts = [plan["positive_query"], plan.get("hyde_text", "")]
    positive_texts.extend(row.get("text", "") for row in plan.get("rewrites", []))

    assert plan["planner"] == "style_safe_llm_structured"
    assert report["summary"]["llm_call_count"] == 0
    assert "product_pitch" in plan["negative_style"]
    assert all("product pitch" not in text.lower() for text in positive_texts)
    assert all("campaign" not in text.lower() for text in positive_texts)
    assert plan["provenance"]["llm_used"] is False


def test_retreieval_lab_planner_understands_real_natural_quality_control_queries():
    report = plan_many(
        ["医疗设备校准与质控复核的技术细节"],
        planner="style_safe_llm_structured",
        cache_path=None,
        config={"require_llm": False, "budget_cny": 20},
    )
    plan = report["plans"][0]

    assert "technology_showcase" in plan["desired_stage"]
    assert "show_technology" in plan["positive_purposes"]
    assert "prove_capability" in plan["positive_purposes"]


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


def test_retreieval_lab_native_retrieval_run_honors_style_safe_ranking_key(tmp_path):
    dataset = tmp_path / "style_dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "dataset_id": "style_dataset",
                "cases": [
                    {
                        "case_id": "case_style",
                        "case_type": "positive",
                        "user_input": "need grounded setup without advertising",
                        "expected_prefer": {
                            "fixture_id": "f1",
                            "scene_id": "scene_safe",
                            "retrieval_id": "ret_safe",
                            "script_stage": "setup",
                            "creative_purpose": ["build_reality"],
                            "title": "Grounded documentary setup",
                            "style": "documentary",
                        },
                        "target": {
                            "fixture_id": "f1",
                            "scene_id": "scene_risk",
                            "retrieval_id": "ret_risk",
                            "script_stage": "setup",
                            "creative_purpose": ["build_reality"],
                            "title": "Advertising campaign setup",
                            "style": "advertising campaign",
                            "style_risks": ["ad_like"],
                        },
                        "target_summary": "Grounded setup with real location pressure.",
                        "target_tags_text": "setup real location",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = retrieval_run(
        dataset_path=dataset,
        split="all",
        limit=0,
        planner="multi_query",
        planner_cache=None,
        top_k=2,
        ranking_key="style_safe_signature",
    )
    row = next(iter(artifact["run_rows"].values()))[0]

    assert row["top_results"][0]["item_id"] == "f1::scene_safe::ret_safe"
    assert row["top_results"][1]["style_guardrail_action"] == "strong_style_penalty"


def test_retreieval_lab_fast_scorer_matches_legacy_scorer_for_small_dataset(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    data = json.loads(dataset.read_text(encoding="utf-8"))
    cases = data["cases"]
    items = index_items_from_cases(cases)
    plan = plan_many(["need grounded setup without product pitch"], planner="multi_query", cache_path=None)["plans"][0]
    prepared = prepare_retrieval_index(items)

    fast = score_items(items, plan=plan, prepared_index=prepared)
    legacy = rows_with_rrf_scores([score_item(item, plan=plan, prepared_index=prepared) for item in items])

    assert [row["item_id"] for row in sorted(fast, key=lambda row: -row["final_score"])] == [
        row["item_id"] for row in sorted(legacy, key=lambda row: -row["final_score"])
    ]


def test_retreieval_lab_retrieval_benchmark_compact_output(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "benchmark.json"
    args = SimpleNamespace(
        dataset=dataset,
        split="all",
        limit=0,
        repeat_to=20,
        planner="multi_query",
        planner_cache=None,
        no_cache=True,
        ranking_key="hybrid_rrf_constraints_signature",
        top_k=2,
        candidate_depth=2,
        compact_output=True,
        output=output,
    )

    report = retrieval_benchmark_command(args)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["summary"]["case_count"] == 20
    assert report["summary"]["llm_call_count"] == 0
    assert report["summary"]["score_cache_hits"] > 0
    assert "run_rows" not in written
    assert written["summary"]["report_bytes"] < 5000000


def test_retreieval_lab_build_qrels_can_target_fuzzy_variants(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "fuzzy_qrels_report.json"
    qrels = tmp_path / "fuzzy_qrels.jsonl"

    summary = run_module(
        "build-pooled-qrels",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--case-variants",
        "fuzzy",
        "--qrels-output",
        str(qrels),
        "--compact-output",
        "--output",
        str(output),
    )
    rows = [json.loads(line) for line in qrels.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert summary["query_count"] == 10
    assert rows
    assert all("::" in row["query_id"] for row in rows)


def test_retreieval_lab_workflow_compare_can_target_fuzzy_variants(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "fuzzy_workflow_report.json"

    summary = run_module(
        "compare-ranking-workflows",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--case-variants",
        "fuzzy",
        "--ranking-keys",
        "hybrid_rrf_constraints_signature,style_safe_signature",
        "--compact-output",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["case_count"] == 10
    assert report["summary"]["ranking_keys"] == ["hybrid_rrf_constraints_signature", "style_safe_signature"]
    assert report["summary"]["compat_backend_used"] is False


def test_retreieval_lab_rerank_upper_bound_can_target_fuzzy_variants(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "fuzzy_rerank_upper_bound.json"

    summary = run_module(
        "compare-rerank-upper-bound",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--case-variants",
        "fuzzy",
        "--ranking-key",
        "hybrid_rrf_constraints_signature",
        "--compact-output",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["case_count"] == 10
    assert report["summary"]["row_counts"]["baseline"] == 10
    assert report["summary"]["oracle_run"].endswith("qrels_oracle_rerank@20")
    assert report["summary"]["compat_backend_used"] is False


def test_retreieval_lab_rerank_gate_can_target_fuzzy_variants(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "fuzzy_rerank_gate.json"

    summary = run_module(
        "validate-rerank-gate",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--case-variants",
        "fuzzy",
        "--ranking-key",
        "hybrid_rrf_constraints_signature",
        "--llm-sample-size",
        "2",
        "--compact-output",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["case_count"] == 10
    assert "selected_count" in report["summary"]
    assert report["summary"]["compat_backend_used"] is False
    sample = report["gated_rerank_sample"]
    assert sample["sample_size"] == 2
    assert "attribution_summary" in sample
    assert "reason_counts" in sample["attribution_summary"]
    assert isinstance(sample["attribution_examples"], list)


def test_retreieval_lab_llm_rerank_prompt_uses_summary_without_target_hint():
    row = {
        "case_id": "case_rerank",
        "user_input": "需要真实现场感，不要广告腔",
        "target_stage": "setup",
        "target_purposes": ["build_reality"],
        "query_plan": {
            "desired_stage": ["setup"],
            "negative_style": ["ad_like"],
            "scene_signature": {"place": ["hospital"], "action": ["inspection"]},
        },
        "query_constraints": {"negative_style": ["ad_like"]},
        "top_results": [
            {
                "item_id": "item_1",
                "score": 0.72,
                "signature_score": 0.5,
                "purpose_score": 1.0,
                "workflow_score_components": {"rrf": 0.1, "signature": 0.5, "constraint": 1.0},
                "constraint_hits": {"desired_stage": ["setup"], "negative_style": ["ad_like"]},
                "metadata": {
                    "script_stage": "setup",
                    "creative_purpose": ["build_reality"],
                    "script_use_sentence": "Use when the scene needs grounded location pressure.",
                    "scene_signature": {"place": ["hospital"], "action": ["inspection"]},
                    "style_risks": ["ad_like"],
                },
                "style_guardrail_action": "strong_style_penalty",
                "risk_evidence": [{"style": "ad_like"}],
            }
        ],
    }

    payload = json.loads(llm_rerank_prompt(row, rerank_depth=1))
    candidate = payload["candidates"][0]

    assert "target_hint" not in payload
    assert payload["query_plan"]["scene_signature"]
    assert candidate["script_usecase"]["sentence"].startswith("Use when")
    assert candidate["scene_signature"]
    assert candidate["score_components"]["signature"] == 0.5
    assert candidate["style_guardrail_action"] == "strong_style_penalty"


def test_retreieval_lab_natural_fuzzy_does_not_expose_internal_metadata_keys(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    data = json.loads(dataset.read_text(encoding="utf-8"))

    variants = natural_fuzzy_variant_cases(data["cases"])

    assert len(variants) == 10
    assert all(row["fuzzy_set_type"] == "natural_language" for row in variants)
    assert all(not metadata_label_leaks(row["user_input"]) for row in variants)
    assert variants[0]["target"]["creative_purpose"] == ["build_reality"]


def test_retreieval_lab_validate_natural_fuzzy_reports_metadata_leakage(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "natural_fuzzy.json"

    summary = run_module(
        "validate-fuzzy-understanding",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--case-variants",
        "natural_fuzzy",
        "--ranking-key",
        "style_safe_signature",
        "--compact-output",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["count"] == 10
    assert summary["metadata_leak_rate"] == 0.0
    assert report["by_fuzzy_set_type"]["natural_language"]["metadata_leak_rate"] == 0.0


def test_retreieval_lab_anti_overfit_fuzzy_report_compares_natural_and_metadata_sets(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "anti_overfit.json"

    summary = run_module(
        "evaluate-anti-overfit-fuzzy",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--compact-output",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["scenario_count"] == 4
    assert "metadata_assisted_style_safe" in report["scenarios"]
    assert "natural_fuzzy_style_safe" in report["scenarios"]
    assert "natural_style_safe_metadata_leak_rate" in report["summary"]
    assert report["recommendation"]["bottleneck"]


def test_retreieval_lab_scene_graph_manifest_cli_builds_local_graph(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "scene_graph.json"
    report_output = tmp_path / "scene_graph_report.json"

    summary = run_module(
        "graph",
        "build-manifest",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--limit",
        "3",
        "--output",
        str(output),
        "--report-output",
        str(report_output),
    )
    graph = json.loads(output.read_text(encoding="utf-8"))
    report = json.loads(report_output.read_text(encoding="utf-8"))

    assert summary["node_count"] > 0
    assert summary["edge_count"] > 0
    assert report["method"] == "retrieval_lab_scene_graph_manifest"
    assert any(node["type"] == "card" for node in graph["nodes"])
    assert any(edge["type"] == "card_to_stage" for edge in graph["edges"])
    assert scene_graph_from_cases([])["nodes"] == []


def test_retreieval_lab_llm_generate_natural_fuzzy_defaults_to_local_generation(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "llm_natural_fuzzy.jsonl"
    report_output = tmp_path / "llm_natural_fuzzy_report.json"

    summary = run_module(
        "llm",
        "generate-natural-fuzzy",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--output",
        str(output),
        "--report-output",
        str(report_output),
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert summary["llm_call_count"] == 0
    assert summary["metadata_leak_rate"] == 0.0
    assert rows and rows[0]["fuzzy_set_type"] == "natural_language"


def test_retreieval_lab_llm_natural_fuzzy_response_accepts_provider_shapes():
    single = natural_fuzzy_response_rows({"base_case_id": "case_1", "query": "natural query"})
    wrapped = natural_fuzzy_response_rows({"queries": [{"base_case_id": "case_2", "query": "wrapped query"}]})
    mapped = natural_fuzzy_response_rows({"queries": {"case_3": "mapped query"}})

    assert single == [{"base_case_id": "case_1", "query": "natural query"}]
    assert wrapped == [{"base_case_id": "case_2", "query": "wrapped query"}]
    assert mapped == [{"base_case_id": "case_3", "query": "mapped query"}]


def test_retreieval_lab_llm_natural_fuzzy_prompt_uses_expected_prefer_for_hard_negative():
    prompt = natural_fuzzy_prompt(
        [
            {
                "case_id": "hard_negative_case",
                "target_summary": "适合用于团队协作段落，表现年轻团队开会讨论。",
                "target": {
                    "title": "bad target",
                    "script_stage": "team_work",
                    "creative_purpose": ["show_collaboration"],
                    "style": "bad_style",
                },
                "expected_prefer": {
                    "title": "preferred opening",
                    "script_stage": "opening",
                    "creative_purpose": ["establish_problem", "build_reality"],
                    "industry": "medical_technology_and_digital_health",
                    "style": "story_driven_enterprise_brand_film",
                },
            }
        ]
    )

    assert "preferred opening" in prompt
    assert "opening" in prompt
    assert "establish_problem" in prompt
    assert "年轻团队开会讨论" not in prompt


def test_retreieval_lab_fuzzy_query_file_keeps_style_safe_llm_planner_natural(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    query_file = tmp_path / "natural_queries.jsonl"
    output = tmp_path / "natural_query_file_eval.json"
    query_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "base_case_id": "case_setup",
                        "variant_type": "llm_natural_fuzzy",
                        "fuzzy_set_type": "natural_language",
                        "user_input": "找一个真实克制的段落，重点是现场感和需求自然成立。",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_module(
        "validate-fuzzy-understanding",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--fuzzy-query-file",
        str(query_file),
        "--query-planner",
        "style_safe_llm_structured",
        "--llm-sample-size",
        "0",
        "--ranking-key",
        "style_safe_signature",
        "--compact-output",
        "--output",
        str(output),
    )

    assert summary["llm_call_count"] == 0
    assert summary["metadata_leak_rate"] == 0.0
    assert summary["negative_leak_rate"] == 0.0


def test_retreieval_lab_planner_understands_dataset_stage_and_purpose_keys():
    plan = plan_many(
        ["导演笔记：找一个scale_reveal阶段的可复用经验，重点是show_network"],
        planner="multi_query",
        cache_path=None,
    )["plans"][0]

    assert "scale_reveal" in plan["desired_stage"]
    assert "show_network" in plan["positive_purposes"]


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


def test_retreieval_lab_core_native_evaluate_hybrid_replaces_compat_backend(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    output = tmp_path / "evaluate_hybrid.json"

    summary = run_module(
        "evaluate-hybrid",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--limit",
        "0",
        "--output",
        str(output),
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert summary["compat_backend_used"] is False
    assert report["method"] == "retrieval_lab_native_evaluate_hybrid"
    assert report["run_rows"]


def test_retreieval_lab_core_native_fuzzy_and_rerank_reports(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    fuzzy_output = tmp_path / "fuzzy.json"
    rerank_output = tmp_path / "rerank.json"

    fuzzy = run_module(
        "validate-fuzzy-understanding",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--output",
        str(fuzzy_output),
    )
    rerank = run_module(
        "compare-rerank-upper-bound",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--output",
        str(rerank_output),
    )
    fuzzy_report = json.loads(fuzzy_output.read_text(encoding="utf-8"))
    rerank_report = json.loads(rerank_output.read_text(encoding="utf-8"))

    assert fuzzy["compat_backend_used"] is False
    assert fuzzy["negative_leak_rate"] == 0
    assert fuzzy_report["summary"]["count"] == 10
    assert rerank["compat_backend_used"] is False
    assert rerank["llm_call_count"] == 0
    assert "bottleneck" in rerank_report["summary"]


def test_retreieval_lab_core_native_mines_hard_negatives_and_certifies(tmp_path):
    dataset = small_retrieval_dataset(tmp_path / "dataset.json")
    report_output = tmp_path / "hard_negatives.json"
    hard_output = tmp_path / "hard_negatives.jsonl"
    certify_output = tmp_path / "certify.json"

    mining = run_module(
        "mine-hard-negatives",
        "--dataset",
        str(dataset),
        "--split",
        "all",
        "--output",
        str(report_output),
        "--hard-negatives-output",
        str(hard_output),
    )
    cert = run_module("migration", "certify", "--round-id", "unit", "--output", str(certify_output))
    cert_report = json.loads(certify_output.read_text(encoding="utf-8"))

    assert mining["compat_backend_used"] is False
    assert hard_output.exists()
    assert cert["core_experiment_replacement_coverage"] == 1.0
    assert cert["critical_command_compat_count"] == 0
    assert cert_report["summary"]["mocktesting_clean"] is True
