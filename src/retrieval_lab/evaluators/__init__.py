"""Metrics, failure attribution, and retrieval quality evaluators."""

from retrieval_lab.evaluators.failures import (
    DEFAULT_FAILURE_REPORT_PATH,
    analyze_failure_rows,
    analyze_failures_from_runs_command,
    classify_failure_from_artifact,
    explain_top1_win_from_artifact,
    failure_analysis_row_from_artifact,
)
from retrieval_lab.evaluators.gates import (
    DEFAULT_ROUND2_OUTPUT_DIR,
    build_round2_taxonomy_gate_report,
    classify_round2_residual,
    round2_taxonomy_gate_report_command,
    validate_round2_gate,
)
from retrieval_lab.evaluators.metrics import (
    graded_metrics,
    qrels_judged_coverage,
    recall_bound_rows,
    recall_bound_summary,
)
from retrieval_lab.evaluators.run_eval import (
    DEFAULT_RUN_EVAL_REPORT_PATH,
    evaluate_run_artifact_command,
    evaluate_run_rows,
    run_metric_selection_score,
)

__all__ = [
    "DEFAULT_FAILURE_REPORT_PATH",
    "DEFAULT_ROUND2_OUTPUT_DIR",
    "DEFAULT_RUN_EVAL_REPORT_PATH",
    "analyze_failure_rows",
    "analyze_failures_from_runs_command",
    "build_round2_taxonomy_gate_report",
    "classify_failure_from_artifact",
    "classify_round2_residual",
    "evaluate_run_artifact_command",
    "evaluate_run_rows",
    "explain_top1_win_from_artifact",
    "failure_analysis_row_from_artifact",
    "graded_metrics",
    "qrels_judged_coverage",
    "recall_bound_rows",
    "recall_bound_summary",
    "round2_taxonomy_gate_report_command",
    "run_metric_selection_score",
    "validate_round2_gate",
]
