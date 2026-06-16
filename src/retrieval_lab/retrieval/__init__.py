"""Recall workflows: semantic, lexical, hybrid, and fusion."""
"""Retrieval workflow contracts and native in-memory execution."""

from retrieval_lab.retrieval.commands import retrieval_compare_legacy_command, retrieval_run_command
from retrieval_lab.retrieval.benchmark import DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT, retrieval_benchmark_command
from retrieval_lab.retrieval.schema import (
    RUN_CONFIG_SCHEMA_NAME,
    RUN_ROW_SCHEMA_NAME,
    normalize_retrieval_run_config,
    normalize_run_row,
    retrieval_run_config_contract,
    run_row_contract,
    validate_retrieval_run_config,
    validate_run_row,
)
from retrieval_lab.retrieval.service import (
    DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT,
    DEFAULT_RETRIEVAL_RUN_OUTPUT,
    compare_run_artifacts,
    retrieval_run,
    retrieval_run_from_cases,
    score_item,
)

__all__ = [
    "DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT",
    "DEFAULT_RETRIEVAL_BENCHMARK_OUTPUT",
    "DEFAULT_RETRIEVAL_RUN_OUTPUT",
    "RUN_CONFIG_SCHEMA_NAME",
    "RUN_ROW_SCHEMA_NAME",
    "compare_run_artifacts",
    "normalize_retrieval_run_config",
    "normalize_run_row",
    "retrieval_compare_legacy_command",
    "retrieval_benchmark_command",
    "retrieval_run",
    "retrieval_run_from_cases",
    "retrieval_run_command",
    "retrieval_run_config_contract",
    "run_row_contract",
    "score_item",
    "validate_retrieval_run_config",
    "validate_run_row",
]
