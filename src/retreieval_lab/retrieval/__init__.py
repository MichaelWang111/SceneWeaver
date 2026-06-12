"""Recall workflows: semantic, lexical, hybrid, and fusion."""
"""Retrieval workflow contracts.

Native search execution is still being migrated. This layer defines the run
configuration and run-row schemas used by artifact-first evaluation.
"""

from retreieval_lab.retrieval.commands import retrieval_compare_legacy_command, retrieval_run_command
from retreieval_lab.retrieval.schema import (
    RUN_CONFIG_SCHEMA_NAME,
    RUN_ROW_SCHEMA_NAME,
    normalize_retrieval_run_config,
    normalize_run_row,
    retrieval_run_config_contract,
    run_row_contract,
    validate_retrieval_run_config,
    validate_run_row,
)
from retreieval_lab.retrieval.service import (
    DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT,
    DEFAULT_RETRIEVAL_RUN_OUTPUT,
    compare_run_artifacts,
    retrieval_run,
    score_item,
)

__all__ = [
    "DEFAULT_RETRIEVAL_LEGACY_COMPARISON_OUTPUT",
    "DEFAULT_RETRIEVAL_RUN_OUTPUT",
    "RUN_CONFIG_SCHEMA_NAME",
    "RUN_ROW_SCHEMA_NAME",
    "compare_run_artifacts",
    "normalize_retrieval_run_config",
    "normalize_run_row",
    "retrieval_compare_legacy_command",
    "retrieval_run",
    "retrieval_run_command",
    "retrieval_run_config_contract",
    "run_row_contract",
    "score_item",
    "validate_retrieval_run_config",
    "validate_run_row",
]
