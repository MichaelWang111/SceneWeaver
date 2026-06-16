"""Optional LLM judges, planners, and rerankers. Default execution remains off."""
"""Optional LLM judgement contracts.

Real LLM calls remain opt-in; this package starts with the records that sampled
judging and reranking must emit.
"""

from retrieval_lab.llm.schema import (
    LLM_JUDGEMENT_SCHEMA_NAME,
    llm_judgement_contract,
    normalize_llm_judgement,
    validate_llm_judgement,
)
from retrieval_lab.llm.adjudication import (
    DEFAULT_LLM_ADJUDICATION_OUTPUT,
    DEFAULT_LLM_ADJUDICATION_REPORT,
    DEFAULT_LLM_JUDGE_CACHE,
    DEFAULT_LLM_NATURAL_FUZZY_OUTPUT,
    DEFAULT_LLM_NATURAL_FUZZY_REPORT,
    llm_adjudicate_qrels_command,
    llm_generate_natural_fuzzy_command,
)
from retrieval_lab.llm.status import llm_status_command

__all__ = [
    "DEFAULT_LLM_ADJUDICATION_OUTPUT",
    "DEFAULT_LLM_ADJUDICATION_REPORT",
    "DEFAULT_LLM_JUDGE_CACHE",
    "DEFAULT_LLM_NATURAL_FUZZY_OUTPUT",
    "DEFAULT_LLM_NATURAL_FUZZY_REPORT",
    "LLM_JUDGEMENT_SCHEMA_NAME",
    "llm_adjudicate_qrels_command",
    "llm_generate_natural_fuzzy_command",
    "llm_status_command",
    "llm_judgement_contract",
    "normalize_llm_judgement",
    "validate_llm_judgement",
]
