"""Optional LLM judges, planners, and rerankers. Default execution remains off."""
"""Optional LLM judgement contracts.

Real LLM calls remain opt-in; this package starts with the records that sampled
judging and reranking must emit.
"""

from retreieval_lab.llm.schema import (
    LLM_JUDGEMENT_SCHEMA_NAME,
    llm_judgement_contract,
    normalize_llm_judgement,
    validate_llm_judgement,
)

__all__ = ["LLM_JUDGEMENT_SCHEMA_NAME", "llm_judgement_contract", "normalize_llm_judgement", "validate_llm_judgement"]
