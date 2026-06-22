from __future__ import annotations


CORE_EXPERIMENT_COMMANDS = {
    "search",
    "evaluate",
    "evaluate-hybrid",
    "compare-ranking-workflows",
    "validate-ranking-keys",
    "compare-query-understanding",
    "validate-fuzzy-understanding",
    "validate-paraphrase-stress",
    "evaluate-fuzzy-multirelevance",
    "evaluate-anti-overfit-fuzzy",
    "evaluate-graded",
    "evaluate-pooled",
    "build-pooled-qrels",
    "build-graded-qrels",
    "sample-active-qrels",
    "compare-strong-baselines",
    "compare-rerank-upper-bound",
    "validate-rerank-gate",
    "compare-rerank-gates",
    "validate-style-negatives",
    "validate-style-risk-mining",
    "mine-hard-negatives",
    "validate-scene-signature",
    "analyze-failures",
    "analyze-recall-bound",
    "build-index",
    "compact-embedding-cache",
    "tune-constraints",
    "evaluate-leave-one-fixture-out",
}


__all__ = ["CORE_EXPERIMENT_COMMANDS"]
