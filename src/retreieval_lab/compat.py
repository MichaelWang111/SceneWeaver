from __future__ import annotations

import sys
from collections.abc import Sequence


LEGACY_COMMANDS: frozenset[str] = frozenset(
    {
        "build-index",
        "compact-embedding-cache",
        "search",
        "evaluate",
        "tune-constraints",
        "evaluate-leave-one-fixture-out",
        "validate-ranking-keys",
        "evaluate-hybrid",
        "compare-ranking-workflows",
        "compare-query-understanding",
        "validate-style-negatives",
        "validate-fuzzy-understanding",
        "build-graded-qrels",
        "evaluate-graded",
        "build-pooled-qrels",
        "evaluate-pooled",
        "analyze-failures",
        "analyze-recall-bound",
        "sample-active-qrels",
        "audit-qrels",
        "merge-adjudicated-qrels",
        "compare-strong-baselines",
        "compare-rerank-upper-bound",
        "evaluate-fuzzy-multirelevance",
        "validate-scene-signature",
        "validate-style-risk-mining",
        "mine-hard-negatives",
        "validate-rerank-gate",
        "compare-rerank-gates",
        "compare-experiments",
        "retrieval-flywheel-guide",
        "record-capability-cycle",
        "generate-capability-report",
        "generate-eval-report",
        "validate-paraphrase-stress",
    }
)


COMMAND_ALIASES: dict[tuple[str, ...], str] = {
    ("index", "build"): "build-index",
    ("index", "compact-cache"): "compact-embedding-cache",
    ("index", "inspect"): "index-inspect",
    ("index", "manifest"): "index-manifest",
    ("retrieval", "run"): "retrieval-run",
    ("retrieval", "compare-legacy"): "retrieval-compare-legacy",
    ("workflow", "compare"): "compare-ranking-workflows",
    ("workflow", "compare-runs"): "workflow-compare-runs",
    ("planner", "compare"): "planner-compare",
    ("planner", "plan"): "planner-plan",
    ("planner", "audit-cache"): "planner-audit-cache",
    ("qrels", "build-pooled"): "build-pooled-qrels",
    ("qrels", "pool-from-runs"): "pool-qrels-from-runs",
    ("qrels", "audit"): "audit-qrels",
    ("qrels", "sample-active"): "sample-active-qrels",
    ("qrels", "sample-active-from-runs"): "sample-active-qrels-from-runs",
    ("qrels", "merge-adjudicated"): "merge-adjudicated-qrels",
    ("eval", "graded"): "evaluate-graded",
    ("eval", "pooled"): "evaluate-pooled",
    ("eval", "hybrid"): "evaluate-hybrid",
    ("eval", "fuzzy"): "evaluate-fuzzy-multirelevance",
    ("eval", "fuzzy-understanding"): "validate-fuzzy-understanding",
    ("eval", "style-negatives"): "validate-style-negatives",
    ("eval", "style-risk"): "validate-style-risk-mining",
    ("eval", "paraphrase"): "validate-paraphrase-stress",
    ("eval", "ranking-keys"): "validate-ranking-keys",
    ("eval", "leave-one-fixture-out"): "evaluate-leave-one-fixture-out",
    ("eval", "rerank-upper-bound"): "compare-rerank-upper-bound",
    ("eval", "rerank-gates"): "compare-rerank-gates",
    ("eval", "failures"): "analyze-failures",
    ("eval", "recall-bound"): "analyze-recall-bound",
    ("rerank", "upper-bound"): "compare-rerank-upper-bound",
    ("rerank", "gates"): "compare-rerank-gates",
    ("cycle", "record"): "record-capability-cycle",
    ("dataset", "inspect"): "inspect-dataset",
    ("datasets", "inspect"): "inspect-dataset",
    ("report", "capability"): "generate-capability-report",
    ("report", "eval"): "generate-eval-report",
    ("artifact", "manifest"): "write-artifact-manifest",
    ("run", "export"): "export-run-artifact",
    ("run", "legacy"): "run-legacy-with-artifacts",
    ("run", "rerank"): "rerank-run-artifact",
    ("run", "workflow-rerank"): "workflow-run-artifact",
    ("run", "analyze-failures"): "analyze-failures-from-runs",
    ("run", "evaluate"): "evaluate-runs",
    ("eval", "runs"): "evaluate-runs",
    ("experiment", "compare"): "compare-experiments",
    ("infra", "audit"): "audit-infra-coverage",
    ("migration", "audit"): "migration-audit",
    ("schema", "catalog"): "schema-catalog",
    ("schema", "show"): "schema-show",
    ("schema", "validate"): "schema-validate",
    ("flywheel", "guide"): "retrieval-flywheel-guide",
}


def translate_argv(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    if not args:
        return args
    if args[0] in LEGACY_COMMANDS:
        return args
    if len(args) >= 2:
        alias = COMMAND_ALIASES.get((args[0], args[1]))
        if alias:
            return [alias, *args[2:]]
    return args


def run_mocktesting_backend(argv: Sequence[str]) -> None:
    from mocktesting.mock_retriever import main as legacy_main

    translated = translate_argv(argv)
    original_argv = sys.argv[:]
    try:
        sys.argv = ["retreieval_lab", *translated]
        legacy_main()
    finally:
        sys.argv = original_argv
