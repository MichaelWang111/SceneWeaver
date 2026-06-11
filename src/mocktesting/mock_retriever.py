from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from mocktesting.constraint_layer import (
    DEFAULT_CONSTRAINT_PROFILE_PATH,
    load_constraint_profile,
    parse_query_constraints,
    profile_with_weights,
    score_constraints,
    write_constraint_profile,
)
from mocktesting.embedding_cache import (
    DEFAULT_CACHE_PATH,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    MAX_DASHSCOPE_BATCH_SIZE,
    EmbeddingCache,
    build_matrix_cache,
)
from mocktesting.embedding_text_builder import (
    DEFAULT_CHANNEL_WEIGHTS,
    build_item_channel_rows,
    build_query_channels,
    load_review_items,
    target_channel_for_query,
    target_item_id,
)
from mocktesting.eval_input_generator import PURPOSE_WORDS, STAGE_WORDS, canonical_stage
from mocktesting.query_planner import (
    DEFAULT_PLANNER_CACHE_NAME,
    ExperimentalQueryPlan,
    VALID_QUERY_PLANNERS,
    build_query_channels_for_plan,
    plan_has_negative_leak,
    plan_queries,
    planner_constraints,
)
from sceneweaver.retrieval.lexical import (
    DEFAULT_RRF_K,
    bm25_scores,
    ranked_indices_from_scores,
    reciprocal_rank_fusion,
    tokenize,
)
from sceneweaver.retrieval.query_plan import build_query_plan

DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_embedding_index.json"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_retrieval_report.json"
DEFAULT_SEARCH_OUTPUT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_search_result.json"
DEFAULT_INPUTS_PATH = Path(__file__).resolve().parent / "eval_inputs" / "review_generated_inputs.json"
DEFAULT_TUNING_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_constraint_tuning_report.json"
DEFAULT_LEAVE_ONE_FIXTURE_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_leave_one_fixture_report.json"
)
DEFAULT_RANKING_VALIDATION_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_ranking_key_validation_report.json"
)
DEFAULT_PARAPHRASE_STRESS_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_paraphrase_stress_report.json"
)
DEFAULT_WORKFLOW_COMPARISON_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_workflow_comparison_report.json"
)
DEFAULT_STYLE_NEGATIVE_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_style_negative_report.json"
)
DEFAULT_QUERY_UNDERSTANDING_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_query_understanding_report.json"
)
DEFAULT_FUZZY_UNDERSTANDING_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_fuzzy_understanding_report.json"
)
DEFAULT_QUERY_PLANNER_CACHE_PATH = Path(__file__).resolve().parent / "eval_outputs" / DEFAULT_PLANNER_CACHE_NAME
DEFAULT_GRADED_QRELS_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_graded_qrels.jsonl"
DEFAULT_GRADED_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_graded_eval_report.json"
DEFAULT_FAILURE_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_failure_analysis_report.json"
DEFAULT_SCENE_SIGNATURE_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_scene_signature_report.json"
)
DEFAULT_STYLE_RISK_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_style_risk_report.json"
DEFAULT_HARD_NEGATIVE_POOL_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_hard_negative_pool.jsonl"
DEFAULT_RERANK_GATE_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_rerank_gate_report.json"
DEFAULT_EXPERIMENT_COMPARISON_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_experiment_comparison.json"
)
DEFAULT_POOLED_QRELS_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_pooled_qrels.jsonl"
DEFAULT_POOLED_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_pooled_eval_report.json"
DEFAULT_RECALL_BOUND_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_recall_bound_report.json"
DEFAULT_ACTIVE_QRELS_SAMPLE_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_active_qrels_sample.jsonl"
DEFAULT_ADJUDICATED_QRELS_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_adjudicated_qrels.jsonl"
DEFAULT_STRONG_BASELINE_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_strong_baseline_report.json"
)
DEFAULT_RERANK_UPPER_BOUND_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_rerank_upper_bound_report.json"
)
DEFAULT_QRELS_AUDIT_REPORT_PATH = Path(__file__).resolve().parent / "eval_outputs" / "mock_qrels_audit_report.json"
DEFAULT_FUZZY_MULTI_REPORT_PATH = (
    Path(__file__).resolve().parent / "eval_outputs" / "mock_fuzzy_multirelevance_report.json"
)
DEFAULT_CAPABILITY_REGISTRY_PATH = Path(".tmp") / "capability_cycles.jsonl"
DEFAULT_CAPABILITY_CYCLE_PATH = Path(".tmp") / "capability_cycle_latest.json"
DEFAULT_CAPABILITY_REPORT_PATH = Path(".tmp") / "capability_report.md"
DEFAULT_CAPABILITY_CHART_DIR = Path(".tmp") / "capability_charts"

VALID_SPLITS = {"dev", "test", "hidden", "all"}
RANKING_KEYS = (
    "final_score",
    "embedding_only",
    "semantic_only",
    "lexical_only",
    "lexical_constraints",
    "lexical_constraints_signature",
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "script_use_only",
    "visual_tags_only",
    "experience_only",
    "combined_only",
    "constraints_only",
    "signature_only",
    "semantic_signature",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
)
LEGACY_RANKING_VALIDATION_KEYS = (
    "final_score",
    "embedding_only",
    "semantic_only",
    "lexical_only",
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "script_use_only",
    "visual_tags_only",
    "experience_only",
    "combined_only",
    "constraints_only",
)
DEFAULT_POOL_RANKING_KEYS = (
    "semantic_only",
    "lexical_only",
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
)
DEFAULT_STRONG_BASELINE_KEYS = (
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
)

TUNING_GRID = {
    "desired_stage_bonus": [0.06, 0.10, 0.12, 0.16],
    "forbidden_stage_penalty": [0.10, 0.15, 0.18, 0.22, 0.28],
    "negative_constraint_penalty": [0.00, 0.05, 0.08, 0.12],
}
HARD_FORBIDDEN_STAGE_VETO = 1000.0
STYLE_POSITIVE_BONUS = 0.8
STYLE_NEGATIVE_PENALTY = 1.5
STYLE_ALIASES = {
    "big_company_office": ("大厂", "互联网大厂", "泛泛办公", "空泛办公", "generic office"),
    "ad_like": ("广告", "宣传片腔", "硬广", "口号", "slogan"),
    "tech_showoff": ("炫技", "技术炫耀", "功能说明", "产品说明", "纯科技", "冷冰冰"),
    "human_warmth": ("有人味", "人味", "人的温度", "human"),
    "documentary": ("纪录片", "纪实", "观察", "documentary"),
    "real_location": ("真实现场", "真实场景", "现场感"),
    "corporate_report_tone": ("汇报片", "汇报感", "总结汇报", "report tone", "corporate report"),
    "slogan_driven": ("口号感", "喊口号", "价值口号", "slogan"),
    "product_pitch": ("产品卖点", "卖点堆叠", "产品推销", "pitch", "selling points"),
    "tech_coldness": ("冷科技", "冷冰冰", "技术冷感", "cold tech"),
    "fortune_500_polish": ("世界500强", "500强", "国际大牌质感", "fortune 500"),
    "generic_brand_film": ("宣传片腔", "品牌片套路", "企业宣传片", "generic brand film"),
}
CONSTRAINT_RANKING_KEYS = {
    "final_score",
    "constraints_only",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
    "lexical_constraints",
    "lexical_constraints_signature",
}
LEXICAL_RANKING_KEYS = {
    "lexical_only",
    "lexical_constraints",
    "lexical_constraints_signature",
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
}
RRF_RANKING_KEYS = {
    "hybrid_rrf",
    "hybrid_rrf_constraints",
    "hybrid_rrf_constraints_rerank",
    "hybrid_rrf_constraints_signature",
    "adaptive_signature",
}
STYLE_FINAL_SCORE_KEYS = {"final_score", "constraints_only", "hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}
SIGNATURE_SCORE_WEIGHT = 0.35


@dataclass
class PreparedMockIndex:
    item_ids: list[str]
    item_index_by_id: dict[str, int]
    metadata: list[dict[str, Any]]
    lexical_texts: list[str]
    lexical_text_lowers: list[str]
    constraint_text_lowers: list[str]
    scene_signatures: list[dict[str, list[str]]]
    signature_token_sets: list[dict[str, set[str]]]
    stage_values: np.ndarray
    style_masks: dict[str, np.ndarray]
    channel_vectors: dict[str, np.ndarray]
    bm25_inverted: dict[str, list[tuple[int, int]]]
    bm25_doc_norms: np.ndarray
    bm25_doc_count: int
    bm25_term_vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def bm25_scores(self, query_terms: list[str], *, k1: float = 1.5) -> np.ndarray:
        if not query_terms or self.bm25_doc_count == 0:
            return np.zeros(self.bm25_doc_count, dtype=np.float64)
        scores = np.zeros(self.bm25_doc_count, dtype=np.float64)
        query_counts = Counter(query_terms)
        for term, query_count in query_counts.items():
            scores += query_count * self.bm25_term_vector(term, k1=k1)
        return np.round(scores, 6)

    def bm25_term_vector(self, term: str, *, k1: float = 1.5) -> np.ndarray:
        cached = self.bm25_term_vectors.get(term)
        if cached is not None:
            return cached
        vector = np.zeros(self.bm25_doc_count, dtype=np.float64)
        postings = self.bm25_inverted.get(term)
        if postings:
            df = len(postings)
            idf = math.log(1 + (self.bm25_doc_count - df + 0.5) / (df + 0.5))
            for doc_index, tf in postings:
                denom = tf + self.bm25_doc_norms[doc_index]
                vector[doc_index] = idf * (tf * (k1 + 1) / denom)
        self.bm25_term_vectors[term] = vector
        return vector


@dataclass
class FastCaseSignals:
    case: dict[str, Any]
    user_input: str
    semantic_scores: np.ndarray
    channel_scores: dict[str, np.ndarray]
    query_constraints: dict[str, Any]
    query_plan: Any
    query_channels: list[dict[str, Any]] = field(default_factory=list)
    lexical_scores: np.ndarray | None = None
    rrf_scores: np.ndarray | None = None
    constraint_scores: np.ndarray | None = None
    constraint_hits: list[dict[str, list[str]]] | None = None
    forbidden_stage_mask: np.ndarray | None = None
    query_signature: dict[str, list[str]] = field(default_factory=dict)
    signature_scores: np.ndarray | None = None
    computed: set[str] = field(default_factory=set)


def main() -> None:
    parser = argparse.ArgumentParser(description="MockTesting multi-channel embedding retriever.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index")
    add_common_paths(build_parser)
    build_parser.add_argument("--dry-run", action="store_true")

    compact_cache_parser = subparsers.add_parser("compact-embedding-cache")
    add_common_paths(compact_cache_parser)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("user_input")
    add_common_paths(search_parser)
    add_constraint_args(search_parser)
    add_planner_args(search_parser)
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--output", type=Path, default=DEFAULT_SEARCH_OUTPUT_PATH)

    eval_parser = subparsers.add_parser("evaluate")
    add_common_paths(eval_parser)
    add_constraint_args(eval_parser)
    eval_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    eval_parser.add_argument("--top-k", type=int, default=10)
    eval_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)

    tune_parser = subparsers.add_parser("tune-constraints")
    add_common_paths(tune_parser)
    tune_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    tune_parser.add_argument("--limit", type=int, default=0)
    tune_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="dev")
    tune_parser.add_argument("--top-k", type=int, default=10)
    tune_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    tune_parser.add_argument("--output", type=Path, default=DEFAULT_TUNING_REPORT_PATH)

    lofo_parser = subparsers.add_parser("evaluate-leave-one-fixture-out")
    add_common_paths(lofo_parser)
    lofo_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    lofo_parser.add_argument("--limit", type=int, default=0)
    lofo_parser.add_argument("--top-k", type=int, default=10)
    lofo_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    lofo_parser.add_argument("--output", type=Path, default=DEFAULT_LEAVE_ONE_FIXTURE_REPORT_PATH)

    validate_parser = subparsers.add_parser("validate-ranking-keys")
    add_common_paths(validate_parser)
    validate_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    validate_parser.add_argument("--limit", type=int, default=0)
    validate_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    validate_parser.add_argument("--top-k", type=int, default=10)
    validate_parser.add_argument("--llm-sample-size", type=int, default=0)
    validate_parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    validate_parser.add_argument("--llm-retries", type=int, default=0)
    validate_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_RANKING_VALIDATION_REPORT_PATH)

    hybrid_parser = subparsers.add_parser("evaluate-hybrid")
    add_common_paths(hybrid_parser)
    hybrid_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    hybrid_parser.add_argument("--limit", type=int, default=0)
    hybrid_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    hybrid_parser.add_argument("--top-k", type=int, default=10)
    hybrid_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    hybrid_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    hybrid_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    hybrid_parser.add_argument("--include-debug-text", action="store_true")
    hybrid_parser.add_argument("--include-planner-debug", action="store_true")
    add_planner_args(hybrid_parser)

    compare_parser = subparsers.add_parser("compare-ranking-workflows")
    add_common_paths(compare_parser)
    compare_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    compare_parser.add_argument("--limit", type=int, default=0)
    compare_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    compare_parser.add_argument("--top-k", type=int, default=10)
    compare_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    compare_parser.add_argument("--output", type=Path, default=DEFAULT_WORKFLOW_COMPARISON_REPORT_PATH)
    compare_parser.add_argument("--markdown-output", type=Path, default=None)
    compare_parser.add_argument("--include-debug-text", action="store_true")
    compare_parser.add_argument("--include-planner-debug", action="store_true")
    add_planner_args(compare_parser)

    understanding_parser = subparsers.add_parser("compare-query-understanding")
    add_common_paths(understanding_parser)
    understanding_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    understanding_parser.add_argument("--limit", type=int, default=0)
    understanding_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    understanding_parser.add_argument("--top-k", type=int, default=10)
    understanding_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    understanding_parser.add_argument("--query-planners", default="rule,multi_query,hyde_card")
    understanding_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    understanding_parser.add_argument("--output", type=Path, default=DEFAULT_QUERY_UNDERSTANDING_REPORT_PATH)
    understanding_parser.add_argument("--markdown-output", type=Path, default=None)
    understanding_parser.add_argument("--include-debug-text", action="store_true")
    understanding_parser.add_argument("--include-planner-debug", action="store_true")
    add_planner_args(understanding_parser, include_query_planner=False)

    style_parser = subparsers.add_parser("validate-style-negatives")
    add_common_paths(style_parser)
    style_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    style_parser.add_argument("--limit", type=int, default=0)
    style_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    style_parser.add_argument("--top-k", type=int, default=10)
    style_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    style_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    style_parser.add_argument("--output", type=Path, default=DEFAULT_STYLE_NEGATIVE_REPORT_PATH)
    style_parser.add_argument("--include-debug-text", action="store_true")
    style_parser.add_argument("--include-planner-debug", action="store_true")
    add_planner_args(style_parser)

    fuzzy_parser = subparsers.add_parser("validate-fuzzy-understanding")
    add_common_paths(fuzzy_parser)
    fuzzy_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    fuzzy_parser.add_argument("--limit", type=int, default=10)
    fuzzy_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    fuzzy_parser.add_argument("--case-type", choices=["simple_positive", "hard_positive"], default="simple_positive")
    fuzzy_parser.add_argument("--top-k", type=int, default=10)
    fuzzy_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    fuzzy_parser.add_argument("--variant-types", default="")
    fuzzy_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    fuzzy_parser.add_argument("--output", type=Path, default=DEFAULT_FUZZY_UNDERSTANDING_REPORT_PATH)
    fuzzy_parser.add_argument("--dry-run", action="store_true")
    fuzzy_parser.add_argument("--include-debug-text", action="store_true")
    fuzzy_parser.add_argument("--include-planner-debug", action="store_true")
    add_planner_args(fuzzy_parser)

    qrels_parser = subparsers.add_parser("build-graded-qrels")
    add_common_paths(qrels_parser)
    qrels_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    qrels_parser.add_argument("--limit", type=int, default=60)
    qrels_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    qrels_parser.add_argument("--top-k", type=int, default=20)
    qrels_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints_signature")
    qrels_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    qrels_parser.add_argument("--qrels-output", type=Path, default=DEFAULT_GRADED_QRELS_PATH)
    qrels_parser.add_argument("--output", type=Path, default=DEFAULT_GRADED_REPORT_PATH)
    qrels_parser.add_argument("--llm-sample-size", type=int, default=0)
    add_planner_args(qrels_parser)

    graded_parser = subparsers.add_parser("evaluate-graded")
    add_common_paths(graded_parser)
    graded_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    graded_parser.add_argument("--limit", type=int, default=60)
    graded_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    graded_parser.add_argument("--top-k", type=int, default=10)
    graded_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints_signature")
    graded_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    graded_parser.add_argument("--qrels", type=Path, default=DEFAULT_GRADED_QRELS_PATH)
    graded_parser.add_argument("--output", type=Path, default=DEFAULT_GRADED_REPORT_PATH)
    graded_parser.add_argument("--markdown-output", type=Path, default=None)
    add_planner_args(graded_parser)

    pooled_qrels_parser = subparsers.add_parser("build-pooled-qrels")
    add_common_paths(pooled_qrels_parser)
    pooled_qrels_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    pooled_qrels_parser.add_argument("--limit", type=int, default=60)
    pooled_qrels_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    pooled_qrels_parser.add_argument("--pool-depth", type=int, default=20)
    pooled_qrels_parser.add_argument("--top-k", type=int, default=10)
    pooled_qrels_parser.add_argument("--pool-ranking-keys", default=",".join(DEFAULT_POOL_RANKING_KEYS))
    pooled_qrels_parser.add_argument("--pool-query-planners", default="rule")
    pooled_qrels_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    pooled_qrels_parser.add_argument("--qrels-output", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    pooled_qrels_parser.add_argument("--output", type=Path, default=DEFAULT_POOLED_REPORT_PATH)
    add_planner_args(pooled_qrels_parser, include_query_planner=False)

    pooled_eval_parser = subparsers.add_parser("evaluate-pooled")
    add_common_paths(pooled_eval_parser)
    pooled_eval_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    pooled_eval_parser.add_argument("--limit", type=int, default=60)
    pooled_eval_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    pooled_eval_parser.add_argument("--top-k", type=int, default=10)
    pooled_eval_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    pooled_eval_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    pooled_eval_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    pooled_eval_parser.add_argument("--output", type=Path, default=DEFAULT_POOLED_REPORT_PATH)
    pooled_eval_parser.add_argument("--markdown-output", type=Path, default=None)
    add_planner_args(pooled_eval_parser)

    failure_parser = subparsers.add_parser("analyze-failures")
    add_common_paths(failure_parser)
    failure_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    failure_parser.add_argument("--limit", type=int, default=60)
    failure_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    failure_parser.add_argument("--top-k", type=int, default=10)
    failure_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints_signature")
    failure_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    failure_parser.add_argument("--output", type=Path, default=DEFAULT_FAILURE_REPORT_PATH)
    failure_parser.add_argument("--markdown-output", type=Path, default=None)
    add_planner_args(failure_parser)

    recall_bound_parser = subparsers.add_parser("analyze-recall-bound")
    add_common_paths(recall_bound_parser)
    recall_bound_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    recall_bound_parser.add_argument("--limit", type=int, default=60)
    recall_bound_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    recall_bound_parser.add_argument("--candidate-depth", type=int, default=100)
    recall_bound_parser.add_argument("--top-k", type=int, default=10)
    recall_bound_parser.add_argument("--baseline-ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    recall_bound_parser.add_argument("--pool-ranking-keys", default=",".join(DEFAULT_POOL_RANKING_KEYS))
    recall_bound_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    recall_bound_parser.add_argument("--output", type=Path, default=DEFAULT_RECALL_BOUND_REPORT_PATH)
    recall_bound_parser.add_argument("--markdown-output", type=Path, default=None)
    add_planner_args(recall_bound_parser)

    active_qrels_parser = subparsers.add_parser("sample-active-qrels")
    add_common_paths(active_qrels_parser)
    active_qrels_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    active_qrels_parser.add_argument("--limit", type=int, default=60)
    active_qrels_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    active_qrels_parser.add_argument("--pool-depth", type=int, default=20)
    active_qrels_parser.add_argument("--sample-size", type=int, default=80)
    active_qrels_parser.add_argument("--pool-ranking-keys", default=",".join(DEFAULT_POOL_RANKING_KEYS))
    active_qrels_parser.add_argument("--pool-query-planners", default="rule")
    active_qrels_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    active_qrels_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    active_qrels_parser.add_argument("--output", type=Path, default=DEFAULT_ACTIVE_QRELS_SAMPLE_PATH)
    active_qrels_parser.add_argument("--include-judged", action="store_true")
    add_planner_args(active_qrels_parser, include_query_planner=False)

    audit_qrels_parser = subparsers.add_parser("audit-qrels")
    audit_qrels_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    audit_qrels_parser.add_argument("--output", type=Path, default=DEFAULT_QRELS_AUDIT_REPORT_PATH)
    audit_qrels_parser.add_argument("--markdown-output", type=Path, default=None)

    merge_qrels_parser = subparsers.add_parser("merge-adjudicated-qrels")
    merge_qrels_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    merge_qrels_parser.add_argument("--adjudications", type=Path, required=True)
    merge_qrels_parser.add_argument("--output", type=Path, default=DEFAULT_ADJUDICATED_QRELS_PATH)
    merge_qrels_parser.add_argument("--report-output", type=Path, default=DEFAULT_QRELS_AUDIT_REPORT_PATH)
    merge_qrels_parser.add_argument("--default-judge-type", choices=["human", "llm"], default="human")
    merge_qrels_parser.add_argument("--default-judge-id", default="adjudicator")
    merge_qrels_parser.add_argument("--judge-version", default="v1")

    strong_baseline_parser = subparsers.add_parser("compare-strong-baselines")
    add_common_paths(strong_baseline_parser)
    strong_baseline_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    strong_baseline_parser.add_argument("--limit", type=int, default=60)
    strong_baseline_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    strong_baseline_parser.add_argument("--top-k", type=int, default=10)
    strong_baseline_parser.add_argument("--rerank-depth", type=int, default=20)
    strong_baseline_parser.add_argument("--ranking-keys", default=",".join(DEFAULT_STRONG_BASELINE_KEYS))
    strong_baseline_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    strong_baseline_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    strong_baseline_parser.add_argument("--output", type=Path, default=DEFAULT_STRONG_BASELINE_REPORT_PATH)
    strong_baseline_parser.add_argument("--markdown-output", type=Path, default=None)
    strong_baseline_parser.add_argument("--llm-rerank-sample-size", type=int, default=0)
    strong_baseline_parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    strong_baseline_parser.add_argument("--llm-retries", type=int, default=0)
    strong_baseline_parser.add_argument("--require-llm-rerank", action="store_true")
    add_planner_args(strong_baseline_parser)

    rerank_upper_parser = subparsers.add_parser("compare-rerank-upper-bound")
    add_common_paths(rerank_upper_parser)
    rerank_upper_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    rerank_upper_parser.add_argument("--limit", type=int, default=60)
    rerank_upper_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    rerank_upper_parser.add_argument("--top-k", type=int, default=10)
    rerank_upper_parser.add_argument("--rerank-depth", type=int, default=20)
    rerank_upper_parser.add_argument("--ranking-keys", default=",".join(DEFAULT_STRONG_BASELINE_KEYS))
    rerank_upper_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    rerank_upper_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    rerank_upper_parser.add_argument("--output", type=Path, default=DEFAULT_RERANK_UPPER_BOUND_REPORT_PATH)
    rerank_upper_parser.add_argument("--markdown-output", type=Path, default=None)
    rerank_upper_parser.add_argument("--llm-rerank-sample-size", type=int, default=0)
    rerank_upper_parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    rerank_upper_parser.add_argument("--llm-retries", type=int, default=0)
    rerank_upper_parser.add_argument("--require-llm-rerank", action="store_true")
    add_planner_args(rerank_upper_parser)

    fuzzy_multi_parser = subparsers.add_parser("evaluate-fuzzy-multirelevance")
    add_common_paths(fuzzy_multi_parser)
    fuzzy_multi_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    fuzzy_multi_parser.add_argument("--limit", type=int, default=60)
    fuzzy_multi_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    fuzzy_multi_parser.add_argument("--case-type", choices=["simple_positive", "hard_positive"], default="simple_positive")
    fuzzy_multi_parser.add_argument("--top-k", type=int, default=10)
    fuzzy_multi_parser.add_argument("--pool-depth", type=int, default=20)
    fuzzy_multi_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints")
    fuzzy_multi_parser.add_argument("--pool-ranking-keys", default=",".join(DEFAULT_POOL_RANKING_KEYS))
    fuzzy_multi_parser.add_argument("--variant-types", default="")
    fuzzy_multi_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    fuzzy_multi_parser.add_argument("--qrels", type=Path, default=DEFAULT_POOLED_QRELS_PATH)
    fuzzy_multi_parser.add_argument("--output", type=Path, default=DEFAULT_FUZZY_MULTI_REPORT_PATH)
    fuzzy_multi_parser.add_argument("--markdown-output", type=Path, default=None)
    fuzzy_multi_parser.add_argument("--write-bootstrap-qrels", type=Path, default=None)
    add_planner_args(fuzzy_multi_parser)

    signature_parser = subparsers.add_parser("validate-scene-signature")
    add_common_paths(signature_parser)
    signature_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    signature_parser.add_argument("--limit", type=int, default=60)
    signature_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    signature_parser.add_argument("--top-k", type=int, default=10)
    signature_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    signature_parser.add_argument("--output", type=Path, default=DEFAULT_SCENE_SIGNATURE_REPORT_PATH)
    signature_parser.add_argument("--markdown-output", type=Path, default=None)
    add_planner_args(signature_parser)

    style_risk_parser = subparsers.add_parser("validate-style-risk-mining")
    add_common_paths(style_risk_parser)
    style_risk_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    style_risk_parser.add_argument("--limit", type=int, default=60)
    style_risk_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    style_risk_parser.add_argument("--top-k", type=int, default=10)
    style_risk_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints_signature")
    style_risk_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    style_risk_parser.add_argument("--output", type=Path, default=DEFAULT_STYLE_RISK_REPORT_PATH)
    add_planner_args(style_risk_parser)

    hard_negative_parser = subparsers.add_parser("mine-hard-negatives")
    add_common_paths(hard_negative_parser)
    hard_negative_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    hard_negative_parser.add_argument("--limit", type=int, default=60)
    hard_negative_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    hard_negative_parser.add_argument("--top-k", type=int, default=10)
    hard_negative_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="hybrid_rrf_constraints_signature")
    hard_negative_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    hard_negative_parser.add_argument("--output", type=Path, default=DEFAULT_HARD_NEGATIVE_POOL_PATH)
    add_planner_args(hard_negative_parser)

    rerank_gate_parser = subparsers.add_parser("validate-rerank-gate")
    add_common_paths(rerank_gate_parser)
    rerank_gate_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    rerank_gate_parser.add_argument("--limit", type=int, default=60)
    rerank_gate_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    rerank_gate_parser.add_argument("--top-k", type=int, default=10)
    rerank_gate_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    rerank_gate_parser.add_argument("--output", type=Path, default=DEFAULT_RERANK_GATE_REPORT_PATH)
    rerank_gate_parser.add_argument("--llm-sample-size", type=int, default=0)
    add_planner_args(rerank_gate_parser)

    compare_rerank_parser = subparsers.add_parser("compare-rerank-gates")
    add_common_paths(compare_rerank_parser)
    compare_rerank_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    compare_rerank_parser.add_argument("--limit", type=int, default=60)
    compare_rerank_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    compare_rerank_parser.add_argument("--top-k", type=int, default=10)
    compare_rerank_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    compare_rerank_parser.add_argument("--output", type=Path, default=DEFAULT_RERANK_GATE_REPORT_PATH)
    compare_rerank_parser.add_argument("--llm-sample-size", type=int, default=0)
    add_planner_args(compare_rerank_parser)

    compare_experiments_parser = subparsers.add_parser("compare-experiments")
    compare_experiments_parser.add_argument("reports", nargs="+", type=Path)
    compare_experiments_parser.add_argument("--output", type=Path, default=DEFAULT_EXPERIMENT_COMPARISON_PATH)

    guide_parser = subparsers.add_parser("retrieval-flywheel-guide")
    guide_parser.add_argument("--output", type=Path, default=None)

    capability_record_parser = subparsers.add_parser("record-capability-cycle")
    capability_record_parser.add_argument("--cycle-id", default="")
    capability_record_parser.add_argument("--label", default="")
    capability_record_parser.add_argument("--reports", nargs="*", type=Path, default=[])
    capability_record_parser.add_argument("--registry", type=Path, default=DEFAULT_CAPABILITY_REGISTRY_PATH)
    capability_record_parser.add_argument("--output", type=Path, default=DEFAULT_CAPABILITY_CYCLE_PATH)
    capability_record_parser.add_argument("--as-origin", action="store_true")

    capability_report_parser = subparsers.add_parser("generate-capability-report")
    capability_report_parser.add_argument("--registry", type=Path, default=DEFAULT_CAPABILITY_REGISTRY_PATH)
    capability_report_parser.add_argument("--output", type=Path, default=DEFAULT_CAPABILITY_REPORT_PATH)
    capability_report_parser.add_argument("--chart-dir", type=Path, default=DEFAULT_CAPABILITY_CHART_DIR)

    report_parser = subparsers.add_parser("generate-eval-report")
    report_parser.add_argument("--input", type=Path, default=DEFAULT_WORKFLOW_COMPARISON_REPORT_PATH)
    report_parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "eval_outputs" / "mock_eval_report.md")

    paraphrase_parser = subparsers.add_parser("validate-paraphrase-stress")
    add_common_paths(paraphrase_parser)
    paraphrase_parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS_PATH)
    paraphrase_parser.add_argument("--limit", type=int, default=10)
    paraphrase_parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="all")
    paraphrase_parser.add_argument(
        "--case-type",
        choices=["simple_positive", "hard_positive"],
        default="simple_positive",
    )
    paraphrase_parser.add_argument("--top-k", type=int, default=10)
    paraphrase_parser.add_argument("--ranking-key", choices=RANKING_KEYS, default="final_score")
    paraphrase_parser.add_argument("--variant-types", default="")
    paraphrase_parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)
    paraphrase_parser.add_argument("--output", type=Path, default=DEFAULT_PARAPHRASE_STRESS_REPORT_PATH)
    paraphrase_parser.add_argument("--dry-run", action="store_true")
    paraphrase_parser.add_argument("--include-debug-text", action="store_true")
    paraphrase_parser.add_argument("--include-planner-debug", action="store_true")
    add_planner_args(paraphrase_parser)

    args = parser.parse_args()
    if args.command == "build-index":
        result = build_index_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "compact-embedding-cache":
        result = compact_embedding_cache_command(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "search":
        result = search_command(args)
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        result = evaluate_command(args)
        write_json(args.output, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "tune-constraints":
        result = tune_constraints_command(args)
        write_json(args.output, result)
        print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-leave-one-fixture-out":
        result = leave_one_fixture_out_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-ranking-keys":
        result = validate_ranking_keys_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-hybrid":
        result = evaluate_hybrid_command(args)
        write_json(args.output, result)
        print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    elif args.command == "compare-ranking-workflows":
        result = compare_ranking_workflows_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "compare-query-understanding":
        result = compare_query_understanding_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-style-negatives":
        result = validate_style_negatives_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-fuzzy-understanding":
        result = validate_fuzzy_understanding_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "build-graded-qrels":
        result = build_graded_qrels_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-graded":
        result = evaluate_graded_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "build-pooled-qrels":
        result = build_pooled_qrels_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-pooled":
        result = evaluate_pooled_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "analyze-failures":
        result = analyze_failures_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "analyze-recall-bound":
        result = analyze_recall_bound_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "sample-active-qrels":
        result = sample_active_qrels_command(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "audit-qrels":
        result = audit_qrels_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "merge-adjudicated-qrels":
        result = merge_adjudicated_qrels_command(args)
        write_json(args.report_output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "compare-strong-baselines":
        result = compare_strong_baselines_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "compare-rerank-upper-bound":
        result = compare_rerank_upper_bound_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-fuzzy-multirelevance":
        result = evaluate_fuzzy_multirelevance_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-scene-signature":
        result = validate_scene_signature_command(args)
        write_json(args.output, result)
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-style-risk-mining":
        result = validate_style_risk_mining_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "mine-hard-negatives":
        result = mine_hard_negatives_command(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "validate-rerank-gate":
        result = validate_rerank_gate_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "compare-rerank-gates":
        result = compare_rerank_gates_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "compare-experiments":
        result = compare_experiments_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "retrieval-flywheel-guide":
        result = retrieval_flywheel_guide()
        if args.output is not None:
            write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "record-capability-cycle":
        result = record_capability_cycle_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "generate-capability-report":
        result = generate_capability_report_command(args)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "generate-eval-report":
        result = json.loads(args.input.read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown_report(result), encoding="utf-8")
        print(str(args.output))
    elif args.command == "validate-paraphrase-stress":
        result = validate_paraphrase_stress_command(args)
        write_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--matrix-cache", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--embedding-batch-size", type=int, default=MAX_DASHSCOPE_BATCH_SIZE)
    parser.add_argument("--cache-load-mode", choices=["lazy", "full"], default="lazy")
    parser.add_argument("--cache-format", choices=["jsonl", "matrix"], default="matrix")
    parser.add_argument("--matrix-dtype", choices=["float16", "float32"], default="float16")


def add_constraint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--constraint-profile", type=Path, default=DEFAULT_CONSTRAINT_PROFILE_PATH)


def make_embedding_cache(args: argparse.Namespace) -> EmbeddingCache:
    return EmbeddingCache(
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.embedding_batch_size,
        load_all=getattr(args, "cache_load_mode", "lazy") == "full",
        prefer_matrix=getattr(args, "cache_format", "matrix") == "matrix",
        matrix_path=getattr(args, "matrix_cache", None),
        matrix_dtype=getattr(args, "matrix_dtype", "float16"),
    )


def compact_embedding_cache_command(args: argparse.Namespace) -> dict[str, Any]:
    return build_matrix_cache(
        args.cache,
        matrix_path=getattr(args, "matrix_cache", None),
        dtype=getattr(args, "matrix_dtype", "float16"),
    )


def add_planner_args(parser: argparse.ArgumentParser, *, include_query_planner: bool = True) -> None:
    if include_query_planner:
        parser.add_argument("--query-planner", choices=VALID_QUERY_PLANNERS, default="rule")
    parser.add_argument("--planner-cache", type=Path, default=DEFAULT_QUERY_PLANNER_CACHE_PATH)
    parser.add_argument("--max-query-texts", type=int, default=0)
    parser.add_argument("--llm-planner-sample-size", type=int, default=0)
    parser.add_argument("--planner-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--planner-retries", type=int, default=0)
    parser.add_argument("--require-llm-planner", action="store_true")


def build_index_command(args: argparse.Namespace) -> dict[str, Any]:
    rows = build_item_channel_rows(load_review_items())
    texts = [channel["text"] for row in rows for channel in row["channels"] if channel.get("enabled", True)]
    cache = make_embedding_cache(args)
    cache_stats = cache.embed_texts(texts, dry_run=args.dry_run)
    result = {
        "item_count": len(rows),
        "channel_count": len(texts),
        "cache": str(args.cache),
        "cache_stats": cache_stats,
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return result
    index = {
        "model": args.model,
        "dimension": args.dimension,
        "channel_weights": DEFAULT_CHANNEL_WEIGHTS,
        "items": [
            {
                **row,
                "channels": [
                    {
                        **channel,
                        "embedding": cache.require_embedding(channel["text"]),
                    }
                    for channel in row["channels"]
                    if channel.get("enabled", True)
                ],
            }
            for row in rows
        ],
    }
    write_json(args.index, index)
    result["index"] = str(args.index)
    return result


def search_command(args: argparse.Namespace) -> dict[str, Any]:
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    profile = None if args.no_constraints else load_constraint_profile(args.constraint_profile)
    plan_result = plan_inputs_from_args([args.user_input], args=args, planner=args.query_planner)
    query_plan = plan_result.plans[0]
    query_channels = query_channels_for_plan_args(query_plan, args)
    cache.embed_texts([channel["text"] for channel in query_channels])
    prepared = prepare_mock_index(index)
    semantic_scores, channel_scores = score_query_fast(prepared, cache, query_channels)
    empty_profile = profile or profile_with_weights(
        desired_stage_bonus=0.0,
        forbidden_stage_penalty=0.0,
        negative_constraint_penalty=0.0,
    )
    signal = FastCaseSignals(
        case={"case_id": "search", "case_type": "search", "expected_relation": "unknown", "user_input": args.user_input},
        user_input=args.user_input,
        semantic_scores=semantic_scores,
        channel_scores=channel_scores,
        query_constraints=planner_constraints(query_plan) if not args.no_constraints else empty_query_constraints(),
        query_plan=query_plan,
        query_channels=query_channels,
    )
    results = rank_fast_items_for_key(
        prepared,
        signal,
        ranking_key="final_score" if not args.no_constraints else "semantic_only",
        constraint_profile=empty_profile,
        include_debug_text=False,
    )[: args.top_k]
    result = {
        "user_input": args.user_input,
        "query_planner": args.query_planner,
        "query_plan": query_plan.to_dict(),
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "top_k": args.top_k,
        "constraints_enabled": not args.no_constraints,
        "constraint_profile": str(args.constraint_profile),
        "results": results,
    }
    if getattr(args, "include_planner_debug", False):
        result["query_channels"] = query_channels
    return result


def evaluate_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = None if args.no_constraints else load_constraint_profile(args.constraint_profile)
    result = evaluate_cases(
        index_path=args.index,
        cache_path=args.cache,
        model=args.model,
        dimension=args.dimension,
        embedding_batch_size=args.embedding_batch_size,
        inputs_path=args.inputs,
        limit=args.limit,
        split=args.split,
        top_k=args.top_k,
        constraint_profile=profile,
        constraint_profile_path=args.constraint_profile,
        constraints_enabled=not args.no_constraints,
    )
    return result


def tune_constraints_command(args: argparse.Namespace) -> dict[str, Any]:
    base_profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    dev_cases = read_cases(args.inputs, args.limit, split=args.split)
    test_cases = read_cases(args.inputs, 0, split="test")
    query_texts = [
        channel["text"]
        for case in dev_cases + test_cases
        for channel in build_query_channels(case["user_input"])
    ]
    cache.embed_texts(query_texts)
    dev_precomputed = precompute_embedding_rankings(index, cache, dev_cases)
    test_precomputed = precompute_embedding_rankings(index, cache, test_cases)

    tuning = tune_profile_from_precomputed(dev_precomputed, base_profile=base_profile, top_k=args.top_k)
    best = tuning["best"]
    best_profile = profile_with_weights(
        desired_stage_bonus=best["weights"]["desired_stage_bonus"],
        forbidden_stage_penalty=best["weights"]["forbidden_stage_penalty"],
        negative_constraint_penalty=best["weights"]["negative_constraint_penalty"],
        base_profile=base_profile,
    )
    write_constraint_profile(args.constraint_profile, best_profile)
    dev_baseline = evaluate_precomputed_cases(
        precomputed=dev_precomputed,
        top_k=args.top_k,
        constraint_profile=base_profile,
        constraints_enabled=False,
        include_cases=False,
    )
    test_baseline = evaluate_precomputed_cases(
        precomputed=test_precomputed,
        top_k=args.top_k,
        constraint_profile=base_profile,
        constraints_enabled=False,
        include_cases=False,
    )
    test_after = evaluate_precomputed_cases(
        precomputed=test_precomputed,
        top_k=args.top_k,
        constraint_profile=best_profile,
        constraints_enabled=True,
        include_cases=False,
    )
    return {
        "method": "mock_constraint_tuning",
        "split": args.split,
        "dev_case_count": len(dev_cases),
        "test_case_count": len(test_cases),
        "limit": args.limit,
        "top_k": args.top_k,
        "candidate_count": len(tuning["candidates"]),
        "profile_output": str(args.constraint_profile),
        "best": {
            "weights": best["weights"],
            "selection_score": best["selection_score"],
            "metrics": best["metrics"],
        },
        "selected_weights": best["weights"],
        "dev_baseline_metrics": dev_baseline["metrics"],
        "dev_metrics": best["metrics"],
        "test_baseline_metrics": test_baseline["metrics"],
        "test_metrics_after_tuning": test_after["metrics"],
        "possible_overfit": possible_overfit(
            dev_baseline["metrics"],
            best["metrics"],
            test_baseline["metrics"],
            test_after["metrics"],
        ),
        "candidates": tuning["candidates"],
    }


def leave_one_fixture_out_command(args: argparse.Namespace) -> dict[str, Any]:
    base_profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    cases = read_cases(args.inputs, args.limit, split="all")
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)
    fixture_ids = sorted({case_fixture_id(row["case"]) for row in precomputed})

    fixture_reports = []
    for fixture_id in fixture_ids:
        dev_precomputed = filter_precomputed_by_fixture(precomputed, fixture_id, include=False)
        test_precomputed = filter_precomputed_by_fixture(precomputed, fixture_id, include=True)
        tuning = tune_profile_from_precomputed(dev_precomputed, base_profile=base_profile, top_k=args.top_k)
        best = tuning["best"]
        best_profile = profile_with_weights(
            desired_stage_bonus=best["weights"]["desired_stage_bonus"],
            forbidden_stage_penalty=best["weights"]["forbidden_stage_penalty"],
            negative_constraint_penalty=best["weights"]["negative_constraint_penalty"],
            base_profile=base_profile,
        )
        test_report = evaluate_precomputed_cases(
            precomputed=test_precomputed,
            top_k=args.top_k,
            constraint_profile=best_profile,
            constraints_enabled=True,
            include_cases=False,
        )
        fixture_reports.append(
            {
                "fixture_id": fixture_id,
                "dev_case_count": len(dev_precomputed),
                "test_case_count": len(test_precomputed),
                "selected_weights": best["weights"],
                "test_metrics": test_report["metrics"],
            }
        )

    summary = summarize_leave_one_fixture(fixture_reports)
    return {
        "method": "mock_leave_one_fixture_out",
        "top_k": args.top_k,
        "fixture_count": len(fixture_reports),
        "summary": summary,
        "fixtures": fixture_reports,
    }


def validate_ranking_keys_command(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    cases = read_cases(args.inputs, args.limit, split=args.split)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    precomputed = precompute_embedding_rankings(index, cache, cases)

    ranked_by_key = {
        key: rank_precomputed_cases(precomputed, ranking_key=key, constraint_profile=profile, top_k=args.top_k)
        for key in LEGACY_RANKING_VALIDATION_KEYS
    }
    ranking_reports = {
        key: summarize_ranking_key_results(results)
        for key, results in ranked_by_key.items()
    }
    pairwise = build_pairwise_report(precomputed, ranked_by_key, index=index, top_k=args.top_k)
    margin = build_margin_report(ranked_by_key["final_score"])
    llm_sample = (
        run_llm_sample_judge(
            ranked_by_key=ranked_by_key,
            sample_size=args.llm_sample_size,
            timeout_seconds=args.llm_timeout_seconds,
            retries=args.llm_retries,
        )
        if args.llm_sample_size > 0
        else None
    )
    summary = {
        "case_count": len(cases),
        "split": args.split,
        "final_score_expected_prefer_vs_forbidden_accuracy": pair_metric(
            pairwise,
            "expected_prefer_vs_forbidden",
            "final_score",
            "accuracy",
        ),
        "embedding_only_expected_prefer_vs_forbidden_accuracy": pair_metric(
            pairwise,
            "expected_prefer_vs_forbidden",
            "embedding_only",
            "accuracy",
        ),
        "final_score_positive_vs_random_accuracy": pair_metric(
            pairwise,
            "positive_vs_random",
            "final_score",
            "accuracy",
        ),
        "final_score_positive_vs_wrong_stage_accuracy": pair_metric(
            pairwise,
            "positive_vs_wrong_stage",
            "final_score",
            "accuracy",
        ),
        "final_score_forbidden_stage_violation_at_3": ranking_reports["final_score"]["metrics"]["overall"][
            "forbidden_stage_violation_at_3"
        ],
        "low_confidence_rate": margin["low_confidence_rate"],
    }
    if llm_sample is not None:
        summary["llm_sample_precision_at_3"] = llm_sample.get("precision_at_3")
    return {
        "method": "mock_ranking_key_validation",
        "split": args.split,
        "case_count": len(cases),
        "top_k": args.top_k,
        "ranking_keys": ranking_reports,
        "pairwise": pairwise,
        "margin": margin,
        "llm_sample_judge": llm_sample,
        "worst_cases": select_worst_cases(ranked_by_key["final_score"], limit=20),
        "summary": summary,
    }


def evaluate_hybrid_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    cases = read_cases(args.inputs, args.limit, split=args.split)
    plan_result = plan_cases_from_args(cases, args=args, planner=args.query_planner)
    query_texts = query_texts_for_plans(plan_result.plans, args)
    cache.embed_texts(query_texts)
    prepared = prepare_mock_index(index)
    signals = precompute_fast_case_signals(
        prepared,
        cache,
        cases,
        constraint_profile=profile,
        query_plans=plan_result.plans,
        max_query_texts=max_query_texts_arg(args),
    )
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key=args.ranking_key,
        constraint_profile=profile,
        top_k=args.top_k,
        include_debug_text=args.include_debug_text,
        include_planner_debug=args.include_planner_debug,
        include_all_results=False,
    )
    compact_rows = compact_case_rows(rows, include_debug_text=args.include_debug_text)
    return {
        "method": "mock_hybrid_retrieval",
        "split": args.split,
        "query_planner": args.query_planner,
        "planner_stats": plan_result.stats,
        "ranking_key": args.ranking_key,
        "embedding_cache": cache.cache_report(),
        "case_count": len(rows),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "metrics": build_metrics(compact_rows),
        "cases": compact_rows,
    }


def compare_ranking_workflows_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    cases = read_cases(args.inputs, args.limit, split=args.split)
    plan_result = plan_cases_from_args(cases, args=args, planner=args.query_planner)
    query_texts = query_texts_for_plans(plan_result.plans, args)
    cache.embed_texts(query_texts)
    prepared = prepare_mock_index(index)
    signals = precompute_fast_case_signals(
        prepared,
        cache,
        cases,
        constraint_profile=profile,
        query_plans=plan_result.plans,
        max_query_texts=max_query_texts_arg(args),
    )
    workflow_keys = [
        "semantic_only",
        "semantic_constraints",
        "lexical_only",
        "hybrid_rrf",
        "hybrid_rrf_constraints",
        "hybrid_rrf_constraints_signature",
        "adaptive_signature",
    ]
    key_map = {
        "semantic_constraints": "final_score",
    }
    ranked_by_workflow = {
        workflow: rank_fast_cases(
            prepared,
            signals,
            ranking_key=key_map.get(workflow, workflow),
            constraint_profile=profile,
            top_k=args.top_k,
            include_debug_text=args.include_debug_text,
            include_planner_debug=args.include_planner_debug,
            include_all_results=False,
        )
        for workflow in workflow_keys
    }
    reports = {
        workflow: summarize_ranking_key_results(rows)
        for workflow, rows in ranked_by_workflow.items()
    }
    baseline = reports["semantic_only"]["metrics"]
    best_workflow = max(
        reports,
        key=lambda workflow: workflow_selection_score(reports[workflow]["metrics"]),
    )
    return {
        "method": "mock_workflow_comparison",
        "split": args.split,
        "query_planner": args.query_planner,
        "planner_stats": plan_result.stats,
        "case_count": len(cases),
        "embedding_cache": cache.cache_report(),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "workflows": reports,
        "summary": {
            "best_workflow": best_workflow,
            "best_selection_score": workflow_selection_score(reports[best_workflow]["metrics"]),
            "workflow_delta_vs_baseline": {
                workflow: workflow_delta(reports[workflow]["metrics"], baseline)
                for workflow in reports
            },
        },
        "worst_cases": {
            workflow: select_worst_cases(rows, limit=10)
            for workflow, rows in ranked_by_workflow.items()
        },
    }


def validate_style_negatives_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    index_items = {item["item_id"]: item for item in index["items"]}
    cache = make_embedding_cache(args)
    source_cases = [
        case
        for case in read_cases(args.inputs, args.limit, split=args.split)
        if case["expected_relation"] == "should_match"
    ]
    generated_variants = [build_style_negative_case(case) for case in source_cases]
    skipped_cases = []
    variants = []
    for variant in generated_variants:
        target_hits = target_negative_style_hits(variant, index_items)
        if target_hits:
            skipped_cases.append(
                {
                    "case_id": variant["case_id"],
                    "target_item_id": target_item_id(variant["target"]),
                    "negative_style_hits": target_hits,
                }
            )
            continue
        variants.append(variant)
    plan_result = plan_cases_from_args(variants, args=args, planner=args.query_planner)
    query_texts = query_texts_for_plans(plan_result.plans, args)
    cache.embed_texts(query_texts)
    prepared = prepare_mock_index(index)
    signals = precompute_fast_case_signals(
        prepared,
        cache,
        variants,
        constraint_profile=profile,
        query_plans=plan_result.plans,
        max_query_texts=max_query_texts_arg(args),
    )
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key=args.ranking_key,
        constraint_profile=profile,
        top_k=args.top_k,
        include_debug_text=args.include_debug_text,
        include_planner_debug=args.include_planner_debug,
        include_all_results=False,
    )
    compact_rows = compact_case_rows(rows, include_debug_text=args.include_debug_text)
    return {
        "method": "mock_style_negative_validation",
        "split": args.split,
        "query_planner": args.query_planner,
        "planner_stats": plan_result.stats,
        "ranking_key": args.ranking_key,
        "embedding_cache": cache.cache_report(),
        "case_count": len(compact_rows),
        "generated_case_count": len(generated_variants),
        "skipped_target_style_violation_count": len(skipped_cases),
        "skipped_target_style_violation_cases": skipped_cases[:20],
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": style_negative_summary(compact_rows),
        "cases": compact_rows,
    }


def compare_query_understanding_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = read_cases(args.inputs, args.limit, split=args.split)
    planners = parse_query_planner_list(args.query_planners)
    reports: dict[str, Any] = {}
    ranked_by_planner: dict[str, list[dict[str, Any]]] = {}
    for planner in planners:
        plan_result, rows = rank_cases_with_planner(
            prepared=prepared,
            cache=cache,
            cases=cases,
            args=args,
            planner=planner,
            constraint_profile=profile,
            ranking_key=args.ranking_key,
            top_k=args.top_k,
            include_debug_text=args.include_debug_text,
            include_planner_debug=args.include_planner_debug,
        )
        compact_rows = compact_case_rows(rows, include_debug_text=args.include_debug_text)
        ranked_by_planner[planner] = rows
        reports[planner] = {
            "metrics": build_metrics(compact_rows),
            "planner_summary": query_understanding_summary(compact_rows, plan_result.stats),
            "planner_stats": plan_result.stats,
            "embedding_cache": cache.cache_report(),
            "case_count": len(compact_rows),
            "cases": compact_rows,
        }
    best_planner = max(reports, key=lambda name: query_understanding_selection_score(reports[name]["planner_summary"]))
    baseline = reports[planners[0]]["planner_summary"]
    return {
        "method": "mock_query_understanding_comparison",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "query_planners": planners,
        "case_count": len(cases),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "planners": reports,
        "summary": {
            "best_planner": best_planner,
            "best_selection_score": query_understanding_selection_score(reports[best_planner]["planner_summary"]),
            "planner_delta_vs_baseline": {
                planner: query_understanding_delta(reports[planner]["planner_summary"], baseline)
                for planner in reports
            },
        },
        "worst_cases": {
            planner: select_worst_cases(rows, limit=10)
            for planner, rows in ranked_by_planner.items()
        },
    }


def validate_fuzzy_understanding_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_cases = [
        case
        for case in read_cases(args.inputs, 0, split=args.split)
        if case["case_type"] == args.case_type and case["expected_relation"] == "should_match"
    ]
    if args.limit > 0:
        source_cases = source_cases[: args.limit]
    variants = [
        variant
        for case in source_cases
        for variant in build_fuzzy_understanding_variants(case)
    ]
    variants = filter_variants_by_type(variants, args.variant_types)
    if args.dry_run:
        return {
            "method": "mock_fuzzy_understanding",
            "dry_run": True,
            "split": args.split,
            "case_type": args.case_type,
            "source_case_count": len(source_cases),
            "variant_count": len(variants),
            "summary": {
                "source_case_count": len(source_cases),
                "variant_count": len(variants),
                "variant_types": sorted({variant["variant_type"] for variant in variants}),
                "expected_granularity": sorted({variant["expected_granularity"] for variant in variants}),
            },
            "cases": variants,
        }

    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    plan_result, ranked_rows = rank_cases_with_planner(
        prepared=prepared,
        cache=cache,
        cases=variants,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.ranking_key,
        top_k=args.top_k,
        include_debug_text=args.include_debug_text,
        include_planner_debug=args.include_planner_debug,
    )
    rows = []
    for variant, row in zip(variants, ranked_rows):
        top_results = row["top_results"]
        margin = top1_top2_margin({"top_results": top_results})
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "all_results"},
                "source_case_id": variant["source_case_id"],
                "variant_type": variant["variant_type"],
                "expected_granularity": variant["expected_granularity"],
                "top1_top2_margin": round(margin, 6),
                "confidence": confidence_bucket(margin),
            }
        )
    compact_rows = compact_case_rows(rows, include_debug_text=args.include_debug_text)
    return {
        "method": "mock_fuzzy_understanding",
        "split": args.split,
        "case_type": args.case_type,
        "query_planner": args.query_planner,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "ranking_key": args.ranking_key,
        "source_case_count": len(source_cases),
        "variant_count": len(compact_rows),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summarize_fuzzy_rows(compact_rows, plan_result.stats),
        "by_variant_type": {
            variant_type: summarize_fuzzy_rows([row for row in compact_rows if row["variant_type"] == variant_type], {})
            for variant_type in sorted({row["variant_type"] for row in compact_rows})
        },
        "cases": compact_rows,
    }


def build_graded_qrels_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    plan_result, rows = rank_cases_with_planner(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.ranking_key,
        top_k=args.top_k,
        include_debug_text=False,
        include_planner_debug=False,
    )
    qrels = bootstrap_qrels_from_rows(rows)
    write_qrels(args.qrels_output, qrels)
    metrics = graded_metrics(rows, qrels, top_k=min(args.top_k, 10))
    report = {
        "method": "mock_build_graded_qrels",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "query_planner": args.query_planner,
        "qrels_output": str(args.qrels_output),
        "case_count": len(rows),
        "qrels_count": len(qrels),
        "llm_sample_size": args.llm_sample_size,
        "llm_call_count": 0,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "graded_metrics": metrics,
        "summary": {
            **metrics,
            "qrels_count": len(qrels),
            "llm_call_count": 0,
        },
    }
    report["experiment"] = experiment_metadata(args, report["summary"], started_at)
    return report


def evaluate_graded_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    plan_result, rows = rank_cases_with_planner(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.ranking_key,
        top_k=args.top_k,
        include_debug_text=False,
        include_planner_debug=False,
    )
    qrels = load_qrels(args.qrels) if args.qrels.exists() else bootstrap_qrels_from_rows(rows)
    metrics = graded_metrics(rows, qrels, top_k=args.top_k)
    compact_rows = compact_case_rows(rows, include_debug_text=False)
    report = {
        "method": "mock_graded_evaluation",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "query_planner": args.query_planner,
        "qrels": str(args.qrels),
        "qrels_source": "file" if args.qrels.exists() else "bootstrap_in_memory",
        "case_count": len(rows),
        "qrels_count": len(qrels),
        "top_k": args.top_k,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "graded_metrics": metrics,
        "summary": metrics,
        "cases": compact_rows,
    }
    report["experiment"] = experiment_metadata(args, report["summary"], started_at)
    return report


def build_pooled_qrels_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    planners = parse_query_planner_list(args.pool_query_planners)
    ranking_keys = parse_ranking_key_list(args.pool_ranking_keys)
    run_rows, planner_stats = rank_pooled_runs(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planners=planners,
        ranking_keys=ranking_keys,
        constraint_profile=profile,
        top_k=args.pool_depth,
    )
    qrels = pooled_qrels_from_run_rows(run_rows)
    write_qrels(args.qrels_output, qrels)
    baseline_run = f"{planners[0]}::{ranking_keys[0]}"
    baseline_rows = run_rows.get(baseline_run) or next(iter(run_rows.values()), [])
    metrics = graded_metrics(baseline_rows, qrels, top_k=args.top_k)
    summary = {
        **metrics,
        **pooled_qrels_summary(qrels, cases, run_rows),
        "pool_depth": args.pool_depth,
        "run_count": len(run_rows),
        "qrels_trust_level": qrels_trust_level(qrels),
    }
    report = {
        "method": "mock_build_pooled_qrels",
        "split": args.split,
        "pool_depth": args.pool_depth,
        "top_k": args.top_k,
        "pool_ranking_keys": ranking_keys,
        "pool_query_planners": planners,
        "qrels_output": str(args.qrels_output),
        "case_count": len(cases),
        "qrels_count": len(qrels),
        "planner_stats": planner_stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "graded_metrics": metrics,
        "summary": summary,
        "run_summaries": {
            run_name: {
                "case_count": len(rows),
                "metrics": build_metrics(compact_case_rows(rows, include_debug_text=False)),
            }
            for run_name, rows in run_rows.items()
        },
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def evaluate_pooled_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    plan_result, rows = rank_cases_with_planner(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.ranking_key,
        top_k=args.top_k,
        include_debug_text=False,
        include_planner_debug=False,
    )
    qrels = load_qrels(args.qrels) if args.qrels.exists() else pooled_qrels_from_run_rows(
        {f"{args.query_planner}::{args.ranking_key}": rows}
    )
    metrics = graded_metrics(rows, qrels, top_k=args.top_k)
    compact_rows = compact_case_rows(rows, include_debug_text=False)
    summary = {
        **metrics,
        "qrels_source": "file" if args.qrels.exists() else "bootstrap_from_current_run",
        "qrels_trust_level": qrels_trust_level(qrels),
    }
    report = {
        "method": "mock_pooled_evaluation",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "query_planner": args.query_planner,
        "qrels": str(args.qrels),
        "qrels_source": summary["qrels_source"],
        "case_count": len(rows),
        "qrels_count": len(qrels),
        "top_k": args.top_k,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "graded_metrics": metrics,
        "metrics": build_metrics(compact_rows),
        "summary": summary,
        "cases": compact_rows,
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def analyze_recall_bound_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    ranking_keys = parse_ranking_key_list(args.pool_ranking_keys)
    if args.baseline_ranking_key not in ranking_keys:
        ranking_keys = [args.baseline_ranking_key, *ranking_keys]
    plan_result, rows, signals = rank_cases_with_signals(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.baseline_ranking_key,
        top_k=max(args.top_k, args.candidate_depth),
    )
    rows_by_key = {args.baseline_ranking_key: rows}
    for ranking_key in ranking_keys:
        if ranking_key == args.baseline_ranking_key:
            continue
        rows_by_key[ranking_key] = rank_fast_cases(
            prepared,
            signals,
            ranking_key=ranking_key,
            constraint_profile=profile,
            top_k=max(args.top_k, args.candidate_depth),
            include_debug_text=False,
            include_planner_debug=False,
            include_all_results=False,
        )
    case_rows = recall_bound_rows(rows_by_key, baseline_key=args.baseline_ranking_key, candidate_depth=args.candidate_depth, top_k=args.top_k)
    summary = recall_bound_summary(case_rows, top_k=args.top_k, candidate_depth=args.candidate_depth)
    report = {
        "method": "mock_recall_bound_analysis",
        "split": args.split,
        "query_planner": args.query_planner,
        "baseline_ranking_key": args.baseline_ranking_key,
        "ranking_keys": ranking_keys,
        "case_count": len(cases),
        "top_k": args.top_k,
        "candidate_depth": args.candidate_depth,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "cases": case_rows[:200],
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def sample_active_qrels_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    planners = parse_query_planner_list(args.pool_query_planners)
    ranking_keys = parse_ranking_key_list(args.pool_ranking_keys)
    run_rows, planner_stats = rank_pooled_runs(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planners=planners,
        ranking_keys=ranking_keys,
        constraint_profile=profile,
        top_k=args.pool_depth,
    )
    existing_qrels = load_qrels(args.qrels) if args.qrels.exists() else []
    samples = active_qrels_samples(
        run_rows,
        existing_qrels=existing_qrels,
        sample_size=args.sample_size,
        include_judged=args.include_judged,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    summary = {
        "sample_count": len(samples),
        "case_count": len(cases),
        "run_count": len(run_rows),
        "pool_depth": args.pool_depth,
        "existing_qrels_count": len(existing_qrels),
        "include_judged": bool(args.include_judged),
        "reason_counts": dict(Counter(reason for row in samples for reason in row.get("reasons", []))),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    return {
        "method": "mock_active_qrels_sampler",
        "split": args.split,
        "output": str(args.output),
        "summary": summary,
        "planner_stats": planner_stats,
        "examples": samples[:20],
    }


def audit_qrels_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    qrels = load_qrels(args.qrels)
    summary = qrels_audit_summary(qrels)
    report = {
        "method": "mock_qrels_audit",
        "qrels": str(args.qrels),
        "qrels_count": len(qrels),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "low_confidence_examples": [
            row for row in qrels
            if qrel_confidence(row) < 0.6
        ][:50],
    }
    report["experiment"] = {
        "command": getattr(args, "command", ""),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_sha": git_sha(),
        "elapsed_seconds": report["elapsed_seconds"],
        "summary": summary,
    }
    return report


def merge_adjudicated_qrels_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    existing_qrels = load_qrels(args.qrels)
    adjudication_votes = load_adjudication_votes(
        args.adjudications,
        default_judge_type=args.default_judge_type,
        default_judge_id=args.default_judge_id,
        judge_version=args.judge_version,
    )
    merged_qrels = merge_adjudicated_qrels(existing_qrels, adjudication_votes)
    write_qrels(args.output, merged_qrels)
    summary = {
        **qrels_audit_summary(merged_qrels),
        "input_qrels_count": len(existing_qrels),
        "adjudication_vote_count": len(adjudication_votes),
        "output": str(args.output),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    report = {
        "method": "mock_merge_adjudicated_qrels",
        "qrels": str(args.qrels),
        "adjudications": str(args.adjudications),
        "output": str(args.output),
        "elapsed_seconds": summary["elapsed_seconds"],
        "summary": summary,
        "changed_examples": [
            row for row in merged_qrels
            if any(qrel_vote_judge_type(vote, row) in {"human", "llm"} for vote in row.get("grade_votes", []))
        ][:50],
    }
    report["experiment"] = {
        "command": getattr(args, "command", ""),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_sha": git_sha(),
        "elapsed_seconds": report["elapsed_seconds"],
        "summary": summary,
    }
    return report


def compare_strong_baselines_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    ranking_keys = parse_ranking_key_list(args.ranking_keys)
    plan_result, _base_rows, signals = rank_cases_with_signals(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=ranking_keys[0],
        top_k=max(args.top_k, args.rerank_depth),
    )
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for ranking_key in ranking_keys:
        rows_by_name[ranking_key] = rank_fast_cases(
            prepared,
            signals,
            ranking_key=ranking_key,
            constraint_profile=profile,
            top_k=max(args.top_k, args.rerank_depth),
            include_debug_text=False,
            include_planner_debug=False,
            include_all_results=False,
        )
    qrels = load_qrels(args.qrels) if args.qrels.exists() else pooled_qrels_from_run_rows(rows_by_name)
    trust_level = qrels_trust_level(qrels)
    rerank_source_key = "hybrid_rrf_constraints_signature" if "hybrid_rrf_constraints_signature" in rows_by_name else ranking_keys[0]
    rows_by_name[f"rule_rerank@{args.rerank_depth}"] = [
        rerank_row_by_rule(row, rerank_depth=args.rerank_depth, top_k=args.top_k)
        for row in rows_by_name[rerank_source_key]
    ]
    rows_by_name[f"qrels_oracle_rerank@{args.rerank_depth}"] = [
        rerank_row_by_qrels(row, qrels, rerank_depth=args.rerank_depth, top_k=args.top_k)
        for row in rows_by_name[rerank_source_key]
    ]
    llm_summary = None
    llm_sample_report = None
    if args.llm_rerank_sample_size > 0:
        llm_rows, llm_summary = rerank_rows_with_llm_sample(
            rows_by_name[rerank_source_key],
            sample_size=args.llm_rerank_sample_size,
            rerank_depth=args.rerank_depth,
            top_k=args.top_k,
            timeout_seconds=args.llm_timeout_seconds,
            retries=args.llm_retries,
            require_llm=args.require_llm_rerank,
        )
        if llm_rows:
            llm_sample_report = {
                "name": f"llm_rerank@{args.rerank_depth}",
                "graded_metrics": graded_metrics(llm_rows, qrels, top_k=args.top_k),
                "metrics": build_metrics(compact_case_rows(llm_rows, include_debug_text=False)),
                "case_count": len(llm_rows),
            }
    reports = {
        name: {
            "graded_metrics": graded_metrics(rows, qrels, top_k=args.top_k),
            "metrics": build_metrics(compact_case_rows(rows, include_debug_text=False)),
            "case_count": len(rows),
        }
        for name, rows in rows_by_name.items()
    }
    best = max(reports, key=lambda name: strong_baseline_selection_score(reports[name]["graded_metrics"]))
    baseline = reports[ranking_keys[0]]["graded_metrics"]
    summary = {
        "best_baseline": best,
        "qrels_source": "file" if args.qrels.exists() else "bootstrap_from_compared_runs",
        "qrels_trust_level": trust_level,
        "baseline": ranking_keys[0],
        "baseline_nDCG@10": baseline.get("nDCG@10", 0.0),
        "best_nDCG@10": reports[best]["graded_metrics"].get("nDCG@10", 0.0),
        "best_delta_nDCG@10": round(reports[best]["graded_metrics"].get("nDCG@10", 0.0) - baseline.get("nDCG@10", 0.0), 6),
        "oracle_nDCG@10": reports[f"qrels_oracle_rerank@{args.rerank_depth}"]["graded_metrics"].get("nDCG@10", 0.0),
        "rerank_depth": args.rerank_depth,
        "case_count": len(cases),
        "llm_rerank": llm_summary,
    }
    report = {
        "method": "mock_strong_baseline_comparison",
        "split": args.split,
        "query_planner": args.query_planner,
        "ranking_keys": ranking_keys,
        "case_count": len(cases),
        "top_k": args.top_k,
        "rerank_depth": args.rerank_depth,
        "qrels": str(args.qrels),
        "qrels_count": len(qrels),
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "baselines": reports,
        "llm_rerank_sample": llm_sample_report,
        "summary": summary,
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def compare_rerank_upper_bound_command(args: argparse.Namespace) -> dict[str, Any]:
    report = compare_strong_baselines_command(args)
    report["method"] = "mock_rerank_upper_bound_comparison"
    baselines = report.get("baselines", {})
    baseline_key = report.get("summary", {}).get("baseline", "")
    rule_key = f"rule_rerank@{args.rerank_depth}"
    oracle_key = f"qrels_oracle_rerank@{args.rerank_depth}"
    baseline_metrics = baselines.get(baseline_key, {}).get("graded_metrics", {})
    rule_metrics = baselines.get(rule_key, {}).get("graded_metrics", {})
    oracle_metrics = baselines.get(oracle_key, {}).get("graded_metrics", {})
    llm_metrics = (report.get("llm_rerank_sample") or {}).get("graded_metrics", {})
    qrels = load_qrels(args.qrels) if args.qrels.exists() else []
    upper_summary = {
        "baseline": baseline_key,
        "baseline_nDCG@10": baseline_metrics.get("nDCG@10", 0.0),
        "rule_rerank_nDCG@10": rule_metrics.get("nDCG@10", 0.0),
        "oracle_rerank_nDCG@10": oracle_metrics.get("nDCG@10", 0.0),
        "llm_sample_nDCG@10": llm_metrics.get("nDCG@10"),
        "rerank_opportunity_nDCG@10": round(
            float(oracle_metrics.get("nDCG@10", 0.0)) - float(baseline_metrics.get("nDCG@10", 0.0)),
            6,
        ),
        "rule_gap_to_oracle_nDCG@10": round(
            float(oracle_metrics.get("nDCG@10", 0.0)) - float(rule_metrics.get("nDCG@10", 0.0)),
            6,
        ),
        "llm_gap_to_oracle_nDCG@10": (
            round(float(oracle_metrics.get("nDCG@10", 0.0)) - float(llm_metrics.get("nDCG@10", 0.0)), 6)
            if llm_metrics
            else None
        ),
        "rerank_diagnosis": rerank_upper_bound_diagnosis(
            baseline_metrics=baseline_metrics,
            oracle_metrics=oracle_metrics,
            llm_metrics=llm_metrics,
        ),
        "qrels_trust_level": qrels_trust_level(qrels) if qrels else "low",
        "llm_rerank": report.get("summary", {}).get("llm_rerank"),
    }
    report["summary"] = {**report.get("summary", {}), **upper_summary}
    report["upper_bound_analysis"] = upper_summary
    if isinstance(report.get("experiment"), dict):
        report["experiment"]["summary"] = report["summary"]
    return report


def rerank_upper_bound_diagnosis(
    *,
    baseline_metrics: dict[str, Any],
    oracle_metrics: dict[str, Any],
    llm_metrics: dict[str, Any],
) -> str:
    baseline_ndcg = float(baseline_metrics.get("nDCG@10", 0.0))
    oracle_ndcg = float(oracle_metrics.get("nDCG@10", 0.0))
    opportunity = oracle_ndcg - baseline_ndcg
    if opportunity < 0.08:
        return "candidate_recall_or_qrels_limit"
    if llm_metrics:
        llm_ndcg = float(llm_metrics.get("nDCG@10", 0.0))
        if oracle_ndcg - llm_ndcg <= 0.08:
            return "reranker_ready_for_gated_path"
        return "candidate_summary_or_reranker_needs_work"
    return "rerank_opportunity_high_needs_real_reranker_sample"


def evaluate_fuzzy_multirelevance_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_cases = [
        case
        for case in read_cases(args.inputs, 0, split=args.split)
        if case["case_type"] == args.case_type and case["expected_relation"] == "should_match"
    ]
    if args.limit > 0:
        source_cases = source_cases[: args.limit]
    variants = [
        variant
        for case in source_cases
        for variant in build_fuzzy_understanding_variants(case)
    ]
    variants = filter_variants_by_type(variants, args.variant_types)
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    pool_keys = parse_ranking_key_list(args.pool_ranking_keys)
    plan_result, rows, signals = rank_cases_with_signals(
        prepared=prepared,
        cache=cache,
        cases=variants,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.ranking_key,
        top_k=max(args.top_k, args.pool_depth),
    )
    rows = attach_variant_metadata(rows, variants)
    rows_by_key = {args.ranking_key: rows}
    for ranking_key in pool_keys:
        if ranking_key == args.ranking_key:
            continue
        rows_by_key[ranking_key] = attach_variant_metadata(rank_fast_cases(
            prepared,
            signals,
            ranking_key=ranking_key,
            constraint_profile=profile,
            top_k=max(args.top_k, args.pool_depth),
            include_debug_text=False,
            include_planner_debug=False,
            include_all_results=False,
        ), variants)
    qrels_source = "file" if args.qrels.exists() else "bootstrap_from_fuzzy_pool"
    qrels = load_qrels(args.qrels) if args.qrels.exists() else []
    if not qrels:
        qrels = pooled_qrels_from_run_rows(rows_by_key)
        qrels_source = "bootstrap_from_fuzzy_pool"
    elif qrels_judged_coverage(rows, qrels, top_k=args.top_k) < 0.5:
        qrels = merge_qrel_lists(qrels, pooled_qrels_from_run_rows(rows_by_key))
        qrels_source = "file_plus_bootstrap_for_unjudged_fuzzy_pool"
    trust_level = qrels_trust_level(qrels)
    if args.write_bootstrap_qrels is not None and not args.qrels.exists():
        write_qrels(args.write_bootstrap_qrels, qrels)
    eval_rows = truncate_ranked_rows(rows, args.top_k)
    compact_rows = compact_case_rows(eval_rows, include_debug_text=False)
    graded = graded_metrics(eval_rows, qrels, top_k=args.top_k)
    fuzzy_summary = summarize_fuzzy_rows(compact_rows, plan_result.stats)
    summary = {
        **graded,
        "qrels_source": qrels_source,
        "qrels_trust_level": trust_level,
        "scene_level_recall_at_10": fuzzy_summary.get("scene_level_recall_at_10", 0.0),
        "stage_level_hit_at_3": fuzzy_summary.get("stage_level_hit_at_3", 0.0),
        "purpose_level_hit_at_3": fuzzy_summary.get("purpose_level_hit_at_3", 0.0),
        "style_violation_at_3": fuzzy_summary.get("style_violation_at_3", 0.0),
        "low_confidence_rate": fuzzy_summary.get("low_confidence_rate", 0.0),
        "variant_count": len(eval_rows),
    }
    report = {
        "method": "mock_fuzzy_multirelevance_evaluation",
        "split": args.split,
        "case_type": args.case_type,
        "query_planner": args.query_planner,
        "ranking_key": args.ranking_key,
        "pool_ranking_keys": pool_keys,
        "qrels": str(args.qrels),
        "qrels_source": qrels_source,
        "qrels_trust_level": trust_level,
        "source_case_count": len(source_cases),
        "variant_count": len(eval_rows),
        "top_k": args.top_k,
        "pool_depth": args.pool_depth,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "graded_metrics": graded,
        "fuzzy_metrics": fuzzy_summary,
        "summary": summary,
        "by_variant_type": {
            variant_type: {
                **graded_metrics(
                    [row for row in eval_rows if row.get("variant_type") == variant_type],
                    qrels,
                    top_k=args.top_k,
                ),
                **summarize_fuzzy_rows(
                    [row for row in compact_rows if row.get("variant_type") == variant_type],
                    {},
                ),
            }
            for variant_type in sorted({row.get("variant_type") for row in eval_rows})
        },
        "cases": compact_rows[:200],
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def analyze_failures_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = positive_eval_cases(args)
    plan_result, rows, signals = rank_cases_with_signals(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key=args.ranking_key,
        top_k=args.top_k,
    )
    failures = [
        failure_analysis_row(prepared, signal, row, ranking_key=args.ranking_key, constraint_profile=profile)
        for signal, row in zip(signals, rows)
        if row.get("target_rank") is None or row.get("target_rank", 999999) > args.top_k
    ]
    counts = Counter(row["failure_type"] for row in failures)
    summary = {
        "case_count": len(rows),
        "failure_count": len(failures),
        "failure_rate": round(len(failures) / max(1, len(rows)), 6),
        "failure_type_counts": dict(sorted(counts.items())),
        "top_failure_type": counts.most_common(1)[0][0] if counts else None,
    }
    report = {
        "method": "mock_failure_analysis",
        "split": args.split,
        "ranking_key": args.ranking_key,
        "query_planner": args.query_planner,
        "case_count": len(rows),
        "top_k": args.top_k,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "failures": failures[:100],
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def validate_scene_signature_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = read_cases(args.inputs, args.limit, split=args.split)
    workflows = ["hybrid_rrf_constraints", "hybrid_rrf_constraints_signature", "adaptive_signature"]
    reports = {}
    rows_by_workflow = {}
    for workflow in workflows:
        plan_result, rows = rank_cases_with_planner(
            prepared=prepared,
            cache=cache,
            cases=cases,
            args=args,
            planner=args.query_planner,
            constraint_profile=profile,
            ranking_key=workflow,
            top_k=args.top_k,
            include_debug_text=False,
            include_planner_debug=False,
        )
        compact_rows = compact_case_rows(rows, include_debug_text=False)
        rows_by_workflow[workflow] = rows
        reports[workflow] = {
            "metrics": build_metrics(compact_rows),
            "summary": query_understanding_summary(compact_rows, plan_result.stats),
            "case_count": len(compact_rows),
        }
    baseline = reports["hybrid_rrf_constraints"]["summary"]
    best = max(reports, key=lambda key: query_understanding_selection_score(reports[key]["summary"]))
    summary = {
        "best_workflow": best,
        "workflow_delta_vs_baseline": {
            workflow: query_understanding_delta(reports[workflow]["summary"], baseline)
            for workflow in workflows
        },
    }
    report = {
        "method": "mock_scene_signature_validation",
        "split": args.split,
        "query_planner": args.query_planner,
        "case_count": len(cases),
        "top_k": args.top_k,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "workflows": reports,
        "summary": summary,
        "worst_cases": {workflow: select_worst_cases(rows, limit=10) for workflow, rows in rows_by_workflow.items()},
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def validate_style_risk_mining_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    source_cases = [
        case for case in read_cases(args.inputs, args.limit, split=args.split)
        if case["expected_relation"] == "should_match"
    ]
    variants = [build_style_risk_case(case) for case in source_cases]
    semantic_status = "available"
    semantic_error: dict[str, Any] | None = None
    effective_ranking_key = args.ranking_key
    try:
        plan_result, rows = rank_cases_with_planner(
            prepared=prepared,
            cache=cache,
            cases=variants,
            args=args,
            planner=args.query_planner,
            constraint_profile=profile,
            ranking_key=args.ranking_key,
            top_k=args.top_k,
            include_debug_text=False,
            include_planner_debug=False,
        )
    except RuntimeError as exc:
        if not is_embedding_setup_error(exc):
            raise
        semantic_status = "fallback_no_embedding"
        semantic_error = embedding_setup_error_payload(exc)
        effective_ranking_key = style_risk_fallback_ranking_key(args.ranking_key)
        plan_result = plan_cases_from_args(variants, args=args, planner=args.query_planner)
        signals = precompute_nonsemantic_case_signals(
            prepared,
            variants,
            query_plans=plan_result.plans,
            max_query_texts=max_query_texts_arg(args),
        )
        rows = rank_fast_cases(
            prepared,
            signals,
            ranking_key=effective_ranking_key,
            constraint_profile=profile,
            top_k=args.top_k,
            include_debug_text=False,
            include_planner_debug=False,
            include_all_results=False,
        )
    compact_rows = compact_case_rows(rows, include_debug_text=False)
    summary = {
        **style_negative_summary(compact_rows),
        "style_risk_violation_at_3": style_violation_at(compact_rows, 3),
        "risk_terms": sorted(set(STYLE_ALIASES) - {"human_warmth", "documentary", "real_location"}),
    }
    report = {
        "method": "mock_style_risk_mining",
        "split": args.split,
        "query_planner": args.query_planner,
        "ranking_key": effective_ranking_key,
        "ranking_key_requested": args.ranking_key,
        "semantic_status": semantic_status,
        "semantic_error": semantic_error,
        "source_case_count": len(source_cases),
        "case_count": len(compact_rows),
        "top_k": args.top_k,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "cases": compact_rows,
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def mine_hard_negatives_command(args: argparse.Namespace) -> dict[str, Any]:
    report = validate_style_risk_mining_command(args)
    mined = []
    for row in report.get("cases", []):
        for result in row.get("top_results", [])[: args.top_k]:
            if result.get("constraint_hits", {}).get("negative_style"):
                mined.append(
                    {
                        "case_id": row["case_id"],
                        "user_input": row["user_input"],
                        "bad_item_id": result["item_id"],
                        "negative_style": result["constraint_hits"]["negative_style"],
                        "score": result["score"],
                        "source": "style_risk_mining",
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mined),
        encoding="utf-8",
    )
    return {
        "method": "mock_hard_negative_mining",
        "output": str(args.output),
        "summary": {
            "mined_count": len(mined),
            "source_case_count": report["source_case_count"],
            "style_risk_violation_at_3": report["summary"].get("style_risk_violation_at_3", 0.0),
        },
        "examples": mined[:20],
    }


def validate_rerank_gate_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    prepared = prepare_mock_index(index)
    cases = read_cases(args.inputs, args.limit, split=args.split)
    plan_result, rows = rank_cases_with_planner(
        prepared=prepared,
        cache=cache,
        cases=cases,
        args=args,
        planner=args.query_planner,
        constraint_profile=profile,
        ranking_key="adaptive_signature",
        top_k=args.top_k,
        include_debug_text=False,
        include_planner_debug=False,
    )
    gated = [rerank_gate_decision(row) for row in rows]
    selected = [row for row in gated if row["should_rerank"]]
    summary = {
        "case_count": len(rows),
        "selected_count": len(selected),
        "selected_rate": round(len(selected) / max(1, len(rows)), 6),
        "llm_sample_size": args.llm_sample_size,
        "llm_call_count": 0,
        "gate_reason_counts": dict(Counter(reason for row in selected for reason in row["gate_reasons"])),
    }
    report = {
        "method": "mock_rerank_gate_validation",
        "split": args.split,
        "query_planner": args.query_planner,
        "ranking_key": "adaptive_signature",
        "top_k": args.top_k,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
        "gated_cases": gated[:100],
    }
    report["experiment"] = experiment_metadata(args, summary, started_at)
    return report


def compare_rerank_gates_command(args: argparse.Namespace) -> dict[str, Any]:
    report = validate_rerank_gate_command(args)
    rows = report.get("gated_cases", [])
    strict = [row for row in rows if row["should_rerank"] and len(row["gate_reasons"]) >= 2]
    summary = {
        "default_selected_rate": report["summary"]["selected_rate"],
        "strict_selected_rate": round(len(strict) / max(1, report["summary"]["case_count"]), 6),
        "default_selected_count": report["summary"]["selected_count"],
        "strict_selected_count": len(strict),
    }
    return {
        "method": "mock_rerank_gate_comparison",
        "split": args.split,
        "summary": summary,
        "default": report["summary"],
        "strict_examples": strict[:20],
    }


def compare_experiments_command(args: argparse.Namespace) -> dict[str, Any]:
    reports = []
    for path in args.reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append(
            {
                "path": str(path),
                "method": data.get("method"),
                "summary": data.get("summary", data.get("metrics", {})),
                "experiment": data.get("experiment", {}),
            }
        )
    return {
        "method": "mock_experiment_comparison",
        "report_count": len(reports),
        "reports": reports,
        "summary": {
            "report_count": len(reports),
            "methods": [row["method"] for row in reports],
        },
    }


def retrieval_flywheel_guide() -> dict[str, Any]:
    steps = [
        {
            "step": 1,
            "name": "build-pooled-qrels",
            "purpose": "Build a pooled bootstrap relevance set from multiple planners/workflows.",
            "command": "python -m mocktesting.mock_retriever build-pooled-qrels --split test --limit 60 --qrels-output .tmp\\pooled_qrels_next.jsonl",
        },
        {
            "step": 2,
            "name": "audit-qrels",
            "purpose": "Check bootstrap-only coverage, low confidence rows, conflicts, and trust level.",
            "command": "python -m mocktesting.mock_retriever audit-qrels --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\qrels_audit_next.json --markdown-output .tmp\\qrels_audit_next.md",
        },
        {
            "step": 3,
            "name": "sample-active-qrels",
            "purpose": "Sample low-confidence, disagreement, miss, and style-risk rows for human/LLM review.",
            "command": "python -m mocktesting.mock_retriever sample-active-qrels --split test --limit 60 --sample-size 80 --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\active_qrels_next.jsonl",
        },
        {
            "step": 4,
            "name": "merge-adjudicated-qrels",
            "purpose": "Merge reviewed votes back into qrels without dropping conflicting bootstrap votes.",
            "command": "python -m mocktesting.mock_retriever merge-adjudicated-qrels --qrels .tmp\\pooled_qrels_next.jsonl --adjudications .tmp\\active_qrels_reviewed.jsonl --output .tmp\\pooled_qrels_adjudicated.jsonl",
        },
        {
            "step": 5,
            "name": "evaluate-fuzzy-multirelevance",
            "purpose": "Evaluate fuzzy variants with graded relevance instead of exact target only.",
            "command": "python -m mocktesting.mock_retriever evaluate-fuzzy-multirelevance --split test --limit 60 --query-planner multi_query --ranking-key hybrid_rrf_constraints --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\fuzzy_multi_next.json --markdown-output .tmp\\fuzzy_multi_next.md",
        },
        {
            "step": 6,
            "name": "compare-strong-baselines",
            "purpose": "Compare baseline, signature/adaptive, rule rerank, oracle rerank, and optional LLM sample rerank.",
            "command": "python -m mocktesting.mock_retriever compare-strong-baselines --split test --limit 60 --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\strong_baselines_next.json --markdown-output .tmp\\strong_baselines_next.md",
        },
        {
            "step": 7,
            "name": "compare-rerank-upper-bound",
            "purpose": "Decide whether the next bottleneck is recall, rerank quality, or candidate summaries.",
            "command": "python -m mocktesting.mock_retriever compare-rerank-upper-bound --split test --limit 60 --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\rerank_upper_bound_next.json --markdown-output .tmp\\rerank_upper_bound_next.md",
        },
        {
            "step": 8,
            "name": "analyze-failures",
            "purpose": "Attribute miss cases to query understanding, recall, fusion/ranking, constraints, or weak labels.",
            "command": "python -m mocktesting.mock_retriever analyze-failures --split test --limit 60 --output .tmp\\failure_analysis_next.json --markdown-output .tmp\\failure_analysis_next.md",
        },
        {
            "step": 9,
            "name": "record-capability-cycle",
            "purpose": "Record this retrieval cycle as a capability baseline with deltas for the next loop.",
            "command": "python -m mocktesting.mock_retriever record-capability-cycle --cycle-id origin --as-origin --reports .tmp\\qrels_audit_next.json .tmp\\fuzzy_multi_next_fixed.json .tmp\\rerank_upper_bound_next.json .tmp\\pooled_qrels_next_report.json",
        },
        {
            "step": 10,
            "name": "generate-capability-report",
            "purpose": "Generate a Markdown + SVG capability trend report from recorded cycles.",
            "command": "python -m mocktesting.mock_retriever generate-capability-report --registry .tmp\\capability_cycles.jsonl --output .tmp\\capability_report.md",
        },
    ]
    return {
        "method": "mock_retrieval_flywheel_guide",
        "steps": steps,
        "summary": {
            "step_count": len(steps),
            "default_llm_usage": "off",
            "recommended_first_loop": (
                "build-pooled-qrels -> audit-qrels -> sample-active-qrels -> "
                "evaluate-fuzzy-multirelevance -> compare-rerank-upper-bound -> record-capability-cycle"
            ),
        },
    }


def record_capability_cycle_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    reports, missing_reports = load_capability_reports(args.reports or default_capability_report_paths())
    previous_cycles = load_capability_cycles(args.registry)
    previous_cycle = None if args.as_origin or not previous_cycles else previous_cycles[-1]
    raw_metrics = extract_capability_raw_metrics(reports, missing_reports=missing_reports)
    capabilities = compute_capability_scores(raw_metrics)
    delta = capability_delta(capabilities, previous_cycle.get("capabilities", {}) if previous_cycle else None)
    recommendations = capability_recommendations(raw_metrics, capabilities)
    cycle_id = args.cycle_id or default_capability_cycle_id()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary = capability_cycle_summary(
        cycle_id=cycle_id,
        label=args.label,
        is_origin=bool(args.as_origin),
        capabilities=capabilities,
        raw_metrics=raw_metrics,
        delta=delta,
        previous_cycle=previous_cycle,
        recommendations=recommendations,
    )
    cycle = {
        "cycle_id": cycle_id,
        "label": args.label,
        "created_at": created_at,
        "git_sha": git_sha(),
        "is_origin": bool(args.as_origin),
        "input_reports": [
            {
                "path": row["path"],
                "method": row.get("method", ""),
                "exists": True,
                "elapsed_seconds": row.get("elapsed_seconds"),
            }
            for row in reports
        ]
        + [{"path": str(path), "exists": False} for path in missing_reports],
        "raw_metrics": raw_metrics,
        "capabilities": capabilities,
        "delta_vs_previous": delta,
        "summary": summary,
        "recommendations": recommendations,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    append_capability_cycle(args.registry, cycle)
    return cycle


def generate_capability_report_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    cycles = load_capability_cycles(args.registry)
    args.chart_dir.mkdir(parents=True, exist_ok=True)
    charts = generate_capability_charts(cycles, args.chart_dir)
    markdown = capability_report_markdown(cycles, charts, output_path=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    latest = cycles[-1] if cycles else {}
    summary = {
        "cycle_count": len(cycles),
        "latest_cycle_id": latest.get("cycle_id"),
        "latest_overall_score": latest.get("summary", {}).get("overall_score"),
        "output": str(args.output),
        "chart_count": len(charts),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
    return {
        "method": "mock_capability_report",
        "registry": str(args.registry),
        "output": str(args.output),
        "charts": {key: str(value) for key, value in charts.items()},
        "summary": summary,
    }


def default_capability_report_paths() -> list[Path]:
    return [
        Path(".tmp") / "qrels_audit_next.json",
        Path(".tmp") / "fuzzy_multi_next_fixed.json",
        Path(".tmp") / "rerank_upper_bound_next.json",
        Path(".tmp") / "pooled_qrels_next_report.json",
    ]


def default_capability_cycle_id() -> str:
    return f"origin_{time.strftime('%Y%m%d_%H%M%S')}"


def load_capability_reports(paths: list[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    reports = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        reports.append(
            {
                "path": str(path),
                "method": data.get("method", ""),
                "summary": data.get("summary", {}),
                "graded_metrics": data.get("graded_metrics", {}),
                "metrics": data.get("metrics", {}),
                "experiment": data.get("experiment", {}),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "data": data,
            }
        )
    return reports, missing


def extract_capability_raw_metrics(
    reports: list[dict[str, Any]],
    *,
    missing_reports: list[Path],
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "report_count": len(reports),
        "missing_report_count": len(missing_reports),
        "missing_reports": [str(path) for path in missing_reports],
    }
    sources: dict[str, dict[str, Any]] = {}
    priorities: dict[str, int] = {}
    elapsed_seconds = 0.0
    for report in reports:
        method = str(report.get("method", ""))
        elapsed_seconds += float(report.get("elapsed_seconds") or 0.0)
        containers = [report.get("summary", {}), report.get("graded_metrics", {})]
        for container in containers:
            if not isinstance(container, dict):
                continue
            for raw_key, metric_key in CAPABILITY_METRIC_ALIASES.items():
                if raw_key in container:
                    set_capability_metric(
                        values,
                        sources,
                        priorities,
                        metric_key,
                        container[raw_key],
                        source={"method": method, "path": report["path"], "raw_key": raw_key},
                        priority=capability_metric_priority(method, metric_key),
                    )
    values["elapsed_seconds_total"] = round(elapsed_seconds, 6)
    values["metric_sources"] = sources
    return values


CAPABILITY_METRIC_ALIASES = {
    "nDCG@10": "ndcg_at_10",
    "MRR@10": "mrr_at_10",
    "scene_level_recall_at_10": "scene_level_recall_at_10",
    "stage_level_hit_at_3": "stage_level_hit_at_3",
    "purpose_level_hit_at_3": "purpose_level_hit_at_3",
    "style_violation_at_3": "style_violation_at_3",
    "rerank_opportunity_nDCG@10": "rerank_opportunity_ndcg_at_10",
    "oracle_rerank_nDCG@10": "oracle_rerank_ndcg_at_10",
    "baseline_nDCG@10": "baseline_ndcg_at_10",
    "qrels_trust_level": "qrels_trust_level",
    "manual_or_llm_count": "manual_or_llm_count",
    "manual_count": "manual_count",
    "llm_count": "llm_count",
    "bootstrap_only_count": "bootstrap_only_count",
    "needs_adjudication_count": "needs_adjudication_count",
    "vote_conflict_rate": "vote_conflict_rate",
    "qrels_count": "qrels_count",
    "sample_count": "active_sample_count",
    "llm_call_count": "llm_call_count",
}


def capability_metric_priority(method: str, metric_key: str) -> int:
    if metric_key.startswith("qrels") or metric_key in {
        "manual_or_llm_count",
        "manual_count",
        "llm_count",
        "bootstrap_only_count",
        "needs_adjudication_count",
        "vote_conflict_rate",
    }:
        return 40 if method == "mock_qrels_audit" else 20
    if metric_key.startswith("rerank") or metric_key in {"oracle_rerank_ndcg_at_10", "baseline_ndcg_at_10"}:
        return 40 if method == "mock_rerank_upper_bound_comparison" else 20
    if metric_key in {
        "ndcg_at_10",
        "mrr_at_10",
        "scene_level_recall_at_10",
        "stage_level_hit_at_3",
        "purpose_level_hit_at_3",
        "style_violation_at_3",
    }:
        return 40 if method == "mock_fuzzy_multirelevance_evaluation" else 20
    return 10


def set_capability_metric(
    values: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    priorities: dict[str, int],
    key: str,
    value: Any,
    *,
    source: dict[str, Any],
    priority: int,
) -> None:
    if key in priorities and priorities[key] > priority:
        return
    if key in priorities and priorities[key] == priority and values.get(key) not in (None, "", 0, 0.0):
        return
    values[key] = normalize_capability_metric_value(value)
    priorities[key] = priority
    sources[key] = source


def normalize_capability_metric_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return value
    return value


def compute_capability_scores(raw_metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    capabilities = {
        "retrieval_quality": capability_row(
            score=weighted_score(
                [
                    (scale_metric(raw_metrics.get("ndcg_at_10"), target=0.75), 0.45),
                    (scale_metric(raw_metrics.get("mrr_at_10"), target=0.9), 0.25),
                    (scale_metric(raw_metrics.get("scene_level_recall_at_10"), target=0.7), 0.30),
                ]
            ),
            kind="capability",
            inputs={
                "ndcg_at_10": raw_metrics.get("ndcg_at_10"),
                "mrr_at_10": raw_metrics.get("mrr_at_10"),
                "scene_level_recall_at_10": raw_metrics.get("scene_level_recall_at_10"),
            },
        ),
        "fuzzy_understanding": capability_row(
            score=weighted_score(
                [
                    (scale_metric(raw_metrics.get("stage_level_hit_at_3"), target=0.95), 0.35),
                    (scale_metric(raw_metrics.get("purpose_level_hit_at_3"), target=0.9), 0.35),
                    (scale_metric(raw_metrics.get("scene_level_recall_at_10"), target=0.7), 0.30),
                ]
            ),
            kind="capability",
            inputs={
                "stage_level_hit_at_3": raw_metrics.get("stage_level_hit_at_3"),
                "purpose_level_hit_at_3": raw_metrics.get("purpose_level_hit_at_3"),
                "scene_level_recall_at_10": raw_metrics.get("scene_level_recall_at_10"),
            },
        ),
        "style_safety": capability_row(
            score=style_safety_score(raw_metrics.get("style_violation_at_3")),
            kind="capability",
            inputs={"style_violation_at_3": raw_metrics.get("style_violation_at_3")},
        ),
        "qrels_trust": capability_row(
            score=qrels_trust_score(raw_metrics),
            kind="capability",
            inputs={
                "qrels_trust_level": raw_metrics.get("qrels_trust_level"),
                "manual_or_llm_count": raw_metrics.get("manual_or_llm_count"),
                "bootstrap_only_count": raw_metrics.get("bootstrap_only_count"),
                "needs_adjudication_count": raw_metrics.get("needs_adjudication_count"),
                "vote_conflict_rate": raw_metrics.get("vote_conflict_rate"),
            },
        ),
        "cycle_operability": capability_row(
            score=cycle_operability_score(raw_metrics),
            kind="capability",
            inputs={
                "report_count": raw_metrics.get("report_count"),
                "missing_report_count": raw_metrics.get("missing_report_count"),
                "active_sample_count": raw_metrics.get("active_sample_count"),
                "llm_call_count": raw_metrics.get("llm_call_count"),
                "elapsed_seconds_total": raw_metrics.get("elapsed_seconds_total"),
            },
        ),
        "rerank_potential": capability_row(
            score=scale_metric(raw_metrics.get("rerank_opportunity_ndcg_at_10"), target=0.30),
            kind="opportunity",
            inputs={"rerank_opportunity_ndcg_at_10": raw_metrics.get("rerank_opportunity_ndcg_at_10")},
        ),
    }
    return capabilities


def capability_row(*, score: float, kind: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"score": round(clamp(score, 0.0, 100.0), 3), "kind": kind, "inputs": inputs}


def scale_metric(value: Any, *, target: float) -> float:
    numeric = optional_float(value)
    if numeric is None or target <= 0:
        return 0.0
    return clamp(numeric / target * 100, 0.0, 100.0)


def style_safety_score(value: Any) -> float:
    violation = optional_float(value)
    if violation is None:
        return 50.0
    if violation <= 0.05:
        return 100.0
    if violation >= 0.20:
        return 0.0
    return clamp((0.20 - violation) / 0.15 * 100, 0.0, 100.0)


def qrels_trust_score(raw_metrics: dict[str, Any]) -> float:
    level = str(raw_metrics.get("qrels_trust_level", "low"))
    base = {"low": 25.0, "medium": 60.0, "high": 90.0}.get(level, 25.0)
    qrels_count = optional_float(raw_metrics.get("qrels_count")) or 0.0
    reviewed = optional_float(raw_metrics.get("manual_or_llm_count")) or 0.0
    needs = optional_float(raw_metrics.get("needs_adjudication_count")) or 0.0
    conflict = optional_float(raw_metrics.get("vote_conflict_rate")) or 0.0
    reviewed_bonus = min(10.0, reviewed / max(1.0, qrels_count) * 25.0)
    needs_penalty = min(15.0, needs / max(1.0, qrels_count) * 20.0)
    conflict_penalty = min(10.0, conflict * 30.0)
    return clamp(base + reviewed_bonus - needs_penalty - conflict_penalty, 0.0, 100.0)


def cycle_operability_score(raw_metrics: dict[str, Any]) -> float:
    report_count = optional_float(raw_metrics.get("report_count")) or 0.0
    missing_count = optional_float(raw_metrics.get("missing_report_count")) or 0.0
    active_sample_count = optional_float(raw_metrics.get("active_sample_count"))
    llm_call_count = optional_float(raw_metrics.get("llm_call_count")) or 0.0
    elapsed = optional_float(raw_metrics.get("elapsed_seconds_total")) or 0.0
    report_score = clamp(report_count / max(1.0, report_count + missing_count) * 100, 0.0, 100.0)
    active_score = 100.0 if active_sample_count and active_sample_count > 0 else 60.0
    llm_score = 100.0 if llm_call_count == 0 else 80.0
    elapsed_score = 100.0 if elapsed > 0 else 60.0
    return weighted_score([(report_score, 0.45), (active_score, 0.20), (llm_score, 0.20), (elapsed_score, 0.15)])


def weighted_score(parts: list[tuple[float, float]]) -> float:
    valid = [(score, weight) for score, weight in parts if score is not None]
    total_weight = sum(weight for _score, weight in valid)
    if total_weight <= 0:
        return 0.0
    return sum(score * weight for score, weight in valid) / total_weight


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def capability_delta(
    capabilities: dict[str, dict[str, Any]],
    previous_capabilities: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    delta = {}
    for name, row in capabilities.items():
        current = float(row.get("score", 0.0))
        previous = None
        if previous_capabilities and name in previous_capabilities:
            previous = float(previous_capabilities[name].get("score", 0.0))
        delta[name] = {
            "current_score": round(current, 3),
            "previous_score": round(previous, 3) if previous is not None else None,
            "score_delta": round(current - previous, 3) if previous is not None else 0.0,
        }
    return delta


def capability_cycle_summary(
    *,
    cycle_id: str,
    label: str,
    is_origin: bool,
    capabilities: dict[str, dict[str, Any]],
    raw_metrics: dict[str, Any],
    delta: dict[str, dict[str, Any]],
    previous_cycle: dict[str, Any] | None,
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    score_rows = [row["score"] for row in capabilities.values() if row.get("kind") == "capability"]
    improved = [name for name, row in delta.items() if float(row.get("score_delta", 0.0)) > 0.01]
    regressed = [name for name, row in delta.items() if float(row.get("score_delta", 0.0)) < -0.01]
    return {
        "cycle_id": cycle_id,
        "label": label,
        "is_origin": is_origin,
        "previous_cycle_id": previous_cycle.get("cycle_id") if previous_cycle else None,
        "overall_score": round(mean([float(value) for value in score_rows]), 3),
        "qrels_trust_level": raw_metrics.get("qrels_trust_level", "low"),
        "rerank_opportunity_nDCG@10": raw_metrics.get("rerank_opportunity_ndcg_at_10"),
        "report_count": raw_metrics.get("report_count", 0),
        "missing_report_count": raw_metrics.get("missing_report_count", 0),
        "improved_capabilities": improved,
        "regressed_capabilities": regressed,
        "top_recommendation": recommendations[0]["title"] if recommendations else "",
    }


def capability_recommendations(raw_metrics: dict[str, Any], capabilities: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    if str(raw_metrics.get("qrels_trust_level", "low")) == "low":
        recommendations.append(
            {
                "priority": 1,
                "title": "Improve qrels trust",
                "reason": "qrels_trust_level is low, so capability deltas are still bootstrap-guided.",
                "command": "python -m mocktesting.mock_retriever sample-active-qrels --split test --limit 60 --sample-size 80 --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\active_qrels_next.jsonl",
            }
        )
    if (optional_float(raw_metrics.get("rerank_opportunity_ndcg_at_10")) or 0.0) >= 0.15:
        recommendations.append(
            {
                "priority": 2,
                "title": "Run real reranker sample",
                "reason": "oracle rerank has a large nDCG@10 opportunity.",
                "command": "python -m mocktesting.mock_retriever compare-rerank-upper-bound --split test --limit 60 --qrels .tmp\\pooled_qrels_next.jsonl --llm-rerank-sample-size 10",
            }
        )
    if (optional_float(raw_metrics.get("style_violation_at_3")) or 0.0) > 0.05:
        recommendations.append(
            {
                "priority": 3,
                "title": "Tighten style negative handling",
                "reason": "style_violation_at_3 is above the first-stage target.",
                "command": "python -m mocktesting.mock_retriever validate-style-risk-mining --split test --limit 60",
            }
        )
    scene_recall = optional_float(raw_metrics.get("scene_level_recall_at_10")) or 0.0
    oracle = optional_float(raw_metrics.get("oracle_rerank_ndcg_at_10")) or 0.0
    if scene_recall < 0.70 and oracle < 0.60:
        recommendations.append(
            {
                "priority": 4,
                "title": "Revisit recall and query understanding",
                "reason": "scene recall is low and oracle rerank is not high enough to fix it alone.",
                "command": "python -m mocktesting.mock_retriever compare-query-understanding --split test --limit 60",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": 9,
                "title": "Continue the measured flywheel",
                "reason": "No critical regression or obvious bottleneck was detected.",
                "command": "python -m mocktesting.mock_retriever retrieval-flywheel-guide",
            }
        )
    return sorted(recommendations, key=lambda row: int(row["priority"]))


def load_capability_cycles(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cycles = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        cycles.append(json.loads(line))
    return cycles


def append_capability_cycle(path: Path, cycle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(cycle, ensure_ascii=False, sort_keys=True) + "\n")


def generate_capability_charts(cycles: list[dict[str, Any]], chart_dir: Path) -> dict[str, Path]:
    if not cycles:
        return {}
    latest = cycles[-1]
    charts = {
        "capability_bar": chart_dir / "capability_bar_latest.svg",
        "delta_bar": chart_dir / "capability_delta_latest.svg",
        "trend_line": chart_dir / "capability_trend.svg",
        "qrels_trust": chart_dir / "qrels_trust_stack.svg",
    }
    charts["capability_bar"].write_text(capability_bar_svg(latest), encoding="utf-8")
    charts["delta_bar"].write_text(capability_delta_svg(latest), encoding="utf-8")
    charts["trend_line"].write_text(capability_trend_svg(cycles), encoding="utf-8")
    charts["qrels_trust"].write_text(qrels_trust_stack_svg(latest), encoding="utf-8")
    return charts


def capability_report_markdown(cycles: list[dict[str, Any]], charts: dict[str, Path], *, output_path: Path) -> str:
    if not cycles:
        return "# Retrieval Capability Report\n\nNo capability cycles recorded yet.\n"
    latest = cycles[-1]
    previous = cycles[-2] if len(cycles) >= 2 else None
    origin = next((cycle for cycle in cycles if cycle.get("is_origin")), cycles[0])
    lines = [
        "# Retrieval Capability Report",
        "",
        "## Cycle Summary",
        "",
        f"- latest_cycle: `{latest.get('cycle_id')}`",
        f"- previous_cycle: `{previous.get('cycle_id') if previous else ''}`",
        f"- origin_cycle: `{origin.get('cycle_id')}`",
        f"- git_sha: `{latest.get('git_sha', '')}`",
        f"- input_reports: `{latest.get('summary', {}).get('report_count', 0)}`",
        f"- overall_score: `{latest.get('summary', {}).get('overall_score', 0.0)}`",
        "",
    ]
    if charts:
        lines.extend(
            [
                "## Charts",
                "",
                f"![Capability scores]({markdown_chart_path(charts.get('capability_bar'), output_path)})",
                "",
                f"![Capability deltas]({markdown_chart_path(charts.get('delta_bar'), output_path)})",
                "",
                f"![Capability trend]({markdown_chart_path(charts.get('trend_line'), output_path)})",
                "",
                f"![Qrels trust]({markdown_chart_path(charts.get('qrels_trust'), output_path)})",
                "",
            ]
        )
    lines.extend(["## Capability Scoreboard", "", "| capability | score | delta | kind |", "|---|---:|---:|---|"])
    for name, row in latest.get("capabilities", {}).items():
        delta = latest.get("delta_vs_previous", {}).get(name, {}).get("score_delta", 0.0)
        lines.append(f"| {name} | {row.get('score', '')} | {signed_delta(delta)} | {row.get('kind', '')} |")
    lines.append("")
    lines.extend(["## Raw Metrics", "", "| metric | value |", "|---|---:|"])
    for key, value in latest.get("raw_metrics", {}).items():
        if key in {"metric_sources", "missing_reports"} or isinstance(value, (dict, list)):
            continue
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.extend(["## Bottleneck Analysis", ""])
    for recommendation in latest.get("recommendations", []):
        lines.append(f"- **{recommendation['title']}**: {recommendation['reason']}")
    lines.append("")
    lines.extend(["## Next Cycle Recommendation", ""])
    for recommendation in latest.get("recommendations", []):
        lines.append(f"- `{recommendation['command']}`")
    lines.append("")
    return "\n".join(lines)


def markdown_chart_path(path: Path | None, output_path: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(output_path.parent).as_posix()
    except ValueError:
        return path.as_posix()


def capability_bar_svg(cycle: dict[str, Any]) -> str:
    rows = [(name, float(row.get("score", 0.0)), row.get("kind", "capability")) for name, row in cycle.get("capabilities", {}).items()]
    return bar_svg(rows, title="Capability Scores", value_min=0.0, value_max=100.0)


def capability_delta_svg(cycle: dict[str, Any]) -> str:
    rows = [
        (name, float(row.get("score_delta", 0.0)), "delta")
        for name, row in cycle.get("delta_vs_previous", {}).items()
    ]
    return diverging_bar_svg(rows, title="Delta vs Previous Cycle", value_min=-100.0, value_max=100.0)


def capability_trend_svg(cycles: list[dict[str, Any]]) -> str:
    series = {
        "retrieval_quality": [float(cycle.get("capabilities", {}).get("retrieval_quality", {}).get("score", 0.0)) for cycle in cycles],
        "fuzzy_understanding": [float(cycle.get("capabilities", {}).get("fuzzy_understanding", {}).get("score", 0.0)) for cycle in cycles],
        "qrels_trust": [float(cycle.get("capabilities", {}).get("qrels_trust", {}).get("score", 0.0)) for cycle in cycles],
    }
    return line_svg(series, title="Capability Trend")


def qrels_trust_stack_svg(cycle: dict[str, Any]) -> str:
    raw = cycle.get("raw_metrics", {})
    qrels_count = optional_float(raw.get("qrels_count")) or 0.0
    manual = optional_float(raw.get("manual_count")) or 0.0
    llm = optional_float(raw.get("llm_count")) or 0.0
    bootstrap = optional_float(raw.get("bootstrap_only_count")) or max(0.0, qrels_count - manual - llm)
    needs = optional_float(raw.get("needs_adjudication_count")) or 0.0
    return stacked_bar_svg(
        [
            ("bootstrap", bootstrap, "#9ca3af"),
            ("manual", manual, "#16a34a"),
            ("llm", llm, "#2563eb"),
            ("needs_adjudication", needs, "#f59e0b"),
        ],
        total=max(qrels_count, bootstrap + manual + llm + needs, 1.0),
        title="Qrels Trust Composition",
    )


def bar_svg(rows: list[tuple[str, float, str]], *, title: str, value_min: float, value_max: float) -> str:
    width = 760
    row_height = 34
    height = 70 + row_height * max(1, len(rows))
    chart_width = 440
    label_x = 24
    bar_x = 250
    lines = svg_header(width, height, title)
    for index, (name, value, kind) in enumerate(rows):
        y = 52 + index * row_height
        pct = clamp((value - value_min) / max(1.0, value_max - value_min), 0.0, 1.0)
        color = "#2563eb" if kind == "capability" else "#f59e0b"
        lines.append(f'<text x="{label_x}" y="{y + 18}" font-size="13" fill="#111827">{svg_escape(name)}</text>')
        lines.append(f'<rect x="{bar_x}" y="{y}" width="{chart_width}" height="20" fill="#e5e7eb" rx="3"/>')
        lines.append(f'<rect x="{bar_x}" y="{y}" width="{round(chart_width * pct, 2)}" height="20" fill="{color}" rx="3"/>')
        lines.append(f'<text x="{bar_x + chart_width + 12}" y="{y + 15}" font-size="12" fill="#111827">{round(value, 2)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def diverging_bar_svg(rows: list[tuple[str, float, str]], *, title: str, value_min: float, value_max: float) -> str:
    width = 760
    row_height = 34
    height = 70 + row_height * max(1, len(rows))
    chart_width = 440
    label_x = 24
    axis_x = 470
    scale = chart_width / max(abs(value_min), abs(value_max)) / 2
    lines = svg_header(width, height, title)
    lines.append(f'<line x1="{axis_x}" y1="45" x2="{axis_x}" y2="{height - 20}" stroke="#6b7280" stroke-width="1"/>')
    for index, (name, value, _kind) in enumerate(rows):
        y = 52 + index * row_height
        bar_width = min(chart_width / 2, abs(value) * scale)
        x = axis_x if value >= 0 else axis_x - bar_width
        color = "#16a34a" if value >= 0 else "#dc2626"
        lines.append(f'<text x="{label_x}" y="{y + 18}" font-size="13" fill="#111827">{svg_escape(name)}</text>')
        lines.append(f'<rect x="{round(x, 2)}" y="{y}" width="{round(bar_width, 2)}" height="20" fill="{color}" rx="3"/>')
        lines.append(f'<text x="{axis_x + chart_width / 2 + 16}" y="{y + 15}" font-size="12" fill="#111827">{signed_delta(value)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def line_svg(series: dict[str, list[float]], *, title: str) -> str:
    width = 760
    height = 320
    left = 58
    top = 50
    chart_width = 620
    chart_height = 210
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#7c3aed"]
    max_points = max([len(values) for values in series.values()] or [1])
    lines = svg_header(width, height, title)
    lines.append(f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d1d5db"/>')
    for tick in range(0, 101, 25):
        y = top + chart_height - chart_height * tick / 100
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_width}" y2="{y}" stroke="#eef2f7"/>')
        lines.append(f'<text x="18" y="{y + 4}" font-size="11" fill="#6b7280">{tick}</text>')
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = []
        for point_index, value in enumerate(values):
            x = left + (chart_width * point_index / max(1, max_points - 1))
            y = top + chart_height - chart_height * clamp(value, 0.0, 100.0) / 100
            points.append(f"{round(x, 2)},{round(y, 2)}")
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = 280 + index * 16
        lines.append(f'<rect x="{left + index * 190}" y="{legend_y - 10}" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{left + 14 + index * 190}" y="{legend_y}" font-size="12" fill="#111827">{svg_escape(name)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def stacked_bar_svg(rows: list[tuple[str, float, str]], *, total: float, title: str) -> str:
    width = 760
    height = 170
    bar_x = 40
    bar_y = 62
    bar_width = 650
    lines = svg_header(width, height, title)
    current_x = bar_x
    for name, value, color in rows:
        width_value = bar_width * value / max(1.0, total)
        lines.append(f'<rect x="{round(current_x, 2)}" y="{bar_y}" width="{round(width_value, 2)}" height="28" fill="{color}"/>')
        current_x += width_value
    lines.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="28" fill="none" stroke="#111827"/>')
    legend_x = bar_x
    for name, value, color in rows:
        lines.append(f'<rect x="{legend_x}" y="112" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{legend_x + 14}" y="122" font-size="12" fill="#111827">{svg_escape(name)}: {int(value)}</text>')
        legend_x += 165
    lines.append("</svg>")
    return "\n".join(lines)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9fafb"/>',
        f'<text x="24" y="30" font-size="18" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
    ]


def svg_escape(value: Any) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def signed_delta(value: Any) -> str:
    numeric = optional_float(value) or 0.0
    sign = "+" if numeric > 0 else ""
    return f"{sign}{round(numeric, 3)}"


def rank_cases_with_planner(
    *,
    prepared: PreparedMockIndex,
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    planner: str,
    constraint_profile: dict[str, Any],
    ranking_key: str,
    top_k: int,
    include_debug_text: bool,
    include_planner_debug: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    plan_result = plan_cases_from_args(cases, args=args, planner=planner)
    query_texts = query_texts_for_plans(plan_result.plans, args)
    cache.embed_texts(query_texts)
    signals = precompute_fast_case_signals(
        prepared,
        cache,
        cases,
        constraint_profile=constraint_profile,
        query_plans=plan_result.plans,
        max_query_texts=max_query_texts_arg(args),
    )
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key=ranking_key,
        constraint_profile=constraint_profile,
        top_k=top_k,
        include_debug_text=include_debug_text,
        include_planner_debug=include_planner_debug,
        include_all_results=False,
    )
    return plan_result, rows


def rank_cases_with_signals(
    *,
    prepared: PreparedMockIndex,
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    planner: str,
    constraint_profile: dict[str, Any],
    ranking_key: str,
    top_k: int,
) -> tuple[Any, list[dict[str, Any]], list[FastCaseSignals]]:
    plan_result = plan_cases_from_args(cases, args=args, planner=planner)
    cache.embed_texts(query_texts_for_plans(plan_result.plans, args))
    signals = precompute_fast_case_signals(
        prepared,
        cache,
        cases,
        constraint_profile=constraint_profile,
        query_plans=plan_result.plans,
        max_query_texts=max_query_texts_arg(args),
    )
    rows = rank_fast_cases(
        prepared,
        signals,
        ranking_key=ranking_key,
        constraint_profile=constraint_profile,
        top_k=top_k,
        include_debug_text=False,
        include_planner_debug=False,
        include_all_results=False,
    )
    return plan_result, rows, signals


def rank_pooled_runs(
    *,
    prepared: PreparedMockIndex,
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    planners: list[str],
    ranking_keys: list[str],
    constraint_profile: dict[str, Any],
    top_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    run_rows: dict[str, list[dict[str, Any]]] = {}
    planner_stats: dict[str, Any] = {}
    for planner in planners:
        plan_result = plan_cases_from_args(cases, args=args, planner=planner)
        planner_stats[planner] = plan_result.stats
        cache.embed_texts(query_texts_for_plans(plan_result.plans, args))
        signals = precompute_fast_case_signals(
            prepared,
            cache,
            cases,
            constraint_profile=constraint_profile,
            query_plans=plan_result.plans,
            max_query_texts=max_query_texts_arg(args),
        )
        for ranking_key in ranking_keys:
            run_name = f"{planner}::{ranking_key}"
            run_rows[run_name] = rank_fast_cases(
                prepared,
                signals,
                ranking_key=ranking_key,
                constraint_profile=constraint_profile,
                top_k=top_k,
                include_debug_text=False,
                include_planner_debug=False,
                include_all_results=False,
            )
    return run_rows, planner_stats


def precompute_nonsemantic_case_signals(
    prepared: PreparedMockIndex,
    cases: list[dict[str, Any]],
    *,
    query_plans: list[ExperimentalQueryPlan],
    max_query_texts: int = 0,
) -> list[FastCaseSignals]:
    if len(query_plans) != len(cases):
        raise ValueError("query_plans must have the same length as cases")
    signals: list[FastCaseSignals] = []
    zeros = np.zeros(prepared.bm25_doc_count, dtype=np.float64)
    for case, query_plan in zip(cases, query_plans):
        signals.append(
            FastCaseSignals(
                case=case,
                user_input=case["user_input"],
                semantic_scores=zeros.copy(),
                channel_scores={},
                query_constraints=planner_constraints(query_plan),
                query_plan=query_plan,
                query_channels=build_query_channels_for_plan(query_plan, max_query_texts=max_query_texts),
                query_signature=query_scene_signature(query_plan),
            )
        )
    return signals


def style_risk_fallback_ranking_key(ranking_key: str) -> str:
    if ranking_key in {"constraints_only", "lexical_only", "signature_only", "lexical_constraints", "lexical_constraints_signature"}:
        return ranking_key
    if "signature" in ranking_key:
        return "lexical_constraints_signature"
    return "lexical_constraints"


def positive_eval_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        case
        for case in read_cases(args.inputs, args.limit, split=args.split)
        if case.get("expected_relation") == "should_match"
    ]


def bootstrap_qrels_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qrels: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        query_id = row["case_id"]
        target_id = row.get("target_item_id")
        if target_id:
            qrels[(query_id, target_id)] = {
                "query_id": query_id,
                "item_id": target_id,
                "grade": 3,
                "reason": "target item from generated eval case",
                "source": "bootstrap",
            }
        for result in row.get("top_results", []):
            item_id = result.get("item_id")
            if not item_id:
                continue
            grade, reason = bootstrap_grade(row, result)
            key = (query_id, item_id)
            current = qrels.get(key)
            if current is None or grade > int(current["grade"]):
                qrels[key] = {
                    "query_id": query_id,
                    "item_id": item_id,
                    "grade": grade,
                    "reason": reason,
                    "source": "bootstrap",
                }
    return list(qrels.values())


def pooled_qrels_from_run_rows(run_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    qrels: dict[tuple[str, str], dict[str, Any]] = {}
    for run_name, rows in run_rows.items():
        for row in rows:
            query_id = row["case_id"]
            target_id = row.get("target_item_id")
            if target_id:
                merge_qrel(
                    qrels,
                    query_id=query_id,
                    item_id=target_id,
                    grade=3,
                    reason="target item from generated eval case",
                    source="pooled_bootstrap",
                    run_name=run_name,
                    rank=row.get("target_rank"),
                )
            for rank, result in enumerate(row.get("top_results", []), start=1):
                item_id = result.get("item_id")
                if not item_id:
                    continue
                grade, reason = bootstrap_grade(row, result)
                merge_qrel(
                    qrels,
                    query_id=query_id,
                    item_id=item_id,
                    grade=grade,
                    reason=reason,
                    source="pooled_bootstrap",
                    run_name=run_name,
                    rank=rank,
                )
    return sorted(qrels.values(), key=lambda row: (row["query_id"], -int(row["grade"]), row["item_id"]))


def merge_qrel(
    qrels: dict[tuple[str, str], dict[str, Any]],
    *,
    query_id: str,
    item_id: str,
    grade: int,
    reason: str,
    source: str,
    run_name: str,
    rank: int | None,
) -> None:
    key = (query_id, item_id)
    pooled_from = {"run": run_name}
    if rank is not None:
        pooled_from["rank"] = int(rank)
    existing = qrels.get(key)
    if existing is None:
        qrels[key] = {
            "query_id": query_id,
            "item_id": item_id,
            "grade": int(grade),
            "reason": reason,
            "source": source,
            "pooled_from": [pooled_from],
            "grade_votes": [
                {
                    "run": run_name,
                    "rank": rank,
                    "grade": int(grade),
                    "reason": reason,
                    "judge_type": "bootstrap",
                    "judge_id": run_name,
                    "judge_version": "pooled_v1",
                }
            ],
        }
        qrels[key]["confidence"] = qrel_confidence(qrels[key])
        return
    existing.setdefault("pooled_from", []).append(pooled_from)
    existing.setdefault("grade_votes", []).append(
        {
            "run": run_name,
            "rank": rank,
            "grade": int(grade),
            "reason": reason,
            "judge_type": "bootstrap",
            "judge_id": run_name,
            "judge_version": "pooled_v1",
        }
    )
    if int(grade) > int(existing["grade"]):
        existing["grade"] = int(grade)
        existing["reason"] = reason
        existing["source"] = source
    existing["confidence"] = qrel_confidence(existing)


def pooled_qrels_summary(
    qrels: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    run_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_query: dict[str, set[str]] = {}
    for qrel in qrels:
        by_query.setdefault(qrel["query_id"], set()).add(qrel["item_id"])
    pool_sizes = [len(by_query.get(case["case_id"], set())) for case in cases]
    return {
        "qrels_count": len(qrels),
        "query_count": len(cases),
        "run_count": len(run_rows),
        "avg_pool_size": round(mean([float(size) for size in pool_sizes]), 6),
        "max_pool_size": max(pool_sizes) if pool_sizes else 0,
        "grade_counts": dict(sorted(Counter(int(row["grade"]) for row in qrels).items())),
    }


def qrels_audit_summary(qrels: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = [qrel_confidence(row) for row in qrels]
    pooled_counts = [len(row.get("pooled_from", [])) for row in qrels]
    conflict_count = sum(1 for row in qrels if qrel_has_vote_conflict(row))
    needs_adjudication_count = sum(1 for row in qrels if qrel_needs_adjudication(row))
    manual_count = sum(1 for row in qrels if qrel_has_judge_type(row, "human"))
    llm_count = sum(1 for row in qrels if qrel_has_judge_type(row, "llm"))
    reviewed_count = manual_count + llm_count
    bootstrap_only_count = sum(1 for row in qrels if qrel_is_bootstrap_only(row))
    source_counts = Counter(str(row.get("source", "unknown")) for row in qrels)
    return {
        "qrels_count": len(qrels),
        "grade_counts": dict(sorted(Counter(int(row.get("grade", 0)) for row in qrels).items())),
        "source_counts": dict(sorted(source_counts.items())),
        "mean_confidence": round(mean(confidences), 6),
        "high_confidence_rate": round(sum(1 for value in confidences if value >= 0.75) / max(1, len(confidences)), 6),
        "low_confidence_count": sum(1 for value in confidences if value < 0.6),
        "conflict_count": conflict_count,
        "conflict_rate": round(conflict_count / max(1, len(qrels)), 6),
        "vote_conflict_rate": round(conflict_count / max(1, len(qrels)), 6),
        "manual_count": manual_count,
        "llm_count": llm_count,
        "bootstrap_only_count": bootstrap_only_count,
        "needs_adjudication_count": needs_adjudication_count,
        "mean_pooled_from_count": round(mean([float(value) for value in pooled_counts]), 6),
        "manual_or_llm_count": reviewed_count,
        "qrels_trust_level": qrels_trust_level(qrels),
    }


def qrel_confidence(row: dict[str, Any]) -> float:
    source = str(row.get("source", ""))
    grade = int(row.get("grade", 0))
    pooled_count = len(row.get("pooled_from", []))
    vote_rows = [vote for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    votes = [int(vote.get("grade", grade)) for vote in vote_rows]
    human_votes = [vote for vote in vote_rows if qrel_vote_judge_type(vote, row) == "human"]
    llm_votes = [vote for vote in vote_rows if qrel_vote_judge_type(vote, row) == "llm"]
    if source.startswith("manual"):
        return round(max([float(vote.get("confidence", 0.95)) for vote in human_votes] or [0.95]), 6)
    if source.startswith("llm"):
        return round(max([float(vote.get("confidence", 0.85)) for vote in llm_votes] or [0.85]), 6)
    if human_votes:
        return round(max(float(vote.get("confidence", 0.95)) for vote in human_votes), 6)
    if llm_votes:
        return round(max(float(vote.get("confidence", 0.85)) for vote in llm_votes), 6)
    if grade == 3 and "target item" in str(row.get("reason", "")):
        return 0.95
    if votes:
        if len(votes) == 1 and grade <= 1:
            return 0.5
        agreement = max(Counter(votes).values()) / len(votes)
        base = 0.45 + 0.35 * agreement + min(0.15, 0.03 * len(votes))
        if len(set(votes)) > 1:
            base -= 0.15
        return round(min(0.9, max(0.3, base)), 6)
    if pooled_count >= 3:
        return 0.75
    if pooled_count >= 2:
        return 0.65
    return 0.5


def qrel_has_vote_conflict(row: dict[str, Any]) -> bool:
    votes = [int(vote.get("grade", row.get("grade", 0))) for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    return len(set(votes)) > 1


def qrel_vote_judge_type(vote: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    judge_type = str(vote.get("judge_type", "")).lower().strip()
    if judge_type in {"human", "manual"}:
        return "human"
    if judge_type in {"llm", "model"}:
        return "llm"
    if judge_type == "bootstrap":
        return "bootstrap"
    source = str((row or {}).get("source", "")).lower()
    if source.startswith("manual"):
        return "human"
    if source.startswith("llm"):
        return "llm"
    return "bootstrap"


def qrel_has_judge_type(row: dict[str, Any], judge_type: str) -> bool:
    votes = [vote for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    if any(qrel_vote_judge_type(vote, row) == judge_type for vote in votes):
        return True
    source = str(row.get("source", "")).lower()
    return judge_type == "human" and source.startswith("manual") or judge_type == "llm" and source.startswith("llm")


def qrel_is_bootstrap_only(row: dict[str, Any]) -> bool:
    return not qrel_has_judge_type(row, "human") and not qrel_has_judge_type(row, "llm")


def qrel_needs_adjudication(row: dict[str, Any]) -> bool:
    return bool(row.get("needs_adjudication")) or qrel_has_vote_conflict(row) or qrel_confidence(row) < 0.6


def qrels_trust_level(qrels: list[dict[str, Any]]) -> str:
    if not qrels:
        return "low"
    reviewed = sum(1 for row in qrels if not qrel_is_bootstrap_only(row))
    reviewed_rate = reviewed / len(qrels)
    needs_rate = sum(1 for row in qrels if qrel_needs_adjudication(row)) / len(qrels)
    conflict_rate = sum(1 for row in qrels if qrel_has_vote_conflict(row)) / len(qrels)
    if reviewed == 0:
        return "low"
    if reviewed_rate >= 0.5 and needs_rate <= 0.1 and conflict_rate <= 0.05:
        return "high"
    return "medium"


def qrel_relevance_vote(
    *,
    query_id: str,
    item_id: str,
    grade: int,
    reason: str,
    judge_type: str,
    judge_id: str,
    judge_version: str,
    confidence: float | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    vote = {
        "query_id": str(query_id),
        "item_id": str(item_id),
        "grade": int(grade),
        "reason": str(reason),
        "judge_type": "human" if judge_type == "manual" else str(judge_type),
        "judge_id": str(judge_id),
        "judge_version": str(judge_version),
    }
    if confidence is not None:
        vote["confidence"] = float(confidence)
    if created_at:
        vote["created_at"] = str(created_at)
    return vote


def load_adjudication_votes(
    path: Path,
    *,
    default_judge_type: str,
    default_judge_id: str,
    judge_version: str,
) -> list[dict[str, Any]]:
    votes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        votes.extend(
            adjudication_votes_from_row(
                row,
                default_judge_type=default_judge_type,
                default_judge_id=default_judge_id,
                judge_version=judge_version,
            )
        )
    return votes


def adjudication_votes_from_row(
    row: dict[str, Any],
    *,
    default_judge_type: str,
    default_judge_id: str,
    judge_version: str,
) -> list[dict[str, Any]]:
    base = {
        "query_id": row.get("query_id"),
        "judge_type": row.get("judge_type", default_judge_type),
        "judge_id": row.get("judge_id", default_judge_id),
        "judge_version": row.get("judge_version", judge_version),
        "created_at": row.get("created_at"),
    }
    raw_votes = row.get("judgements") or row.get("judgments")
    if isinstance(raw_votes, list):
        votes = []
        for item in raw_votes:
            if not isinstance(item, dict):
                continue
            if "grade" not in item and "score" not in item:
                continue
            votes.append(
                qrel_relevance_vote(
                    query_id=str(item.get("query_id", base["query_id"])),
                    item_id=str(item.get("item_id", "")),
                    grade=int(item.get("grade", item.get("score", 0))),
                    reason=str(item.get("reason", row.get("reason", ""))),
                    judge_type=str(item.get("judge_type", base["judge_type"])),
                    judge_id=str(item.get("judge_id", base["judge_id"])),
                    judge_version=str(item.get("judge_version", base["judge_version"])),
                    confidence=_optional_float(item.get("confidence", row.get("confidence"))),
                    created_at=str(item.get("created_at", base.get("created_at") or "")) or None,
                )
            )
        return [vote for vote in votes if vote["query_id"] and vote["item_id"]]
    if "grade" not in row and "score" not in row:
        return []
    return [
        qrel_relevance_vote(
            query_id=str(row.get("query_id", "")),
            item_id=str(row.get("item_id", "")),
            grade=int(row.get("grade", row.get("score", 0))),
            reason=str(row.get("reason", row.get("suggested_reason", ""))),
            judge_type=str(base["judge_type"]),
            judge_id=str(base["judge_id"]),
            judge_version=str(base["judge_version"]),
            confidence=_optional_float(row.get("confidence")),
            created_at=str(base.get("created_at") or "") or None,
        )
    ]


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def merge_adjudicated_qrels(existing_qrels: list[dict[str, Any]], votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {(str(row["query_id"]), str(row["item_id"])): dict(row) for row in existing_qrels}
    for vote in votes:
        validate_qrel(vote)
        key = (str(vote["query_id"]), str(vote["item_id"]))
        row = merged.setdefault(
            key,
            {
                "query_id": vote["query_id"],
                "item_id": vote["item_id"],
                "grade": int(vote["grade"]),
                "reason": vote.get("reason", ""),
                "source": f"{qrel_vote_judge_type(vote)}_adjudicated",
                "pooled_from": [],
                "grade_votes": [],
            },
        )
        row.setdefault("grade_votes", []).append(dict(vote))
        recompute_qrel_from_votes(row)
    return sorted(merged.values(), key=lambda row: (row["query_id"], -int(row["grade"]), row["item_id"]))


def recompute_qrel_from_votes(row: dict[str, Any]) -> None:
    votes = [vote for vote in row.get("grade_votes", []) if isinstance(vote, dict)]
    selected_vote = selected_qrel_vote(votes, row)
    if selected_vote is not None:
        row["grade"] = int(selected_vote.get("grade", row.get("grade", 0)))
        row["reason"] = str(selected_vote.get("reason", row.get("reason", "")))
        selected_type = qrel_vote_judge_type(selected_vote, row)
        if selected_type == "human":
            row["source"] = "manual_adjudicated"
        elif selected_type == "llm":
            row["source"] = "llm_adjudicated"
        else:
            row.setdefault("source", "pooled_bootstrap")
    row["needs_adjudication"] = qrel_has_vote_conflict(row)
    row["confidence"] = qrel_confidence(row)


def selected_qrel_vote(votes: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any] | None:
    for judge_type in ("human", "llm", "bootstrap"):
        typed = [vote for vote in votes if qrel_vote_judge_type(vote, row) == judge_type]
        if not typed:
            continue
        counts = Counter(int(vote.get("grade", 0)) for vote in typed)
        selected_grade = max(counts.items(), key=lambda pair: (pair[1], pair[0]))[0]
        selected = [vote for vote in typed if int(vote.get("grade", 0)) == selected_grade]
        return max(selected, key=lambda vote: float(vote.get("confidence", 0.0)))
    return None


def bootstrap_grade(row: dict[str, Any], result: dict[str, Any]) -> tuple[int, str]:
    if result.get("constraint_hits", {}).get("negative_style"):
        return 0, "candidate violates negative style constraint"
    if result.get("item_id") == row.get("target_item_id"):
        return 3, "exact generated target"
    result_stage_value = result_stage(result)
    stage_match = result_stage_value == row.get("target_stage")
    target_purposes = set(row.get("target_purposes", []))
    result_purposes = set(result.get("metadata", {}).get("creative_purpose", []))
    purpose_match = bool(target_purposes & result_purposes)
    if stage_match and purpose_match:
        return 2, "same stage and overlapping creative purpose"
    if stage_match or purpose_match:
        return 1, "partial stage or purpose match"
    return 0, "no generated relevance signal"


def write_qrels(path: Path, qrels: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for row in qrels:
        validate_qrel(row)
        key = (str(row["query_id"]), str(row["item_id"]))
        if key not in seen or int(row["grade"]) > int(seen[key]["grade"]):
            seen[key] = row
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in seen.values()),
        encoding="utf-8",
    )


def merge_qrel_lists(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in [*primary, *secondary]:
        validate_qrel(row)
        key = (str(row["query_id"]), str(row["item_id"]))
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(row)
            continue
        if qrel_row_priority(row) > qrel_row_priority(existing):
            combined = dict(row)
            combined["grade_votes"] = [*existing.get("grade_votes", []), *row.get("grade_votes", [])]
            combined["pooled_from"] = [*existing.get("pooled_from", []), *row.get("pooled_from", [])]
            recompute_qrel_from_votes(combined)
            merged[key] = combined
        else:
            existing["grade_votes"] = [*existing.get("grade_votes", []), *row.get("grade_votes", [])]
            existing["pooled_from"] = [*existing.get("pooled_from", []), *row.get("pooled_from", [])]
            recompute_qrel_from_votes(existing)
    return sorted(merged.values(), key=lambda row: (row["query_id"], -int(row["grade"]), row["item_id"]))


def qrel_row_priority(row: dict[str, Any]) -> tuple[int, float, int]:
    if qrel_has_judge_type(row, "human"):
        tier = 3
    elif qrel_has_judge_type(row, "llm"):
        tier = 2
    else:
        tier = 1
    return (tier, qrel_confidence(row), int(row.get("grade", 0)))


def qrels_judged_coverage(rows: list[dict[str, Any]], qrels: list[dict[str, Any]], *, top_k: int) -> float:
    qrel_map = {(row["query_id"], row["item_id"]): int(row["grade"]) for row in qrels}
    return mean([judged_at(row, qrel_map, top_k) for row in rows])


def load_qrels(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validate_qrel(row)
        rows.append(row)
    return rows


def validate_qrel(row: dict[str, Any]) -> None:
    if not row.get("query_id") or not row.get("item_id"):
        raise ValueError("qrel requires query_id and item_id")
    grade = int(row.get("grade", -1))
    if grade < 0 or grade > 3:
        raise ValueError(f"qrel grade must be 0..3, got {grade}")


def graded_metrics(rows: list[dict[str, Any]], qrels: list[dict[str, Any]], *, top_k: int) -> dict[str, float]:
    qrel_map = {(row["query_id"], row["item_id"]): int(row["grade"]) for row in qrels}
    by_query: dict[str, list[int]] = {}
    for qrel in qrels:
        by_query.setdefault(qrel["query_id"], []).append(int(qrel["grade"]))
    return {
        "nDCG@3": round(mean([ndcg_at(row, qrel_map, by_query, 3) for row in rows]), 6),
        "nDCG@10": round(mean([ndcg_at(row, qrel_map, by_query, 10) for row in rows]), 6),
        "ERR@10": round(mean([err_at(row, qrel_map, 10) for row in rows]), 6),
        "MRR@10": round(mean([mrr_at(row, qrel_map, 10) for row in rows]), 6),
        "Judged@10": round(mean([judged_at(row, qrel_map, 10) for row in rows]), 6),
        "Unjudged@10": round(mean([unjudged_at(row, qrel_map, 10) for row in rows]), 6),
        "Recall@10": round(mean([graded_recall_at(row, qrel_map, by_query, 10) for row in rows]), 6),
        "case_count": len(rows),
        "qrels_count": len(qrels),
    }


def ndcg_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], by_query: dict[str, list[int]], k: int) -> float:
    query_id = row["case_id"]
    grades = [qrels.get((query_id, result["item_id"]), 0) for result in row.get("top_results", [])[:k]]
    dcg = discounted_gain(grades)
    ideal = discounted_gain(sorted(by_query.get(query_id, []), reverse=True)[:k])
    return 0.0 if ideal == 0 else dcg / ideal


def discounted_gain(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def err_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int, *, max_grade: int = 3) -> float:
    query_id = row["case_id"]
    carry = 1.0
    total = 0.0
    max_gain = float(2**max_grade)
    for rank, result in enumerate(row.get("top_results", [])[:k], start=1):
        grade = qrels.get((query_id, result["item_id"]), 0)
        relevance_probability = (2**grade - 1) / max_gain
        total += carry * relevance_probability / rank
        carry *= 1 - relevance_probability
    return total


def mrr_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int) -> float:
    query_id = row["case_id"]
    for rank, result in enumerate(row.get("top_results", [])[:k], start=1):
        if qrels.get((query_id, result["item_id"]), 0) > 0:
            return 1 / rank
    return 0.0


def judged_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int) -> float:
    query_id = row["case_id"]
    results = row.get("top_results", [])[:k]
    if not results:
        return 0.0
    return sum(1 for result in results if (query_id, result["item_id"]) in qrels) / len(results)


def unjudged_at(row: dict[str, Any], qrels: dict[tuple[str, str], int], k: int) -> float:
    results = row.get("top_results", [])[:k]
    if not results:
        return 0.0
    return 1.0 - judged_at(row, qrels, k)


def graded_recall_at(
    row: dict[str, Any],
    qrels: dict[tuple[str, str], int],
    by_query: dict[str, list[int]],
    k: int,
) -> float:
    query_id = row["case_id"]
    relevant_total = sum(1 for grade in by_query.get(query_id, []) if grade >= 2)
    if relevant_total == 0:
        return 0.0
    retrieved = sum(1 for result in row.get("top_results", [])[:k] if qrels.get((query_id, result["item_id"]), 0) >= 2)
    return retrieved / relevant_total


def recall_bound_rows(
    rows_by_key: dict[str, list[dict[str, Any]]],
    *,
    baseline_key: str,
    candidate_depth: int,
    top_k: int,
) -> list[dict[str, Any]]:
    baseline_rows = rows_by_key.get(baseline_key, [])
    rows_by_case = {
        key: {row["case_id"]: row for row in rows}
        for key, rows in rows_by_key.items()
    }
    result = []
    for baseline_row in baseline_rows:
        case_id = baseline_row["case_id"]
        ranks = {
            key: rows.get(case_id, {}).get("target_rank")
            for key, rows in rows_by_case.items()
        }
        known_ranks = [rank for rank in ranks.values() if rank is not None]
        oracle_best_rank = min(known_ranks) if known_ranks else None
        candidate_hit = bool(oracle_best_rank is not None and oracle_best_rank <= candidate_depth)
        baseline_rank = ranks.get(baseline_key)
        failure_type = recall_bound_failure_type(
            baseline_rank=baseline_rank,
            oracle_best_rank=oracle_best_rank,
            candidate_depth=candidate_depth,
            top_k=top_k,
        )
        result.append(
            {
                "case_id": case_id,
                "user_input": baseline_row.get("user_input", ""),
                "target_item_id": baseline_row.get("target_item_id"),
                "baseline_rank": baseline_rank,
                "oracle_best_rank": oracle_best_rank,
                "candidate_depth_hit": candidate_hit,
                "rank_by_workflow": ranks,
                "failure_type": failure_type,
                "top1_by_baseline": baseline_row.get("top_results", [{}])[0].get("item_id")
                if baseline_row.get("top_results")
                else None,
            }
        )
    return result


def recall_bound_failure_type(
    *,
    baseline_rank: int | None,
    oracle_best_rank: int | None,
    candidate_depth: int,
    top_k: int,
) -> str:
    if baseline_rank is not None and baseline_rank <= top_k:
        return "success"
    if oracle_best_rank is None or oracle_best_rank > candidate_depth:
        return "candidate_recall_failure"
    if oracle_best_rank <= top_k:
        return "workflow_selection_failure"
    return "fusion_ranking_failure"


def recall_bound_summary(rows: list[dict[str, Any]], *, top_k: int, candidate_depth: int) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        f"baseline_recall_at_{top_k}": round(
            sum(1 for row in rows if row.get("baseline_rank") is not None and row["baseline_rank"] <= top_k)
            / max(1, len(rows)),
            6,
        ),
        f"oracle_recall_at_{top_k}": round(
            sum(1 for row in rows if row.get("oracle_best_rank") is not None and row["oracle_best_rank"] <= top_k)
            / max(1, len(rows)),
            6,
        ),
        f"oracle_recall_at_{candidate_depth}": round(
            sum(1 for row in rows if row.get("candidate_depth_hit")) / max(1, len(rows)),
            6,
        ),
        "failure_type_counts": dict(sorted(Counter(row["failure_type"] for row in rows).items())),
        "mean_oracle_best_rank": round(
            mean([float(row["oracle_best_rank"]) for row in rows if row.get("oracle_best_rank") is not None]),
            6,
        ),
    }


def active_qrels_samples(
    run_rows: dict[str, list[dict[str, Any]]],
    *,
    existing_qrels: list[dict[str, Any]],
    sample_size: int,
    include_judged: bool,
) -> list[dict[str, Any]]:
    reviewed = {
        (row["query_id"], row["item_id"])
        for row in existing_qrels
        if not qrel_is_bootstrap_only(row) and not qrel_needs_adjudication(row)
    }
    existing_by_key = {(row["query_id"], row["item_id"]): row for row in existing_qrels}
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    ranks: dict[tuple[str, str], list[int]] = {}
    for run_name, rows in run_rows.items():
        for row in rows:
            query_id = row["case_id"]
            query_bucket = by_query.setdefault(query_id, {})
            for rank, result in enumerate(row.get("top_results", []), start=1):
                item_id = result["item_id"]
                candidate = query_bucket.setdefault(
                    item_id,
                    {
                        "query_id": query_id,
                        "item_id": item_id,
                        "user_input": row.get("user_input", ""),
                        "target_item_id": row.get("target_item_id"),
                        "metadata": result.get("metadata", {}),
                        "constraint_hits": result.get("constraint_hits", {}),
                        "scores": {},
                        "pooled_from": [],
                    },
                )
                candidate["scores"][run_name] = result.get("score")
                candidate["pooled_from"].append({"run": run_name, "rank": rank})
                ranks.setdefault((query_id, item_id), []).append(rank)
    samples = []
    for query_id, items in by_query.items():
        target_missed = all(
            row.get("target_rank") is None or row.get("target_rank", 999999) > len(row.get("top_results", []))
            for rows in run_rows.values()
            for row in rows
            if row.get("case_id") == query_id
        )
        query_top_candidates = top_candidates_for_active_review(items, ranks, limit=5)
        for item_id, candidate in items.items():
            if not include_judged and (query_id, item_id) in reviewed:
                continue
            candidate_ranks = ranks.get((query_id, item_id), [])
            reasons = active_sample_reasons(candidate, candidate_ranks, target_missed=target_missed)
            existing_qrel = existing_by_key.get((query_id, item_id))
            if existing_qrel is not None and qrel_confidence(existing_qrel) < 0.6:
                reasons.append("low_confidence_qrel")
            if existing_qrel is not None and qrel_needs_adjudication(existing_qrel):
                reasons.append("needs_adjudication")
            if not reasons:
                continue
            suggested_grade, reason = bootstrap_grade(
                {
                    "case_id": query_id,
                    "target_item_id": candidate.get("target_item_id"),
                    "target_stage": target_stage_from_target_id(candidate.get("target_item_id"), run_rows, query_id),
                    "target_purposes": target_purposes_from_rows(run_rows, query_id),
                },
                candidate,
            )
            samples.append(
                {
                    **candidate,
                    "query": candidate.get("user_input", ""),
                    "target": {
                        "item_id": candidate.get("target_item_id"),
                        "stage": target_stage_from_target_id(candidate.get("target_item_id"), run_rows, query_id),
                        "purposes": target_purposes_from_rows(run_rows, query_id),
                    },
                    "candidate_summary": reranker_candidate_summary(candidate, rank=min(candidate_ranks or [0]) or None),
                    "top_candidates": query_top_candidates,
                    "suggested_grade": suggested_grade,
                    "suggested_reason": reason,
                    "suggested_granularity": suggested_adjudication_granularity(reasons),
                    "low_confidence_reasons": low_confidence_reasons_for_active_sample(reasons, candidate_ranks),
                    "workflow_disagreement": "workflow_rank_disagreement" in reasons,
                    "existing_qrel": compact_existing_qrel(existing_qrel),
                    "adjudication_schema": {
                        "query_id": query_id,
                        "item_id": item_id,
                        "grade": "0..3",
                        "reason": "short reason",
                        "judge_type": "human|llm",
                        "confidence": "0..1 optional",
                    },
                    "reasons": reasons,
                    "priority": active_sample_priority(reasons, candidate_ranks),
                    "judged": (query_id, item_id) in reviewed,
                }
            )
    samples.sort(key=lambda row: (-float(row["priority"]), row["query_id"], row["item_id"]))
    return samples[:sample_size]


def active_sample_reasons(candidate: dict[str, Any], ranks: list[int], *, target_missed: bool) -> list[str]:
    reasons = []
    if target_missed:
        reasons.append("target_miss_query")
    if len(ranks) >= 2 and max(ranks) - min(ranks) >= 8:
        reasons.append("workflow_rank_disagreement")
    if len(candidate.get("scores", {})) == 1:
        reasons.append("single_workflow_only")
    if candidate.get("constraint_hits", {}).get("negative_style"):
        reasons.append("style_risk_candidate")
    if min(ranks or [999999]) <= 3:
        reasons.append("top3_candidate")
    return reasons


def active_sample_priority(reasons: list[str], ranks: list[int]) -> float:
    weights = {
        "target_miss_query": 3.0,
        "workflow_rank_disagreement": 2.0,
        "style_risk_candidate": 2.0,
        "needs_adjudication": 2.5,
        "low_confidence_qrel": 2.0,
        "top3_candidate": 1.5,
        "single_workflow_only": 0.75,
    }
    return round(sum(weights.get(reason, 0.5) for reason in reasons) + 1 / max(1, min(ranks or [100])), 6)


def top_candidates_for_active_review(
    items: dict[str, dict[str, Any]],
    ranks: dict[tuple[str, str], list[int]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for item_id, candidate in items.items():
        query_id = str(candidate.get("query_id", ""))
        item_ranks = ranks.get((query_id, item_id), [])
        candidates.append(
            {
                **reranker_candidate_summary(candidate, rank=min(item_ranks or [999999])),
                "pooled_from": candidate.get("pooled_from", [])[:5],
            }
        )
    candidates.sort(key=lambda row: (int(row.get("rank") or 999999), str(row.get("item_id", ""))))
    return candidates[:limit]


def compact_existing_qrel(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "grade": int(row.get("grade", 0)),
        "reason": row.get("reason", ""),
        "source": row.get("source", ""),
        "confidence": qrel_confidence(row),
        "needs_adjudication": qrel_needs_adjudication(row),
        "vote_count": len(row.get("grade_votes", [])),
    }


def suggested_adjudication_granularity(reasons: list[str]) -> str:
    if "style_risk_candidate" in reasons:
        return "constraint_level"
    if "target_miss_query" in reasons or "workflow_rank_disagreement" in reasons:
        return "scene_or_purpose_level"
    return "scene_level"


def low_confidence_reasons_for_active_sample(reasons: list[str], ranks: list[int]) -> list[str]:
    low_reasons = []
    if "single_workflow_only" in reasons:
        low_reasons.append("candidate only appears in one workflow")
    if "workflow_rank_disagreement" in reasons:
        low_reasons.append("candidate rank differs strongly across workflows")
    if "target_miss_query" in reasons:
        low_reasons.append("target is missing from the sampled pool")
    if "low_confidence_qrel" in reasons:
        low_reasons.append("existing qrel confidence is low")
    if "needs_adjudication" in reasons:
        low_reasons.append("existing qrel has conflicting votes or needs adjudication")
    if ranks and min(ranks) <= 3 and max(ranks) >= 10:
        low_reasons.append("candidate is top ranked by one workflow but weak in another")
    return low_reasons


def target_stage_from_target_id(target_id: str | None, run_rows: dict[str, list[dict[str, Any]]], query_id: str) -> str:
    for rows in run_rows.values():
        for row in rows:
            if row.get("case_id") == query_id:
                return row.get("target_stage", "")
    return ""


def target_purposes_from_rows(run_rows: dict[str, list[dict[str, Any]]], query_id: str) -> list[str]:
    for rows in run_rows.values():
        for row in rows:
            if row.get("case_id") == query_id:
                return list(row.get("target_purposes", []))
    return []


def rerank_row_by_rule(row: dict[str, Any], *, rerank_depth: int, top_k: int) -> dict[str, Any]:
    candidates = row.get("top_results", [])[:rerank_depth]
    reranked = sorted(candidates, key=rule_rerank_score, reverse=True)
    return row_with_reranked_results(row, reranked[:top_k], ranking_key=f"rule_rerank@{rerank_depth}")


def rule_rerank_score(result: dict[str, Any]) -> float:
    hits = result.get("constraint_hits", {})
    score = float(result.get("score", 0.0))
    score += 0.45 * float(result.get("signature_score", 0.0))
    score += 0.25 * float(result.get("constraint_score", 0.0))
    if hits.get("desired_stage"):
        score += 0.15
    if hits.get("positive_style"):
        score += 0.1 * len(hits["positive_style"])
    if hits.get("negative_style"):
        score -= 0.5 * len(hits["negative_style"])
    return score


def rerank_row_by_qrels(
    row: dict[str, Any],
    qrels: list[dict[str, Any]],
    *,
    rerank_depth: int,
    top_k: int,
) -> dict[str, Any]:
    qrel_map = {(qrel["query_id"], qrel["item_id"]): int(qrel["grade"]) for qrel in qrels}
    query_id = row["case_id"]
    candidates = row.get("top_results", [])[:rerank_depth]
    reranked = sorted(
        candidates,
        key=lambda result: (qrel_map.get((query_id, result["item_id"]), 0), float(result.get("score", 0.0))),
        reverse=True,
    )
    return row_with_reranked_results(row, reranked[:top_k], ranking_key=f"qrels_oracle_rerank@{rerank_depth}")


def rerank_rows_with_llm_sample(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    rerank_depth: int,
    top_k: int,
    timeout_seconds: float,
    retries: int,
    require_llm: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = sorted(rows, key=llm_rerank_priority, reverse=True)[:sample_size]
    reranked_rows = []
    errors = []
    calls = 0
    payload_chars = []
    candidate_counts = []
    for row in selected:
        calls += 1
        preview_payload = llm_rerank_payload(row, rerank_depth=rerank_depth)
        payload_chars.append(len(json.dumps(preview_payload, ensure_ascii=False)))
        candidate_counts.append(len(preview_payload.get("candidates", [])))
        try:
            reranked_rows.append(
                rerank_row_by_llm(
                    row,
                    rerank_depth=rerank_depth,
                    top_k=top_k,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
            )
        except Exception as exc:
            if require_llm:
                raise
            errors.append({"case_id": row.get("case_id"), "error": str(exc)})
            reranked_rows.append(row_with_reranked_results(row, row.get("top_results", [])[:top_k], ranking_key=f"llm_rerank_failed@{rerank_depth}"))
    return reranked_rows, {
        "requested_sample_size": sample_size,
        "selected_count": len(selected),
        "llm_call_count": calls,
        "fallback_count": len(errors),
        "avg_candidates_per_call": round(mean([float(value) for value in candidate_counts]), 6) if candidate_counts else 0.0,
        "estimated_payload_chars": sum(payload_chars),
        "errors": errors[:10],
    }


def llm_rerank_priority(row: dict[str, Any]) -> float:
    margin = case_margin(row)
    priority = 1.0 - min(1.0, max(0.0, margin))
    if row.get("target_rank") is None or row.get("target_rank", 999999) > 10:
        priority += 1.0
    if style_violation_at([row], 3) > 0:
        priority += 0.75
    return priority


def rerank_row_by_llm(
    row: dict[str, Any],
    *,
    rerank_depth: int,
    top_k: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    from sceneweaver.llm.client import VisionLLMClient

    candidates = row.get("top_results", [])[:rerank_depth]
    payload = llm_rerank_payload(row, rerank_depth=rerank_depth)
    response = VisionLLMClient().analyze_text_json(
        system_prompt="You are a strict retrieval reranker. Use only the supplied candidates and return JSON.",
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=1200,
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    ranked_ids = [str(item_id) for item_id in response.get("ranked_item_ids", []) if str(item_id)]
    veto_ids = {str(item_id) for item_id in response.get("veto_item_ids", []) if str(item_id)}
    by_id = {result.get("item_id"): result for result in candidates}
    ordered = []
    for item_id in ranked_ids:
        result = by_id.get(item_id)
        if result is not None and item_id not in veto_ids:
            ordered.append(result)
    ordered_ids = {result.get("item_id") for result in ordered}
    ordered.extend(result for result in candidates if result.get("item_id") not in ordered_ids and result.get("item_id") not in veto_ids)
    ordered.extend(result for result in candidates if result.get("item_id") in veto_ids)
    reranked = row_with_reranked_results(row, ordered[:top_k], ranking_key=f"llm_rerank@{rerank_depth}")
    reranked["llm_rerank_reason"] = response.get("reason", "")
    reranked["llm_veto_item_ids"] = sorted(veto_ids)
    reranked["llm_payload_chars"] = len(json.dumps(payload, ensure_ascii=False))
    return reranked


def llm_rerank_payload(row: dict[str, Any], *, rerank_depth: int) -> dict[str, Any]:
    candidates = row.get("top_results", [])[:rerank_depth]
    return {
        "user_input": row.get("user_input", ""),
        "target_stage": row.get("target_stage"),
        "target_purposes": row.get("target_purposes", []),
        "instruction": (
            "Rank candidates by whether they satisfy the positive request, avoid forbidden/style risks, "
            "and are reusable as the best scene experience card. Return JSON only."
        ),
        "candidates": [
            reranker_candidate_summary(result, rank=index + 1)
            for index, result in enumerate(candidates)
        ],
        "output_schema": {"ranked_item_ids": ["item_id"], "veto_item_ids": ["item_id"], "reason": "short string"},
    }


def reranker_candidate_summary(result: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata", {}), dict) else {}
    summary = {
        "item_id": result.get("item_id"),
        "rank": rank,
        "score": result.get("score"),
        "script_stage": metadata.get("script_stage"),
        "creative_purpose": metadata.get("creative_purpose", []),
        "script_usecase": metadata.get("script_usecase") or {
            "best_usage": metadata.get("script_usecase_best_usage"),
            "risk": metadata.get("script_usecase_risk"),
            "sentence": metadata.get("script_use_sentence"),
        },
        "scene_signature": result.get("scene_signature") or metadata.get("scene_signature") or {},
        "style_traits": metadata.get("style_traits", []),
        "style_risks": metadata.get("style_risks", []),
        "constraint_hits": result.get("constraint_hits", {}),
    }
    return {key: value for key, value in summary.items() if value not in (None, {}, [])}


def row_with_reranked_results(row: dict[str, Any], top_results: list[dict[str, Any]], *, ranking_key: str) -> dict[str, Any]:
    target_id = row.get("target_item_id")
    target_rank = None
    copied_results = []
    for rank, result in enumerate(top_results, start=1):
        result = dict(result)
        result["ranking_key"] = ranking_key
        copied_results.append(result)
        if result.get("item_id") == target_id:
            target_rank = rank
    target_stage = row.get("target_stage")
    target_purposes = set(row.get("target_purposes", []))
    return {
        **row,
        "ranking_key": ranking_key,
        "target_rank": target_rank,
        "target_score": score_of(top_results, target_id),
        "stage_hit_at_1": bool(copied_results and result_stage(copied_results[0]) == target_stage),
        "stage_hit_at_3": any(result_stage(result) == target_stage for result in copied_results[:3]),
        "purpose_hit_at_3": purpose_hit_at(copied_results, target_purposes, 3),
        "top_results": copied_results,
    }


def truncate_ranked_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    truncated = []
    for row in rows:
        top_results = row.get("top_results", [])[:top_k]
        truncated.append(row_with_reranked_results(row, top_results, ranking_key=row.get("ranking_key", top_results[0].get("ranking_key", "")) if top_results else row.get("ranking_key", "")))
    return truncated


def attach_variant_metadata(rows: list[dict[str, Any]], variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_id = {variant["case_id"]: variant for variant in variants}
    enriched = []
    for row in rows:
        variant = by_case_id.get(row["case_id"], {})
        enriched.append(
            {
                **row,
                "source_case_id": variant.get("source_case_id", row.get("source_case_id")),
                "variant_type": variant.get("variant_type", row.get("variant_type")),
                "expected_granularity": variant.get("expected_granularity", row.get("expected_granularity")),
            }
        )
    return enriched


def strong_baseline_selection_score(metrics: dict[str, Any]) -> list[float]:
    return [
        float(metrics.get("nDCG@10", 0.0)),
        float(metrics.get("nDCG@3", 0.0)),
        float(metrics.get("MRR@10", 0.0)),
        -float(metrics.get("Unjudged@10", 0.0)),
    ]


def experiment_metadata(args: argparse.Namespace, summary: dict[str, Any], started_at: float) -> dict[str, Any]:
    return {
        "command": getattr(args, "command", ""),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"output", "markdown_output"}
        },
        "git_sha": git_sha(),
        "index_fingerprint": file_fingerprint(getattr(args, "index", DEFAULT_INDEX_PATH)),
        "cache_fingerprint": file_fingerprint(getattr(args, "cache", DEFAULT_CACHE_PATH)),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summary,
    }


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def file_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def failure_analysis_row(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    row: dict[str, Any],
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
) -> dict[str, Any]:
    target_id = row.get("target_item_id")
    target_index = prepared.item_index_by_id.get(target_id) if target_id else None
    top1 = row.get("top_results", [{}])[0] if row.get("top_results") else {}
    target_scores = (
        component_scores_for_index(prepared, signal, target_index, constraint_profile=constraint_profile)
        if target_index is not None
        else {}
    )
    top1_index = prepared.item_index_by_id.get(top1.get("item_id"))
    top1_scores = (
        component_scores_for_index(prepared, signal, top1_index, constraint_profile=constraint_profile)
        if top1_index is not None
        else {}
    )
    failure_type = classify_failure(row, signal, target_scores, top1)
    return {
        "case_id": row["case_id"],
        "user_input": row["user_input"],
        "failure_type": failure_type,
        "target_item_id": target_id,
        "target_rank": row.get("target_rank"),
        "target_in_top100": bool(row.get("target_rank") and row["target_rank"] <= 100),
        "target_scores": target_scores,
        "top1_item_id": top1.get("item_id"),
        "top1_scores": top1_scores,
        "top1_metadata": top1.get("metadata", {}),
        "top1_reason": explain_top1_win(row, top1, target_scores, top1_scores),
        "ranking_key": ranking_key,
    }


def component_scores_for_index(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    item_index: int | None,
    *,
    constraint_profile: dict[str, Any],
) -> dict[str, float]:
    if item_index is None:
        return {}
    ensure_fast_lexical(prepared, signal)
    ensure_fast_rrf(prepared, signal)
    ensure_fast_signature(prepared, signal)
    ensure_fast_constraints(prepared, signal, constraint_profile=constraint_profile)
    return {
        "semantic": round(float(signal.semantic_scores[item_index]), 6),
        "lexical": round(float(signal.lexical_scores[item_index]), 6) if signal.lexical_scores is not None else 0.0,
        "rrf": round(float(signal.rrf_scores[item_index] * 100), 6) if signal.rrf_scores is not None else 0.0,
        "constraint": round(float(signal.constraint_scores[item_index]), 6) if signal.constraint_scores is not None else 0.0,
        "signature": round(float(signal.signature_scores[item_index]), 6) if signal.signature_scores is not None else 0.0,
    }


def classify_failure(
    row: dict[str, Any],
    signal: FastCaseSignals,
    target_scores: dict[str, float],
    top1: dict[str, Any],
) -> str:
    if row.get("target_rank") is None or row.get("target_rank", 999999) > 100:
        return "candidate_recall_failure"
    if top1.get("constraint_hits", {}).get("negative_style"):
        return "style_risk_miss"
    if top1.get("constraint_hits", {}).get("forbidden_stage"):
        return "constraint_failure"
    ambiguity = getattr(signal.query_plan, "ambiguity", {}) or {}
    if ambiguity.get("level") == "high" or (not getattr(signal.query_plan, "desired_stage", []) and not target_scores):
        return "query_understanding_failure"
    if row.get("target_rank") and row["target_rank"] <= 10:
        return "fusion_ranking_failure"
    if float(target_scores.get("semantic", 0.0)) <= 0:
        return "candidate_recall_failure"
    return "weak_target_label"


def explain_top1_win(
    row: dict[str, Any],
    top1: dict[str, Any],
    target_scores: dict[str, float],
    top1_scores: dict[str, float],
) -> str:
    if top1.get("constraint_hits", {}).get("negative_style"):
        return "top1 still has negative style hits"
    if result_stage(top1) == row.get("target_stage") and top1_scores.get("signature", 0.0) >= target_scores.get("signature", 0.0):
        return "top1 matches target stage and has equal or better signature score"
    if top1_scores.get("rrf", 0.0) > target_scores.get("rrf", 0.0):
        return "top1 won by RRF fusion score"
    if top1_scores.get("semantic", 0.0) > target_scores.get("semantic", 0.0):
        return "top1 won by semantic score"
    return "top1 score margin is small or target label may be ambiguous"


def build_style_risk_case(case: dict[str, Any]) -> dict[str, Any]:
    target = case["target"]
    stage = stage_word(target.get("script_stage", ""))
    purposes = purpose_text(target.get("creative_purpose", []))
    return {
        **case,
        "case_id": f"{case['case_id']}__style_risk",
        "case_type": "style_risk_negative",
        "variant_type": "style_risk_negative",
        "source_case_id": case["case_id"],
        "user_input": (
            f"需要一个{stage}段落，服务{purposes}，要真实、有人味、有现场感。"
            "不要大厂味，不要广告感，不要汇报片，不要世界500强品牌质感，不要产品卖点堆叠，也不要冷科技。"
        ),
    }


def rerank_gate_decision(row: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    margin = case_margin(row)
    if margin < 0.08:
        reasons.append("low_margin")
    if row.get("planner_confidence", 1.0) < 0.6:
        reasons.append("low_planner_confidence")
    if row.get("query_constraints", {}).get("negative_constraints") or row.get("query_constraints", {}).get("forbidden_stage"):
        reasons.append("has_negative_constraints")
    if style_violation_at([row], 3) > 0:
        reasons.append("style_risk")
    if row.get("target_rank") and row["target_rank"] <= 20 and row["target_rank"] > 3:
        reasons.append("target_near_miss")
    return {
        "case_id": row["case_id"],
        "user_input": row["user_input"],
        "target_rank": row.get("target_rank"),
        "top1_top2_margin": round(margin, 6),
        "should_rerank": bool(reasons),
        "gate_reasons": reasons,
        "top_results": row.get("top_results", [])[:3],
    }


def query_understanding_summary(rows: list[dict[str, Any]], planner_stats: dict[str, Any]) -> dict[str, Any]:
    metrics = build_metrics(rows).get("overall", {})
    margin = build_margin_report(rows)
    return {
        "target_recall_at_1": metrics.get("recall_at_1", 0.0),
        "target_recall_at_3": metrics.get("recall_at_3", 0.0),
        "target_recall_at_10": metrics.get("recall_at_10", 0.0),
        "stage_hit_at_1": average_bool(rows, "stage_hit_at_1"),
        "stage_hit_at_3": average_bool(rows, "stage_hit_at_3"),
        "purpose_hit_at_3": average_bool(rows, "purpose_hit_at_3"),
        "style_violation_at_3": style_violation_at(rows, 3),
        "low_confidence_rate": margin["low_confidence_rate"],
        "mean_top1_top2_margin": margin["mean_top1_top2_margin"],
        "planner_cache_hit_rate": planner_stats.get("cache_hit_rate", 0.0),
        "llm_call_count": planner_stats.get("llm_call_count", 0),
        "planner_fallback_count": planner_stats.get("fallback_count", 0),
        "negative_leak_rate": planner_stats.get("negative_leak_rate", 0.0),
    }


def query_understanding_selection_score(summary: dict[str, Any]) -> list[float]:
    return [
        float(summary.get("target_recall_at_10", 0.0)),
        float(summary.get("stage_hit_at_3", 0.0)),
        float(summary.get("purpose_hit_at_3", 0.0)),
        -float(summary.get("style_violation_at_3", 0.0)),
        -float(summary.get("negative_leak_rate", 0.0)),
    ]


def query_understanding_delta(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = (
        "target_recall_at_1",
        "target_recall_at_3",
        "target_recall_at_10",
        "stage_hit_at_3",
        "purpose_hit_at_3",
        "style_violation_at_3",
        "low_confidence_rate",
        "negative_leak_rate",
    )
    return {
        key: round(float(summary.get(key, 0.0)) - float(baseline.get(key, 0.0)), 6)
        for key in keys
    }


def summarize_fuzzy_rows(rows: list[dict[str, Any]], planner_stats: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "scene_level_recall_at_1": 0.0,
            "scene_level_recall_at_3": 0.0,
            "scene_level_recall_at_10": 0.0,
            "stage_level_hit_at_1": 0.0,
            "stage_level_hit_at_3": 0.0,
            "purpose_level_hit_at_3": 0.0,
            "style_violation_at_3": 0.0,
            "low_confidence_rate": 0.0,
            "negative_leak_rate": planner_stats.get("negative_leak_rate", 0.0),
        }
    return {
        "count": len(rows),
        "scene_level_recall_at_1": recall_at(rows, 1),
        "scene_level_recall_at_3": recall_at(rows, 3),
        "scene_level_recall_at_10": recall_at(rows, 10),
        "stage_level_hit_at_1": average_bool(rows, "stage_hit_at_1"),
        "stage_level_hit_at_3": average_bool(rows, "stage_hit_at_3"),
        "purpose_level_hit_at_3": average_bool(rows, "purpose_hit_at_3"),
        "style_violation_at_3": style_violation_at(rows, 3),
        "low_confidence_rate": round(sum(1 for row in rows if row.get("confidence") == "low") / len(rows), 6),
        "mean_top1_top2_margin": round(mean([float(row.get("top1_top2_margin", 0.0)) for row in rows]), 6),
        "planner_cache_hit_rate": planner_stats.get("cache_hit_rate", 0.0),
        "llm_call_count": planner_stats.get("llm_call_count", 0),
        "planner_fallback_count": planner_stats.get("fallback_count", 0),
        "negative_leak_rate": planner_stats.get("negative_leak_rate", 0.0),
    }


def average_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get(key)) / len(rows), 6)


def workflow_selection_score(metrics: dict[str, Any]) -> list[float]:
    overall = metrics.get("overall", {})
    by_type = metrics.get("by_case_type", {})
    hard_negative = by_type.get("hard_negative", {})
    return [
        float(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", 0.0)),
        float(overall.get("recall_at_10", 0.0)),
        -float(overall.get("forbidden_stage_violation_at_3", 0.0)),
    ]


def workflow_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    overall = metrics.get("overall", {})
    base_overall = baseline.get("overall", {})
    keys = ("recall_at_1", "recall_at_3", "recall_at_10", "forbidden_stage_violation_at_3")
    return {
        key: round(float(overall.get(key, 0.0)) - float(base_overall.get(key, 0.0)), 6)
        for key in keys
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Mock Retrieval Evaluation Report",
        "",
        f"- method: `{report.get('method', '')}`",
        f"- split: `{report.get('split', '')}`",
        f"- case_count: `{report.get('case_count', 0)}`",
        f"- top_k: `{report.get('top_k', 0)}`",
        "",
    ]
    summary = report.get("summary", {})
    if summary:
        lines.extend(["## Summary", ""])
        if summary.get("best_workflow") is not None:
            lines.append(f"- best_workflow: `{summary.get('best_workflow', '')}`")
        if summary.get("best_selection_score") is not None:
            lines.append(f"- best_selection_score: `{summary.get('best_selection_score', '')}`")
        metric_rows = simple_metric_rows(summary, skip={"best_workflow", "best_selection_score"})
        if metric_rows:
            lines.extend(["", "| metric | value |", "|---|---:|"])
            lines.extend(f"| {key} | {value} |" for key, value in metric_rows)
        lines.append("")
    metrics = report.get("metrics", {})
    if metrics:
        overall_metrics = metrics.get("overall", metrics)
        metric_rows = simple_metric_rows(overall_metrics)
        if metric_rows:
            lines.extend(["## Metrics", "", "| metric | value |", "|---|---:|"])
            lines.extend(f"| {key} | {value} |" for key, value in metric_rows)
            lines.append("")
    workflows = report.get("workflows", {})
    if workflows:
        lines.extend(["## Workflow Metrics", "", "| workflow | recall@1 | recall@3 | recall@10 | hard_neg_prefer | forbidden@3 |", "|---|---:|---:|---:|---:|---:|"])
        for workflow, row in workflows.items():
            metrics = row.get("metrics", {})
            overall = metrics.get("overall", {})
            hard_negative = metrics.get("by_case_type", {}).get("hard_negative", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        workflow,
                        str(overall.get("recall_at_1", "")),
                        str(overall.get("recall_at_3", "")),
                        str(overall.get("recall_at_10", "")),
                        str(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", "")),
                        str(overall.get("forbidden_stage_violation_at_3", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
        if any(row.get("summary") for row in workflows.values()):
            lines.extend(
                [
                    "## Workflow Summary",
                    "",
                    "| workflow | scene@1 | scene@3 | scene@10 | stage@3 | purpose@3 | style_violation@3 | low_conf |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for workflow, row in workflows.items():
                workflow_summary = row.get("summary", {})
                if not workflow_summary:
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            workflow,
                            str(workflow_summary.get("target_recall_at_1", "")),
                            str(workflow_summary.get("target_recall_at_3", "")),
                            str(workflow_summary.get("target_recall_at_10", "")),
                            str(workflow_summary.get("stage_hit_at_3", "")),
                            str(workflow_summary.get("purpose_hit_at_3", "")),
                            str(workflow_summary.get("style_violation_at_3", "")),
                            str(workflow_summary.get("low_confidence_rate", "")),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    planners = report.get("planners", {})
    if planners:
        lines.extend(
            [
                "## Query Understanding Metrics",
                "",
                "| planner | scene@1 | scene@3 | scene@10 | stage@3 | purpose@3 | style_violation@3 | low_conf | neg_leak | llm_calls | fallback |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for planner, row in planners.items():
            summary = row.get("planner_summary", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        planner,
                        str(summary.get("target_recall_at_1", "")),
                        str(summary.get("target_recall_at_3", "")),
                        str(summary.get("target_recall_at_10", "")),
                        str(summary.get("stage_hit_at_3", "")),
                        str(summary.get("purpose_hit_at_3", "")),
                        str(summary.get("style_violation_at_3", "")),
                        str(summary.get("low_confidence_rate", "")),
                        str(summary.get("negative_leak_rate", "")),
                        str(summary.get("llm_call_count", "")),
                        str(summary.get("planner_fallback_count", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    baselines = report.get("baselines", {})
    if baselines:
        lines.extend(
            [
                "## Baseline Metrics",
                "",
                "| baseline | nDCG@3 | nDCG@10 | ERR@10 | MRR@10 | Judged@10 | Recall@10 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in baselines.items():
            metrics = row.get("graded_metrics", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        str(metrics.get("nDCG@3", "")),
                        str(metrics.get("nDCG@10", "")),
                        str(metrics.get("ERR@10", "")),
                        str(metrics.get("MRR@10", "")),
                        str(metrics.get("Judged@10", "")),
                        str(metrics.get("Recall@10", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    by_variant_type = report.get("by_variant_type", {})
    if by_variant_type:
        lines.extend(
            [
                "## Variant Metrics",
                "",
                "| variant | nDCG@3 | nDCG@10 | ERR@10 | MRR@10 | scene@10 | stage@3 | purpose@3 | style_violation@3 | low_conf |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, summary in by_variant_type.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        variant,
                        str(summary.get("nDCG@3", "")),
                        str(summary.get("nDCG@10", "")),
                        str(summary.get("ERR@10", "")),
                        str(summary.get("MRR@10", "")),
                        str(summary.get("scene_level_recall_at_10", summary.get("target_recall_at_10", ""))),
                        str(summary.get("stage_level_hit_at_3", summary.get("stage_hit_at_3", ""))),
                        str(summary.get("purpose_level_hit_at_3", summary.get("purpose_hit_at_3", ""))),
                        str(summary.get("style_violation_at_3", "")),
                        str(summary.get("low_confidence_rate", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- LLM judgement is not run by default; use explicit sample flags for token-safe spot checks.",
            "- Treat this report as an experiment comparison, not a production default decision by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def simple_metric_rows(data: dict[str, Any], *, skip: set[str] | None = None) -> list[tuple[str, str]]:
    skip = skip or set()
    rows = []
    for key, value in data.items():
        if key in skip or isinstance(value, (dict, list)):
            continue
        rows.append((key, str(value)))
    return rows


def validate_paraphrase_stress_command(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    source_cases = [
        case
        for case in read_cases(args.inputs, 0, split=args.split)
        if case["case_type"] == args.case_type and case["expected_relation"] == "should_match"
    ]
    if args.limit > 0:
        source_cases = source_cases[: args.limit]
    variants = [
        variant
        for case in source_cases
        for variant in build_paraphrase_variants(case)
    ]
    variants = filter_variants_by_type(variants, args.variant_types)
    if args.dry_run:
        return {
            "method": "mock_paraphrase_stress",
            "dry_run": True,
            "split": args.split,
            "case_type": args.case_type,
            "source_case_count": len(source_cases),
            "variant_count": len(variants),
            "summary": {
                "source_case_count": len(source_cases),
                "variant_count": len(variants),
                "variant_types": sorted({variant["variant_type"] for variant in variants}),
            },
            "cases": variants,
        }

    profile = load_constraint_profile(args.constraint_profile)
    index = read_index(args.index)
    cache = make_embedding_cache(args)
    plan_result = plan_cases_from_args(variants, args=args, planner=args.query_planner)
    query_texts = query_texts_for_plans(plan_result.plans, args)
    cache.embed_texts(query_texts)

    prepared = prepare_mock_index(index)
    signals = precompute_fast_case_signals(
        prepared,
        cache,
        variants,
        constraint_profile=profile,
        query_plans=plan_result.plans,
        max_query_texts=max_query_texts_arg(args),
    )
    rows = []
    for variant, signal in zip(variants, signals):
        scores, ranked_indices = fast_rank_indices_for_key(
            prepared,
            signal,
            ranking_key=args.ranking_key,
            constraint_profile=profile,
        )
        top_indices = ranked_indices[: args.top_k]
        top_results = [
            fast_result_row(
                prepared,
                signal,
                item_index,
                score=float(scores[item_index]),
                ranking_key=args.ranking_key,
                include_debug_text=args.include_debug_text,
            )
            for item_index in top_indices
        ]
        target_id = target_item_id(variant["target"])
        target_stage = canonical_stage(variant["target"].get("script_stage"))
        target_purposes = set(variant["target"].get("creative_purpose", []))
        rows.append(
            {
                "case_id": variant["case_id"],
                "source_case_id": variant["source_case_id"],
                "variant_type": variant["variant_type"],
                "user_input": variant["user_input"],
                "target_item_id": target_id,
                "target_stage": target_stage,
                "target_purposes": list(target_purposes),
                "target_rank": rank_of_prepared_id(prepared, ranked_indices, target_id),
                "target_score": score_of_prepared_id(prepared, scores, target_id),
                "stage_hit_at_1": bool(top_results and result_stage(top_results[0]) == target_stage),
                "stage_hit_at_3": any(result_stage(result) == target_stage for result in top_results[:3]),
                "purpose_hit_at_3": purpose_hit_at(top_results, target_purposes, 3),
                "top1_top2_margin": round(top1_top2_margin({"top_results": top_results}), 6),
                "confidence": confidence_bucket(top1_top2_margin({"top_results": top_results})),
                "top_results": top_results,
                "query_planner": signal.query_plan.planner,
                "planner_confidence": signal.query_plan.confidence,
                "planner_negative_leak": plan_has_negative_leak(signal.query_plan),
            }
        )
        if args.include_planner_debug:
            rows[-1]["query_plan"] = signal.query_plan.to_dict()
            rows[-1]["query_channels"] = signal.query_channels
    return {
        "method": "mock_paraphrase_stress",
        "split": args.split,
        "query_planner": args.query_planner,
        "planner_stats": plan_result.stats,
        "embedding_cache": cache.cache_report(),
        "case_type": args.case_type,
        "source_case_count": len(source_cases),
        "variant_count": len(rows),
        "top_k": args.top_k,
        "ranking_key": args.ranking_key,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "summary": summarize_paraphrase_rows(rows),
        "by_variant_type": {
            variant_type: summarize_paraphrase_rows([row for row in rows if row["variant_type"] == variant_type])
            for variant_type in sorted({row["variant_type"] for row in rows})
        },
        "by_source_case": summarize_paraphrase_by_source_case(rows),
        "cases": compact_case_rows(rows, include_debug_text=args.include_debug_text),
    }


def evaluate_cases(
    *,
    index_path: Path,
    cache_path: Path,
    model: str,
    dimension: int,
    embedding_batch_size: int,
    inputs_path: Path,
    limit: int,
    split: str,
    top_k: int,
    constraint_profile: dict[str, Any] | None,
    constraint_profile_path: Path,
    constraints_enabled: bool,
    include_cases: bool = True,
) -> dict[str, Any]:
    index = read_index(index_path)
    cache = EmbeddingCache(
        cache_path=cache_path,
        model=model,
        dimension=dimension,
        batch_size=embedding_batch_size,
    )
    cases = read_cases(inputs_path, limit, split=split)
    query_texts = [channel["text"] for case in cases for channel in build_query_channels(case["user_input"])]
    cache.embed_texts(query_texts)
    result = evaluate_loaded_cases(
        index=index,
        cache=cache,
        cases=cases,
        top_k=top_k,
        constraint_profile=constraint_profile,
        constraints_enabled=constraints_enabled,
        include_cases=include_cases,
    )
    result["split"] = split
    result["constraint_profile"] = str(constraint_profile_path)
    return result


def evaluate_loaded_cases(
    *,
    index: dict[str, Any],
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
    top_k: int,
    constraint_profile: dict[str, Any] | None,
    constraints_enabled: bool,
    include_cases: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    case_results = []
    for case in cases:
        query_channels = build_query_channels(case["user_input"])
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = search_index(
            index,
            query_channels,
            cache,
            top_k=max(top_k, len(index["items"])),
            user_input=case["user_input"],
            constraint_profile=constraint_profile,
            constraints_enabled=constraints_enabled,
        )
        target_id = target_item_id(case["target"])
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_rank = rank_of(ranked, target_id)
        expected_prefer_rank = rank_of(ranked, expected_prefer_id) if expected_prefer_id else None
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id) if expected_prefer_id else None
        case_results.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "user_input": case["user_input"],
                "query_constraints": query_constraints,
                "target_item_id": target_id,
                "target_rank": target_rank,
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": expected_prefer_rank,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": ranked[:top_k],
            }
        )
    elapsed = round(time.perf_counter() - started_at, 3)
    result = {
        "method": "mock_multi_channel_embedding_retrieval",
        "case_count": len(case_results),
        "top_k": top_k,
        "constraints_enabled": constraints_enabled,
        "elapsed_seconds": elapsed,
        "seconds_per_case": round(elapsed / max(1, len(case_results)), 6),
        "metrics": build_metrics(case_results),
    }
    if include_cases:
        result["cases"] = case_results
    return result


def read_cases(path: Path, limit: int, *, split: str = "all") -> list[dict[str, Any]]:
    cases = split_cases(json.loads(path.read_text(encoding="utf-8"))["cases"], split)
    if limit > 0:
        return cases[:limit]
    return cases


def split_cases(cases: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if split == "all":
        return list(cases)
    return [case for case in cases if case_split(case["case_id"]) == split]


def case_split(case_id: str) -> str:
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket <= 3:
        return "dev"
    if bucket <= 7:
        return "test"
    return "hidden"


def filter_cases_by_fixture(
    cases: list[dict[str, Any]],
    fixture_id: str,
    *,
    include: bool = True,
) -> list[dict[str, Any]]:
    return [case for case in cases if (case_fixture_id(case) == fixture_id) is include]


def case_fixture_id(case: dict[str, Any]) -> str:
    return str(case["target"]["fixture_id"])


def plan_cases_from_args(
    cases: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    planner: str,
) -> Any:
    return plan_inputs_from_args(
        [case["user_input"] for case in cases],
        args=args,
        planner=planner,
    )


def plan_inputs_from_args(
    user_inputs: list[str],
    *,
    args: argparse.Namespace | None = None,
    planner: str = "rule",
) -> Any:
    return plan_queries(
        user_inputs,
        planner=planner,
        cache_path=getattr(args, "planner_cache", None) if args is not None else None,
        llm_sample_size=getattr(args, "llm_planner_sample_size", 0) if args is not None else 0,
        timeout_seconds=getattr(args, "planner_timeout_seconds", 60.0) if args is not None else 60.0,
        retries=getattr(args, "planner_retries", 0) if args is not None else 0,
        require_llm=getattr(args, "require_llm_planner", False) if args is not None else False,
        include_debug=getattr(args, "include_planner_debug", False) if args is not None else False,
    )


def parse_query_planner_list(value: str) -> list[str]:
    planners = [part.strip() for part in value.split(",") if part.strip()]
    if not planners:
        raise ValueError("At least one query planner is required.")
    unknown = [planner for planner in planners if planner not in VALID_QUERY_PLANNERS]
    if unknown:
        raise ValueError(f"Unknown query planner(s): {', '.join(unknown)}")
    return planners


def parse_ranking_key_list(value: str) -> list[str]:
    keys = [part.strip() for part in str(value or "").split(",") if part.strip()]
    if not keys:
        raise ValueError("At least one ranking key is required.")
    unknown = [key for key in keys if key not in RANKING_KEYS]
    if unknown:
        raise ValueError(f"Unknown ranking key(s): {', '.join(unknown)}")
    return keys


def parse_variant_types(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def filter_variants_by_type(variants: list[dict[str, Any]], value: str) -> list[dict[str, Any]]:
    allowed = parse_variant_types(value)
    if not allowed:
        return variants
    return [variant for variant in variants if variant.get("variant_type") in allowed]


def empty_query_constraints() -> dict[str, Any]:
    return {"desired_stage": [], "forbidden_stage": [], "negative_constraints": [], "visual_hints": []}


def query_channels_for_plan_args(plan: ExperimentalQueryPlan, args: argparse.Namespace | None) -> list[dict[str, Any]]:
    return build_query_channels_for_plan(
        plan,
        max_query_texts=max_query_texts_arg(args),
    )


def max_query_texts_arg(args: argparse.Namespace | None) -> int:
    return max(0, int(getattr(args, "max_query_texts", 0) or 0)) if args is not None else 0


def query_texts_for_plans(plans: list[ExperimentalQueryPlan], args: argparse.Namespace | None) -> list[str]:
    return [
        channel["text"]
        for plan in plans
        for channel in query_channels_for_plan_args(plan, args)
    ]


def prepare_mock_index(index: dict[str, Any]) -> PreparedMockIndex:
    items = index.get("items", [])
    item_ids = [item["item_id"] for item in items]
    metadata = [item.get("metadata", {}) for item in items]
    lexical_texts = [index_item_lexical_text(item) for item in items]
    lexical_text_lowers = [text.lower() for text in lexical_texts]
    constraint_text_lowers = [constraint_text_for_metadata(item.get("metadata", {})).lower() for item in items]
    scene_signatures = [
        infer_scene_signature(metadata_row, lexical_text)
        for metadata_row, lexical_text in zip(metadata, lexical_texts)
    ]
    signature_token_sets = [signature_tokens(signature) for signature in scene_signatures]
    stage_values = np.asarray(
        [canonical_stage(item.get("metadata", {}).get("script_stage", "")) for item in items],
        dtype=object,
    )
    style_masks = {
        style: np.asarray(
            [
                style in text or any(alias.lower() in text for alias in aliases)
                for text in lexical_text_lowers
            ],
            dtype=bool,
        )
        for style, aliases in STYLE_ALIASES.items()
    }
    channel_names = sorted(
        {
            channel.get("channel")
            for item in items
            for channel in item.get("channels", [])
            if isinstance(channel, dict) and channel.get("channel")
        }
    )
    dimension = next(
        (
            len(channel.get("embedding", []))
            for item in items
            for channel in item.get("channels", [])
            if channel.get("embedding")
        ),
        0,
    )
    channel_vectors = {
        channel_name: normalized_channel_matrix(items, channel_name, dimension)
        for channel_name in channel_names
    }
    bm25_inverted, bm25_doc_norms = build_prepared_bm25(lexical_texts)
    return PreparedMockIndex(
        item_ids=item_ids,
        item_index_by_id={item_id: index for index, item_id in enumerate(item_ids)},
        metadata=metadata,
        lexical_texts=lexical_texts,
        lexical_text_lowers=lexical_text_lowers,
        constraint_text_lowers=constraint_text_lowers,
        scene_signatures=scene_signatures,
        signature_token_sets=signature_token_sets,
        stage_values=stage_values,
        style_masks=style_masks,
        channel_vectors=channel_vectors,
        bm25_inverted=bm25_inverted,
        bm25_doc_norms=bm25_doc_norms,
        bm25_doc_count=len(items),
    )


SIGNATURE_FIELD_WEIGHTS = {
    "people": 0.9,
    "place": 0.9,
    "action": 0.8,
    "objects": 0.7,
    "emotion_function": 1.1,
    "narrative_position": 1.2,
    "camera_experience": 0.5,
    "script_reuse_pattern": 1.0,
}


SIGNATURE_ALIASES = {
    "people": (
        "人",
        "人物",
        "医生",
        "工程师",
        "员工",
        "团队",
        "客户",
        "用户",
        "worker",
        "doctor",
        "engineer",
        "team",
        "customer",
    ),
    "place": (
        "现场",
        "办公室",
        "工厂",
        "实验室",
        "诊室",
        "医院",
        "会议室",
        "城市",
        "real location",
        "office",
        "factory",
        "lab",
        "hospital",
    ),
    "action": (
        "沟通",
        "讨论",
        "操作",
        "检查",
        "复核",
        "协作",
        "进入",
        "观察",
        "communication",
        "check",
        "inspect",
        "work",
        "collaboration",
    ),
    "objects": (
        "屏幕",
        "设备",
        "系统",
        "数据",
        "记录",
        "标签",
        "文件",
        "仪器",
        "screen",
        "device",
        "system",
        "data",
        "record",
    ),
    "camera_experience": (
        "特写",
        "远景",
        "跟拍",
        "手持",
        "镜头",
        "close",
        "wide",
        "tracking",
        "handheld",
        "shot",
    ),
}


def infer_scene_signature(metadata: dict[str, Any], lexical_text: str) -> dict[str, list[str]]:
    text = " ".join(
        str(part)
        for part in (
            lexical_text,
            metadata.get("script_use_sentence", ""),
            metadata.get("style", ""),
            metadata.get("industry", ""),
            " ".join(metadata.get("creative_purpose", [])),
            metadata.get("script_stage", ""),
        )
        if part
    )
    signature = {
        "people": alias_hits(text, SIGNATURE_ALIASES["people"]),
        "place": alias_hits(text, SIGNATURE_ALIASES["place"]),
        "action": alias_hits(text, SIGNATURE_ALIASES["action"]),
        "objects": alias_hits(text, SIGNATURE_ALIASES["objects"]),
        "emotion_function": list(metadata.get("creative_purpose", [])),
        "narrative_position": [canonical_stage(metadata.get("script_stage", ""))],
        "camera_experience": alias_hits(text, SIGNATURE_ALIASES["camera_experience"]),
        "script_reuse_pattern": tokenize_signature_text(metadata.get("script_use_sentence", "") or lexical_text)[:16],
    }
    signature = {key: dedupe_values(values) for key, values in signature.items()}
    if not any(signature.values()):
        signature["script_reuse_pattern"] = tokenize_signature_text(text)[:16]
    return signature


def query_scene_signature(plan: Any) -> dict[str, list[str]]:
    text = " ".join(
        str(part)
        for part in (
            getattr(plan, "positive_query", ""),
            getattr(plan, "original_text", ""),
            " ".join(getattr(plan, "visual_hints", []) or []),
            " ".join(getattr(plan, "positive_purposes", []) or []),
            " ".join(getattr(plan, "positive_style", []) or []),
        )
        if part
    )
    scene_signature = getattr(plan, "scene_signature", {}) or {}
    signature = {
        "people": _string_values(scene_signature.get("people")) + alias_hits(text, SIGNATURE_ALIASES["people"]),
        "place": _string_values(scene_signature.get("place")) + alias_hits(text, SIGNATURE_ALIASES["place"]),
        "action": _string_values(scene_signature.get("actions")) + alias_hits(text, SIGNATURE_ALIASES["action"]),
        "objects": _string_values(scene_signature.get("objects")) + alias_hits(text, SIGNATURE_ALIASES["objects"]),
        "emotion_function": _string_values(scene_signature.get("emotional_function"))
        + list(getattr(plan, "positive_purposes", []) or []),
        "narrative_position": list(getattr(plan, "desired_stage", []) or []),
        "camera_experience": alias_hits(text, SIGNATURE_ALIASES["camera_experience"]),
        "script_reuse_pattern": tokenize_signature_text(getattr(plan, "positive_query", "") or text)[:16],
    }
    signature = {key: dedupe_values(values) for key, values in signature.items()}
    if not any(signature.values()):
        signature["script_reuse_pattern"] = tokenize_signature_text(text)[:16]
    return signature


def alias_hits(text: str, aliases: tuple[str, ...]) -> list[str]:
    lower = str(text or "").lower()
    return [alias for alias in aliases if alias and alias.lower() in lower]


def tokenize_signature_text(text: str) -> list[str]:
    return [
        token
        for token in tokenize(str(text or ""))
        if len(token) > 1 and not token.isdigit()
    ]


def signature_tokens(signature: dict[str, list[str]]) -> dict[str, set[str]]:
    return {field: set(tokenize_signature_text(" ".join(values))) for field, values in signature.items()}


def _string_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def dedupe_values(values: list[str]) -> list[str]:
    result = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def constraint_text_for_metadata(metadata: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in (
            metadata.get("script_stage", ""),
            metadata.get("script_use_sentence", ""),
            " ".join(metadata.get("creative_purpose", [])),
            metadata.get("style", ""),
            metadata.get("industry", ""),
        )
        if value
    )


def normalized_channel_matrix(items: list[dict[str, Any]], channel_name: str, dimension: int) -> np.ndarray:
    matrix = np.zeros((len(items), dimension), dtype=np.float32)
    if dimension == 0:
        return matrix
    for item_index, item in enumerate(items):
        for channel in item.get("channels", []):
            if channel.get("channel") != channel_name:
                continue
            embedding = channel.get("embedding", [])
            if len(embedding) == dimension:
                matrix[item_index] = np.asarray(embedding, dtype=np.float32)
            break
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > 0
    matrix[nonzero] = matrix[nonzero] / norms[nonzero, None]
    return matrix


def build_prepared_bm25(
    lexical_texts: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[dict[str, list[tuple[int, int]]], np.ndarray]:
    documents = [tokenize(text) for text in lexical_texts]
    doc_lengths = np.asarray([len(document) for document in documents], dtype=np.float64)
    avg_length = float(doc_lengths.mean()) if len(doc_lengths) else 1.0
    if avg_length <= 0:
        avg_length = 1.0
    doc_norms = k1 * (1 - b + b * doc_lengths / avg_length)
    inverted: dict[str, list[tuple[int, int]]] = {}
    for doc_index, document in enumerate(documents):
        for term, tf in Counter(document).items():
            inverted.setdefault(term, []).append((doc_index, tf))
    return inverted, doc_norms


def precompute_fast_case_signals(
    prepared: PreparedMockIndex,
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
    *,
    constraint_profile: dict[str, Any],
    query_plans: list[ExperimentalQueryPlan] | None = None,
    max_query_texts: int = 0,
) -> list[FastCaseSignals]:
    if query_plans is not None and len(query_plans) != len(cases):
        raise ValueError("query_plans must have the same length as cases")
    signals = []
    for index, case in enumerate(cases):
        query_plan = query_plans[index] if query_plans is not None else plan_inputs_from_args([case["user_input"]]).plans[0]
        query_channels = build_query_channels_for_plan(query_plan, max_query_texts=max_query_texts)
        semantic_scores, channel_scores = score_query_fast(prepared, cache, query_channels)
        signals.append(
            FastCaseSignals(
                case=case,
                user_input=case["user_input"],
                semantic_scores=semantic_scores,
                channel_scores=channel_scores,
                query_constraints=planner_constraints(query_plan),
                query_plan=query_plan,
                query_channels=query_channels,
                query_signature=query_scene_signature(query_plan),
            )
        )
    return signals


def score_query_fast(
    prepared: PreparedMockIndex,
    cache: EmbeddingCache,
    query_channels: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total = np.zeros(prepared.bm25_doc_count, dtype=np.float64)
    channel_scores: dict[str, np.ndarray] = {}
    for query_channel in query_channels:
        if not query_channel.get("enabled", True):
            continue
        target_channel = query_channel.get("target_channel") or target_channel_for_query(query_channel["channel"])
        item_matrix = prepared.channel_vectors.get(target_channel)
        if item_matrix is None or item_matrix.size == 0:
            continue
        if hasattr(cache, "require_embedding_array"):
            query_vector = cache.require_embedding_array(query_channel["text"])
        else:
            query_vector = np.asarray(cache.require_embedding(query_channel["text"]), dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0:
            contribution = np.zeros(prepared.bm25_doc_count, dtype=np.float64)
        else:
            similarity = item_matrix @ (query_vector / query_norm)
            weight = float(query_channel.get("weight", DEFAULT_CHANNEL_WEIGHTS.get(target_channel, 0.0)))
            contribution = similarity.astype(np.float64) * weight
        channel_scores[target_channel] = channel_scores.get(
            target_channel,
            np.zeros(prepared.bm25_doc_count, dtype=np.float64),
        ) + contribution
        total += contribution
    return np.round(total, 6), {key: np.round(value, 6) for key, value in channel_scores.items()}


def rank_fast_cases(
    prepared: PreparedMockIndex,
    signals: list[FastCaseSignals],
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
    top_k: int,
    include_debug_text: bool = False,
    include_planner_debug: bool = False,
    include_all_results: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for signal in signals:
        scores, ranked_indices = fast_rank_indices_for_key(
            prepared,
            signal,
            ranking_key=ranking_key,
            constraint_profile=constraint_profile,
        )
        top_indices = ranked_indices[:top_k]
        top_results = [
            fast_result_row(
                prepared,
                signal,
                item_index,
                score=float(scores[item_index]),
                ranking_key=ranking_key,
                include_debug_text=include_debug_text,
            )
            for item_index in top_indices
        ]
        case = signal.case
        target_id = target_item_id(case["target"])
        target_stage = canonical_stage(case["target"].get("script_stage"))
        target_purposes = set(case["target"].get("creative_purpose", []))
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_rank = rank_of_prepared_id(prepared, ranked_indices, target_id)
        expected_prefer_rank = rank_of_prepared_id(prepared, ranked_indices, expected_prefer_id) if expected_prefer_id else None
        target_score = score_of_prepared_id(prepared, scores, target_id)
        expected_prefer_score = score_of_prepared_id(prepared, scores, expected_prefer_id) if expected_prefer_id else None
        rows.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "user_input": signal.user_input,
                "query_planner": signal.query_plan.planner,
                "planner_confidence": signal.query_plan.confidence,
                "query_constraints": signal.query_constraints,
                "target_item_id": target_id,
                "target_stage": target_stage,
                "target_purposes": list(target_purposes),
                "target_rank": target_rank,
                "target_score": target_score,
                "stage_hit_at_1": bool(top_results and result_stage(top_results[0]) == target_stage),
                "stage_hit_at_3": any(result_stage(result) == target_stage for result in top_results[:3]),
                "purpose_hit_at_3": purpose_hit_at(top_results, target_purposes, 3),
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": expected_prefer_rank,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": top_results,
            }
        )
        if include_all_results:
            rows[-1]["all_results"] = [
                fast_result_row(
                    prepared,
                    signal,
                    item_index,
                    score=float(scores[item_index]),
                    ranking_key=ranking_key,
                    include_debug_text=include_debug_text,
                )
                for item_index in ranked_indices
            ]
        if include_planner_debug:
            rows[-1]["query_plan"] = signal.query_plan.to_dict()
            rows[-1]["query_channels"] = signal.query_channels
    return rows


def fast_rank_indices_for_key(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
) -> tuple[np.ndarray, list[int]]:
    scores = fast_scores_for_key(prepared, signal, ranking_key=ranking_key, constraint_profile=constraint_profile)
    ranked_indices = sorted(range(prepared.bm25_doc_count), key=lambda index: (-float(scores[index]), index))
    return scores, ranked_indices


def rank_fast_items_for_key(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
    include_debug_text: bool = False,
) -> list[dict[str, Any]]:
    scores, ranked_indices = fast_rank_indices_for_key(
        prepared,
        signal,
        ranking_key=ranking_key,
        constraint_profile=constraint_profile,
    )
    return [
        fast_result_row(
            prepared,
            signal,
            item_index,
            score=float(scores[item_index]),
            ranking_key=ranking_key,
            include_debug_text=include_debug_text,
        )
        for item_index in ranked_indices
    ]


def fast_scores_for_key(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
) -> np.ndarray:
    if ranking_key == "final_score":
        scores = signal.semantic_scores + ensure_fast_constraints(prepared, signal, constraint_profile=constraint_profile)[0]
    elif ranking_key in {"embedding_only", "semantic_only"}:
        scores = signal.semantic_scores.copy()
    elif ranking_key == "lexical_only":
        scores = ensure_fast_lexical(prepared, signal).copy()
    elif ranking_key == "lexical_constraints":
        scores = ensure_fast_lexical(prepared, signal) + ensure_fast_constraints(
            prepared,
            signal,
            constraint_profile=constraint_profile,
        )[0]
    elif ranking_key == "lexical_constraints_signature":
        scores = (
            ensure_fast_lexical(prepared, signal)
            + ensure_fast_constraints(prepared, signal, constraint_profile=constraint_profile)[0]
            + ensure_fast_signature(prepared, signal) * SIGNATURE_SCORE_WEIGHT
        )
    elif ranking_key == "hybrid_rrf":
        scores = ensure_fast_rrf(prepared, signal) * 100
    elif ranking_key in {"hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}:
        scores = (ensure_fast_rrf(prepared, signal) * 100) + ensure_fast_constraints(
            prepared,
            signal,
            constraint_profile=constraint_profile,
        )[0]
    elif ranking_key == "signature_only":
        scores = ensure_fast_signature(prepared, signal).copy()
    elif ranking_key == "semantic_signature":
        scores = signal.semantic_scores + ensure_fast_signature(prepared, signal) * SIGNATURE_SCORE_WEIGHT
    elif ranking_key == "hybrid_rrf_constraints_signature":
        scores = (
            (ensure_fast_rrf(prepared, signal) * 100)
            + ensure_fast_constraints(prepared, signal, constraint_profile=constraint_profile)[0]
            + ensure_fast_signature(prepared, signal) * SIGNATURE_SCORE_WEIGHT
        )
    elif ranking_key == "adaptive_signature":
        ambiguity = getattr(signal.query_plan, "ambiguity", {}) or {}
        ambiguity_level = ambiguity.get("level")
        signature_weight = 0.55 if ambiguity_level == "high" else 0.45 if ambiguity_level == "medium" else 0.3
        scores = (
            (ensure_fast_rrf(prepared, signal) * 100)
            + ensure_fast_constraints(prepared, signal, constraint_profile=constraint_profile)[0]
            + ensure_fast_signature(prepared, signal) * signature_weight
        )
    elif ranking_key == "constraints_only":
        scores = ensure_fast_constraints(prepared, signal, constraint_profile=constraint_profile)[0].copy()
    elif ranking_key.endswith("_only"):
        channel = ranking_key.removesuffix("_only")
        scores = signal.channel_scores.get(channel, np.zeros(prepared.bm25_doc_count, dtype=np.float64)).copy()
    else:
        raise ValueError(f"Unknown ranking key: {ranking_key}")
    if ranking_key in CONSTRAINT_RANKING_KEYS:
        _constraint_scores, _constraint_hits, forbidden_mask = ensure_fast_constraints(
            prepared,
            signal,
            constraint_profile=constraint_profile,
        )
        scores = scores.copy()
        scores[forbidden_mask] -= HARD_FORBIDDEN_STAGE_VETO
    return np.round(scores, 6)


def ensure_fast_lexical(prepared: PreparedMockIndex, signal: FastCaseSignals) -> np.ndarray:
    if signal.lexical_scores is None:
        query_text = " ".join(
            part
            for part in (
                " ".join(signal.query_plan.semantic_texts())
                if hasattr(signal.query_plan, "semantic_texts")
                else signal.query_plan.positive_query,
                " ".join(signal.query_plan.desired_stage),
                " ".join(signal.query_plan.positive_purposes),
                " ".join(signal.query_plan.positive_style),
                " ".join(signal.query_plan.visual_hints),
            )
            if part
        )
        signal.lexical_scores = prepared.bm25_scores(tokenize(query_text))
        signal.computed.add("lexical")
    return signal.lexical_scores


def ensure_fast_rrf(prepared: PreparedMockIndex, signal: FastCaseSignals) -> np.ndarray:
    if signal.rrf_scores is None:
        lexical_scores = ensure_fast_lexical(prepared, signal)
        signal.rrf_scores = np.asarray(
            reciprocal_rank_fusion(
                [
                    ranked_indices_from_scores(signal.semantic_scores),
                    ranked_indices_from_scores(lexical_scores),
                ],
                item_count=prepared.bm25_doc_count,
                k=DEFAULT_RRF_K,
            ),
            dtype=np.float64,
        )
        signal.computed.add("rrf")
    return signal.rrf_scores


def ensure_fast_signature(prepared: PreparedMockIndex, signal: FastCaseSignals) -> np.ndarray:
    if signal.signature_scores is None:
        query_tokens = signature_tokens(signal.query_signature)
        scores = np.zeros(prepared.bm25_doc_count, dtype=np.float64)
        for item_index, item_tokens in enumerate(prepared.signature_token_sets):
            scores[item_index] = signature_similarity(query_tokens, item_tokens)
        signal.signature_scores = np.round(scores, 6)
        signal.computed.add("signature")
    return signal.signature_scores


def signature_similarity(query_tokens: dict[str, set[str]], item_tokens: dict[str, set[str]]) -> float:
    total_weight = 0.0
    total_score = 0.0
    for field, weight in SIGNATURE_FIELD_WEIGHTS.items():
        query_values = query_tokens.get(field, set())
        if not query_values:
            continue
        item_values = item_tokens.get(field, set())
        overlap = len(query_values & item_values)
        if overlap == 0:
            continue
        denom = max(1, min(len(query_values), len(item_values)))
        total_score += weight * (overlap / denom)
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(total_score / total_weight, 6)


def ensure_fast_constraints(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    *,
    constraint_profile: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, list[str]]], np.ndarray]:
    if signal.constraint_scores is None or signal.constraint_hits is None or signal.forbidden_stage_mask is None:
        weights = constraint_profile.get("weights", {})
        desired = {canonical_stage(stage) for stage in signal.query_constraints.get("desired_stage", [])}
        forbidden_stages = {canonical_stage(stage) for stage in signal.query_constraints.get("forbidden_stage", [])}
        negative_constraints = list(signal.query_constraints.get("negative_constraints", []))
        aliases = constraint_profile.get("negative_aliases", {})
        scores = np.zeros(prepared.bm25_doc_count, dtype=np.float64)
        hits_by_item: list[dict[str, list[str]]] = [{} for _ in range(prepared.bm25_doc_count)]
        forbidden = (
            np.isin(prepared.stage_values, list(forbidden_stages))
            if forbidden_stages
            else np.zeros(prepared.bm25_doc_count, dtype=bool)
        )
        if forbidden_stages:
            scores[forbidden] -= float(weights.get("forbidden_stage_penalty", 0.0))
            for item_index in np.flatnonzero(forbidden):
                hits_by_item[int(item_index)]["forbidden_stage"] = [str(prepared.stage_values[int(item_index)])]

        if desired:
            desired_mask = np.isin(prepared.stage_values, list(desired)) & ~forbidden
            scores[desired_mask] += float(weights.get("desired_stage_bonus", 0.0))
            for item_index in np.flatnonzero(desired_mask):
                hits_by_item[int(item_index)]["desired_stage"] = [str(prepared.stage_values[int(item_index)])]

        negative_penalty = float(weights.get("negative_constraint_penalty", 0.0))
        if negative_constraints and negative_penalty:
            for constraint in negative_constraints:
                candidates = aliases.get(constraint, [constraint])
                if isinstance(candidates, str):
                    candidates = [candidates]
                lowered = [str(candidate).lower() for candidate in candidates if str(candidate)]
                if not lowered:
                    continue
                for item_index, item_text in enumerate(prepared.constraint_text_lowers):
                    if forbidden[item_index]:
                        continue
                    if any(candidate in item_text for candidate in lowered):
                        scores[item_index] -= negative_penalty
                        hits_by_item[item_index].setdefault("negative_constraints", []).append(constraint)

        for style in negative_styles_for_signal(signal):
            mask = prepared.style_masks.get(style)
            if mask is None:
                continue
            scores[mask] -= STYLE_NEGATIVE_PENALTY
            for item_index in np.flatnonzero(mask):
                hits_by_item[int(item_index)].setdefault("negative_style", []).append(style)

        for style in signal.query_plan.positive_style:
            mask = prepared.style_masks.get(style)
            if mask is None:
                continue
            scores[mask] += STYLE_POSITIVE_BONUS
            for item_index in np.flatnonzero(mask):
                hits_by_item[int(item_index)].setdefault("positive_style", []).append(style)

        signal.constraint_scores = np.round(scores, 6)
        signal.constraint_hits = hits_by_item
        signal.forbidden_stage_mask = forbidden
        signal.computed.add("constraints")
    return signal.constraint_scores, signal.constraint_hits, signal.forbidden_stage_mask


def negative_styles_for_signal(signal: FastCaseSignals) -> list[str]:
    values = list(getattr(signal.query_plan, "negative_style", []) or [])
    text = " ".join(
        str(part)
        for part in (
            getattr(signal.query_plan, "original_text", ""),
            " ".join(signal.query_constraints.get("negative_constraints", [])),
        )
        if part
    )
    lower = text.lower()
    for style, aliases in STYLE_ALIASES.items():
        if style in values:
            continue
        if style in {"human_warmth", "documentary", "real_location"}:
            continue
        if any(alias.lower() in lower for alias in aliases):
            values.append(style)
    return values


def fast_result_row(
    prepared: PreparedMockIndex,
    signal: FastCaseSignals,
    item_index: int,
    *,
    score: float,
    ranking_key: str,
    include_debug_text: bool,
) -> dict[str, Any]:
    constraint_score = 0.0
    constraint_hits: dict[str, list[str]] = {}
    if signal.constraint_scores is not None and signal.constraint_hits is not None:
        constraint_score = float(signal.constraint_scores[item_index])
        constraint_hits = signal.constraint_hits[item_index]
    lexical_score = float(signal.lexical_scores[item_index]) if signal.lexical_scores is not None else 0.0
    rrf_score = float(signal.rrf_scores[item_index] * 100) if signal.rrf_scores is not None else 0.0
    signature_score = float(signal.signature_scores[item_index]) if signal.signature_scores is not None else 0.0
    row = {
        "item_id": prepared.item_ids[item_index],
        "score": round(score, 6),
        "final_score": round(score, 6),
        "embedding_score": round(float(signal.semantic_scores[item_index]), 6),
        "constraint_score": round(constraint_score, 6),
        "constraint_hits": constraint_hits,
        "channel_scores": {
            channel: round(float(values[item_index]), 6)
            for channel, values in signal.channel_scores.items()
        },
        "metadata": prepared.metadata[item_index],
        "ranking_key": ranking_key,
        "lexical_score": round(lexical_score, 6),
        "rrf_score": round(rrf_score, 6),
        "signature_score": round(signature_score, 6),
    }
    if include_debug_text:
        row["lexical_text"] = prepared.lexical_texts[item_index]
        row["scene_signature"] = prepared.scene_signatures[item_index]
    return row


def precompute_embedding_rankings(
    index: dict[str, Any],
    cache: EmbeddingCache,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        query_channels = build_query_channels(case["user_input"])
        ranked = search_index(
            index,
            query_channels,
            cache,
            top_k=len(index["items"]),
            constraints_enabled=False,
        )
        rows.append({"case": case, "embedding_ranked": ranked})
    return rows


def tune_profile_from_precomputed(
    precomputed: list[dict[str, Any]],
    *,
    base_profile: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    candidates = []
    for desired_stage_bonus in TUNING_GRID["desired_stage_bonus"]:
        for forbidden_stage_penalty in TUNING_GRID["forbidden_stage_penalty"]:
            for negative_constraint_penalty in TUNING_GRID["negative_constraint_penalty"]:
                profile = profile_with_weights(
                    desired_stage_bonus=desired_stage_bonus,
                    forbidden_stage_penalty=forbidden_stage_penalty,
                    negative_constraint_penalty=negative_constraint_penalty,
                    base_profile=base_profile,
                )
                report = evaluate_precomputed_cases(
                    precomputed=precomputed,
                    top_k=top_k,
                    constraint_profile=profile,
                    constraints_enabled=True,
                    include_cases=False,
                )
                candidates.append(
                    {
                        "weights": profile["weights"],
                        "metrics": report["metrics"],
                        "selection_score": tuning_selection_key(report["metrics"]),
                    }
                )
    return {"best": select_best_tuning_result(candidates), "candidates": candidates}


def filter_precomputed_by_fixture(
    rows: list[dict[str, Any]],
    fixture_id: str,
    *,
    include: bool = True,
) -> list[dict[str, Any]]:
    return [row for row in rows if (case_fixture_id(row["case"]) == fixture_id) is include]


def evaluate_precomputed_cases(
    *,
    precomputed: list[dict[str, Any]],
    top_k: int,
    constraint_profile: dict[str, Any],
    constraints_enabled: bool = True,
    include_cases: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    case_results = []
    for row in precomputed:
        case = row["case"]
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = (
            rerank_with_constraints(row["embedding_ranked"], query_constraints, constraint_profile)
            if constraints_enabled
            else row["embedding_ranked"]
        )
        target_id = target_item_id(case["target"])
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_rank = rank_of(ranked, target_id)
        expected_prefer_rank = rank_of(ranked, expected_prefer_id) if expected_prefer_id else None
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id) if expected_prefer_id else None
        case_results.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "query_constraints": query_constraints,
                "target_item_id": target_id,
                "target_rank": target_rank,
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": expected_prefer_rank,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": ranked[:top_k],
            }
        )
    elapsed = round(time.perf_counter() - started_at, 3)
    result = {
        "method": "mock_constraint_tuning_retrieval",
        "case_count": len(case_results),
        "top_k": top_k,
        "constraints_enabled": constraints_enabled,
        "elapsed_seconds": elapsed,
        "seconds_per_case": round(elapsed / max(1, len(case_results)), 6),
        "metrics": build_metrics(case_results),
    }
    if include_cases:
        result["cases"] = case_results
    return result


def rank_precomputed_cases(
    precomputed: list[dict[str, Any]],
    *,
    ranking_key: str,
    constraint_profile: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for row in precomputed:
        case = row["case"]
        query_constraints = parse_query_constraints(case["user_input"], constraint_profile)
        ranked = rank_items_for_key(
            row["embedding_ranked"],
            ranking_key=ranking_key,
            user_input=case["user_input"],
            query_constraints=query_constraints,
            constraint_profile=constraint_profile,
        )
        target_id = target_item_id(case["target"])
        expected_prefer_id = target_item_id(case["expected_prefer"]) if case.get("expected_prefer") else None
        target_score = score_of(ranked, target_id)
        expected_prefer_score = score_of(ranked, expected_prefer_id) if expected_prefer_id else None
        rows.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "expected_relation": case["expected_relation"],
                "query_constraints": query_constraints,
                "target_item_id": target_id,
                "target_rank": rank_of(ranked, target_id),
                "target_score": target_score,
                "expected_prefer_item_id": expected_prefer_id,
                "expected_prefer_rank": rank_of(ranked, expected_prefer_id) if expected_prefer_id else None,
                "expected_prefer_score": expected_prefer_score,
                "expected_prefer_margin": (
                    round(expected_prefer_score - target_score, 6)
                    if expected_prefer_score is not None and target_score is not None
                    else None
                ),
                "top_results": ranked[:top_k],
                "all_results": ranked,
            }
        )
    return rows


def rank_items_for_key(
    embedding_ranked: list[dict[str, Any]],
    *,
    ranking_key: str,
    user_input: str = "",
    query_constraints: dict[str, Any],
    constraint_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    lexical_scores = lexical_scores_for_ranked(user_input, embedding_ranked)
    semantic_scores = [float(row["embedding_score"]) for row in embedding_ranked]
    rrf_scores = reciprocal_rank_fusion(
        [
            ranked_indices_from_scores(semantic_scores),
            ranked_indices_from_scores(lexical_scores),
        ],
        item_count=len(embedding_ranked),
        k=DEFAULT_RRF_K,
    )
    ranked = []
    for index, row in enumerate(embedding_ranked):
        constraint_score, constraint_hits = score_constraints(
            query_constraints,
            row.get("metadata", {}),
            constraint_profile,
        )
        style_score, style_hits = score_mock_style_constraints(user_input, row)
        full_constraint_score = constraint_score + style_score
        full_constraint_hits = {**constraint_hits}
        full_constraint_hits.update(style_hits)
        if ranking_key == "final_score":
            score = float(row["embedding_score"]) + full_constraint_score
        elif ranking_key in {"embedding_only", "semantic_only"}:
            score = float(row["embedding_score"])
        elif ranking_key == "lexical_only":
            score = lexical_scores[index]
        elif ranking_key == "hybrid_rrf":
            score = rrf_scores[index] * 100
        elif ranking_key in {"hybrid_rrf_constraints", "hybrid_rrf_constraints_rerank"}:
            score = (rrf_scores[index] * 100) + full_constraint_score
        elif ranking_key == "constraints_only":
            score = full_constraint_score
        elif ranking_key.endswith("_only"):
            channel = ranking_key.removesuffix("_only")
            score = float(row.get("channel_scores", {}).get(channel, 0.0))
        else:
            raise ValueError(f"Unknown ranking key: {ranking_key}")
        if ranking_key in CONSTRAINT_RANKING_KEYS and has_forbidden_stage_hit(query_constraints, row.get("metadata", {})):
            score -= HARD_FORBIDDEN_STAGE_VETO
        ranked.append(
            {
                **row,
                "score": round(score, 6),
                "final_score": round(score, 6),
                "ranking_key": ranking_key,
                "lexical_score": round(lexical_scores[index], 6),
                "rrf_score": round(rrf_scores[index] * 100, 6),
                "constraint_score": round(full_constraint_score, 6),
                "constraint_hits": full_constraint_hits,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def summarize_ranking_key_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact_rows = [
        {key: value for key, value in row.items() if key != "all_results"}
        for row in rows
    ]
    return {
        "metrics": build_metrics(compact_rows),
        "case_count": len(compact_rows),
    }


def compact_case_rows(rows: list[dict[str, Any]], *, include_debug_text: bool) -> list[dict[str, Any]]:
    return [compact_case_row(row, include_debug_text=include_debug_text) for row in rows]


def compact_case_row(row: dict[str, Any], *, include_debug_text: bool) -> dict[str, Any]:
    compact = {key: value for key, value in row.items() if key != "all_results"}
    if "top_results" in compact:
        compact["top_results"] = [
            compact_result_row(result, include_debug_text=include_debug_text)
            for result in compact["top_results"]
        ]
    return compact


def compact_result_row(row: dict[str, Any], *, include_debug_text: bool) -> dict[str, Any]:
    if include_debug_text:
        return row
    return {key: value for key, value in row.items() if key != "lexical_text"}


def lexical_scores_for_ranked(user_input: str, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    plan = build_query_plan(user_input)
    query_text = " ".join(
        part
        for part in (
            plan.positive_query,
            " ".join(plan.desired_stage),
            " ".join(plan.positive_purposes),
            " ".join(plan.positive_style),
            " ".join(plan.visual_hints),
        )
        if part
    )
    docs = [tokenize(row.get("lexical_text") or mock_item_lexical_text(row)) for row in rows]
    return bm25_scores(tokenize(query_text), docs)


def score_mock_style_constraints(user_input: str, row: dict[str, Any]) -> tuple[float, dict[str, list[str]]]:
    plan = build_query_plan(user_input)
    text = (row.get("lexical_text") or mock_item_lexical_text(row)).lower()
    return score_style_constraints_from_plan(plan, text)


def score_style_constraints_from_plan(plan: Any, text: str) -> tuple[float, dict[str, list[str]]]:
    score = 0.0
    hits: dict[str, list[str]] = {}
    negative_hits = [style for style in negative_styles_for_plan(plan) if style in text or style_alias_hit(style, text)]
    if negative_hits:
        score -= STYLE_NEGATIVE_PENALTY * len(negative_hits)
        hits["negative_style"] = negative_hits
    positive_hits = [style for style in plan.positive_style if style in text or style_alias_hit(style, text)]
    if positive_hits:
        score += STYLE_POSITIVE_BONUS * len(positive_hits)
        hits["positive_style"] = positive_hits
    return round(score, 6), hits


def negative_styles_for_plan(plan: Any) -> list[str]:
    values = list(getattr(plan, "negative_style", []) or [])
    text = " ".join(
        str(part)
        for part in (
            getattr(plan, "original_text", ""),
            " ".join(getattr(plan, "negative_constraints", []) or []),
        )
        if part
    ).lower()
    for style, aliases in STYLE_ALIASES.items():
        if style in values or style in {"human_warmth", "documentary", "real_location"}:
            continue
        if any(alias.lower() in text for alias in aliases):
            values.append(style)
    return values


def style_alias_hit(style: str, text: str) -> bool:
    return any(alias.lower() in text for alias in STYLE_ALIASES.get(style, ()))


def target_negative_style_hits(case: dict[str, Any], index_items: dict[str, dict[str, Any]]) -> list[str]:
    item = index_items.get(target_item_id(case["target"]))
    if item is None:
        return []
    row = {
        "metadata": item.get("metadata", {}),
        "lexical_text": index_item_lexical_text(item),
    }
    _score, hits = score_mock_style_constraints(case["user_input"], row)
    return list(hits.get("negative_style", []))


def has_forbidden_stage_hit(query_constraints: dict[str, Any], metadata: dict[str, Any]) -> bool:
    forbidden = {canonical_stage(stage) for stage in query_constraints.get("forbidden_stage", [])}
    return bool(forbidden and canonical_stage(metadata.get("script_stage", "")) in forbidden)


def mock_item_lexical_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    parts = [
        metadata.get("script_stage", ""),
        " ".join(metadata.get("creative_purpose", [])),
        metadata.get("script_use_sentence", ""),
        metadata.get("industry", ""),
        metadata.get("style", ""),
        row.get("lexical_text", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def build_pairwise_report(
    precomputed: list[dict[str, Any]],
    ranked_by_key: dict[str, list[dict[str, Any]]],
    *,
    index: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    pairs = build_pairwise_pairs(precomputed, index=index, top_k=top_k)
    report: dict[str, Any] = {}
    by_case_and_key = {
        key: {row["case_id"]: row for row in rows}
        for key, rows in ranked_by_key.items()
    }
    for pair_type, type_pairs in pairs.items():
        report[pair_type] = {}
        for ranking_key in ranked_by_key:
            key_rows = by_case_and_key[ranking_key]
            correct = 0
            scored = 0
            skipped = 0
            margins: list[float] = []
            for pair in type_pairs:
                row = key_rows.get(pair["case_id"])
                if row is None:
                    skipped += 1
                    continue
                better_score = score_of(row["all_results"], pair["better_item_id"])
                worse_score = score_of(row["all_results"], pair["worse_item_id"])
                if better_score is None or worse_score is None:
                    skipped += 1
                    continue
                margin = round(better_score - worse_score, 6)
                margins.append(margin)
                scored += 1
                if margin > 0:
                    correct += 1
            report[pair_type][ranking_key] = {
                "count": scored,
                "correct": correct,
                "skipped_pairs": skipped,
                "accuracy": round(correct / scored, 6) if scored else 0.0,
                "mean_margin": round(mean(margins), 6) if margins else 0.0,
            }
    return report


def build_pairwise_pairs(
    precomputed: list[dict[str, Any]],
    *,
    index: dict[str, Any],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    index_items = {item["item_id"]: item for item in index["items"]}
    pairs: dict[str, list[dict[str, Any]]] = {
        "positive_vs_random": [],
        "positive_vs_wrong_stage": [],
        "expected_prefer_vs_forbidden": [],
    }
    for row in precomputed:
        case = row["case"]
        target_id = target_item_id(case["target"])
        if case["case_type"] in {"simple_positive", "hard_positive"}:
            random_negative = find_random_negative_item_id(index_items, case)
            if random_negative:
                pairs["positive_vs_random"].append(
                    {"case_id": case["case_id"], "better_item_id": target_id, "worse_item_id": random_negative}
                )
            wrong_stage = find_visually_similar_wrong_stage_item_id(row["embedding_ranked"], case, top_k=top_k)
            if wrong_stage:
                pairs["positive_vs_wrong_stage"].append(
                    {"case_id": case["case_id"], "better_item_id": target_id, "worse_item_id": wrong_stage}
                )
        elif case["case_type"] == "hard_negative" and case.get("expected_prefer"):
            pairs["expected_prefer_vs_forbidden"].append(
                {
                    "case_id": case["case_id"],
                    "better_item_id": target_item_id(case["expected_prefer"]),
                    "worse_item_id": target_id,
                }
            )
    return pairs


def find_random_negative_item_id(index_items: dict[str, dict[str, Any]], case: dict[str, Any]) -> str | None:
    target = case["target"]
    for item_id, item in sorted(index_items.items()):
        metadata = item["metadata"]
        if metadata.get("fixture_id") == target.get("fixture_id"):
            continue
        if canonical_stage(metadata.get("script_stage")) == canonical_stage(target.get("script_stage")):
            continue
        return item_id
    return None


def find_visually_similar_wrong_stage_item_id(
    embedding_ranked: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    top_k: int,
) -> str | None:
    target_id = target_item_id(case["target"])
    target_stage = canonical_stage(case["target"].get("script_stage"))
    for row in embedding_ranked[:top_k]:
        if row["item_id"] == target_id:
            continue
        if canonical_stage(row.get("metadata", {}).get("script_stage")) != target_stage:
            return row["item_id"]
    return None


def pair_metric(pairwise: dict[str, Any], pair_type: str, ranking_key: str, metric_name: str) -> float:
    return float(pairwise.get(pair_type, {}).get(ranking_key, {}).get(metric_name, 0.0))


def build_margin_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [case_margin(row) for row in rows]
    buckets = {"low": 0, "medium": 0, "high": 0}
    for margin in margins:
        buckets[confidence_bucket(margin)] += 1
    count = len(margins)
    expected_prefer_margins = [
        row["expected_prefer_margin"]
        for row in rows
        if row.get("expected_prefer_margin") is not None
    ]
    top1_top2_margins = [top1_top2_margin(row) for row in rows if len(row.get("top_results", [])) > 1]
    return {
        "count": count,
        "low_confidence_rate": round(buckets["low"] / count, 6) if count else 0.0,
        "medium_confidence_rate": round(buckets["medium"] / count, 6) if count else 0.0,
        "high_confidence_rate": round(buckets["high"] / count, 6) if count else 0.0,
        "mean_margin": round(mean(margins), 6) if margins else 0.0,
        "mean_top1_top2_margin": round(mean(top1_top2_margins), 6) if top1_top2_margins else 0.0,
        "mean_expected_prefer_minus_forbidden_margin": (
            round(mean(expected_prefer_margins), 6) if expected_prefer_margins else 0.0
        ),
        "cases": [
            {
                "case_id": row["case_id"],
                "case_type": row["case_type"],
                "margin": round(case_margin(row), 6),
                "confidence": confidence_bucket(case_margin(row)),
                "top1_item_id": row.get("top_results", [{}])[0].get("item_id") if row.get("top_results") else None,
            }
            for row in rows
        ],
    }


def case_margin(row: dict[str, Any]) -> float:
    if row.get("expected_prefer_margin") is not None:
        return float(row["expected_prefer_margin"])
    return top1_top2_margin(row)


def top1_top2_margin(row: dict[str, Any]) -> float:
    top_results = row.get("top_results", [])
    if len(top_results) < 2:
        return 0.0
    return float(top_results[0]["score"]) - float(top_results[1]["score"])


def confidence_bucket(margin: float) -> str:
    if margin < 0.02:
        return "low"
    if margin < 0.05:
        return "medium"
    return "high"


def select_worst_cases(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=case_margin)
    worst = []
    for row in sorted_rows[:limit]:
        worst.append(
            {
                "case_id": row["case_id"],
                "case_type": row["case_type"],
                "margin": round(case_margin(row), 6),
                "confidence": confidence_bucket(case_margin(row)),
                "target_rank": row.get("target_rank"),
                "expected_prefer_rank": row.get("expected_prefer_rank"),
                "top_results": [
                    {
                        "item_id": result["item_id"],
                        "score": result["score"],
                        "script_stage": result.get("metadata", {}).get("script_stage"),
                    }
                    for result in row.get("top_results", [])[:3]
                ],
            }
        )
    return worst


def run_llm_sample_judge(
    *,
    ranked_by_key: dict[str, list[dict[str, Any]]],
    sample_size: int,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    candidates = select_llm_sample_candidates(ranked_by_key, sample_size=sample_size)
    results = []
    errors = []
    for row in candidates:
        try:
            results.append(
                judge_top_results_with_llm(
                    row,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
            )
        except Exception as exc:
            errors.append({"case_id": row["case_id"], "error": str(exc)})
    scored_items = [
        item
        for result in results
        for item in result.get("judgements", [])
        if isinstance(item.get("score"), (int, float))
    ]
    relevant = [item for item in scored_items if float(item["score"]) >= 2.0]
    return {
        "sample_size": sample_size,
        "attempted": len(candidates),
        "judged": len(results),
        "precision_at_3": round(len(relevant) / len(scored_items), 6) if scored_items else 0.0,
        "average_score": round(mean([float(item["score"]) for item in scored_items]), 6) if scored_items else 0.0,
        "results": results,
        "llm_errors": errors,
    }


def select_llm_sample_candidates(
    ranked_by_key: dict[str, list[dict[str, Any]]],
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    final_rows = ranked_by_key["final_score"]
    embedding_rows = {row["case_id"]: row for row in ranked_by_key["embedding_only"]}

    def priority(row: dict[str, Any]) -> tuple[int, float, str]:
        low_confidence = confidence_bucket(case_margin(row)) == "low"
        embedding_row = embedding_rows.get(row["case_id"], {})
        final_top = row.get("top_results", [{}])[0].get("item_id") if row.get("top_results") else None
        embedding_top = (
            embedding_row.get("top_results", [{}])[0].get("item_id")
            if embedding_row.get("top_results")
            else None
        )
        changed = final_top != embedding_top
        hard_negative = row["case_type"] == "hard_negative"
        return (
            0 if low_confidence else 1 if changed else 2 if hard_negative else 3,
            case_margin(row),
            row["case_id"],
        )

    return sorted(final_rows, key=priority)[:sample_size]


def judge_top_results_with_llm(
    row: dict[str, Any],
    *,
    timeout_seconds: float,
    retries: int,
) -> dict[str, Any]:
    from sceneweaver.llm.client import VisionLLMClient

    payload = {
        "task": "你是企业宣传片导演，请判断每个检索结果是否适合用户需求。重点看脚本阶段、创作目的、误用风险。只输出 JSON。",
        "score_rule": {
            "0": "完全不适合",
            "1": "画面相似但脚本用途不对",
            "2": "基本适合",
            "3": "非常适合",
        },
        "output_schema": {
            "judgements": [
                {
                    "item_id": "string",
                    "score": "0..3 integer",
                    "reason": "short Chinese sentence",
                }
            ]
        },
        "user_input": row["user_input"],
        "query_constraints": row.get("query_constraints", {}),
        "results": [
            {
                "item_id": result["item_id"],
                "script_stage": result.get("metadata", {}).get("script_stage"),
                "creative_purpose": result.get("metadata", {}).get("creative_purpose"),
                "script_use_sentence": result.get("metadata", {}).get("script_use_sentence"),
                "score": result.get("score"),
            }
            for result in row.get("top_results", [])[:3]
        ],
    }
    result = VisionLLMClient().analyze_text_json(
        system_prompt="你是严格的检索质量评审。只输出 JSON。",
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=600,
        timeout_seconds=timeout_seconds,
        retries=retries,
        enable_thinking=False,
    )
    raw_judgements = result.get("judgements", [])
    if not isinstance(raw_judgements, list):
        raise ValueError("LLM response must contain a judgements list")
    judgements = []
    top_item_ids = {item["item_id"] for item in row.get("top_results", [])[:3]}
    for item in raw_judgements:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id", ""))
        if item_id not in top_item_ids:
            continue
        judgements.append(
            {
                "item_id": item_id,
                "score": int(item.get("score", 0)),
                "reason": str(item.get("reason", "")),
            }
        )
    return {
        "case_id": row["case_id"],
        "case_type": row["case_type"],
        "margin": round(case_margin(row), 6),
        "judgements": judgements,
    }


def build_paraphrase_variants(case: dict[str, Any]) -> list[dict[str, Any]]:
    target = case["target"]
    stage = stage_word(target.get("script_stage", ""))
    purposes = purpose_text(target.get("creative_purpose", []))
    tags = tag_hint(case)
    summary = case.get("target_summary", "")
    industry = target.get("industry", "").replace("_", " ")
    variants = [
        (
            "explicit",
            f"我要一个{stage}段落，核心就是{summary}",
        ),
        (
            "fuzzy",
            f"想让这一段先把观众带进情绪和问题里，不急着讲产品，重点是{purposes}。",
        ),
        (
            "style",
            f"别太像广告，也不要宣传片腔，要更像纪录片观察，但仍然服务于{stage}和{purposes}。",
        ),
        (
            "negative",
            f"不要炫技，不要互联网大厂味，也不要只摆概念；我需要它完成{stage}，重点是{purposes}。",
        ),
        (
            "constraint_first",
            f"先排除那种产品说明和技术炫耀，我真正要的是{stage}，让观众感到{purposes}。",
        ),
        (
            "human_value",
            f"这段要有人味，不要只展示系统；如果出现技术，也只是帮助观众理解{purposes}。",
        ),
        (
            "director_brief",
            f"按导演阐述来讲，这里需要承担{stage}的叙事功能，画面要克制，服务于{purposes}。",
        ),
        (
            "mixed",
            mixed_paraphrase(stage, purposes, tags),
        ),
    ]
    return [
        {
            "case_id": f"paraphrase__{variant_type}__{case['case_id']}",
            "source_case_id": case["case_id"],
            "variant_type": variant_type,
            "user_input": text,
            "target": target,
            "target_summary": summary,
        }
        for variant_type, text in variants
    ]


def build_fuzzy_understanding_variants(case: dict[str, Any]) -> list[dict[str, Any]]:
    target = case["target"]
    stage = stage_word(target.get("script_stage", ""))
    purposes = purpose_text(target.get("creative_purpose", []))
    tags = tag_hint(case)
    summary = case.get("target_summary", "")
    variants = [
        (
            "implicit_stage",
            "stage",
            f"先让观众进入现场和问题，不急着讲产品，重点是让人理解{purposes}。",
        ),
        (
            "fuzzy_style",
            "purpose",
            f"高级但别端着，要真实、有温度，不要宣传片腔；这一段需要服务{purposes}。",
        ),
        (
            "underspecified_tone",
            "purpose",
            f"想要真实一点、有现场感，别太像汇报片，方向大概是{purposes}。",
        ),
        (
            "negative_style",
            "stage",
            f"不要大厂味，不要广告感，也不要炫技；找一个能承担{stage}功能的段落。",
        ),
        (
            "director_brief",
            "scene",
            f"按导演阐述，这里要承担{stage}的叙事功能，画面线索可以靠{tags}，参考：{summary}",
        ),
    ]
    return [
        {
            "case_id": f"fuzzy__{variant_type}__{case['case_id']}",
            "source_case_id": case["case_id"],
            "case_type": "fuzzy_understanding",
            "variant_type": variant_type,
            "expected_granularity": granularity,
            "expected_relation": "should_match",
            "user_input": text,
            "target": target,
            "target_summary": summary,
        }
        for variant_type, granularity, text in variants
    ]


def build_style_negative_case(case: dict[str, Any]) -> dict[str, Any]:
    target = case["target"]
    stage = stage_word(target.get("script_stage", ""))
    purposes = purpose_text(target.get("creative_purpose", []))
    return {
        **case,
        "case_id": f"{case['case_id']}__style_negative",
        "case_type": "style_negative",
        "difficulty": "hard",
        "expected_relation": "should_match",
        "variant_type": "style_negative",
        "source_case_id": case["case_id"],
        "user_input": (
            f"需要一个{stage}段落，重点是{purposes}，要有人味、像纪录片、真实现场。"
            "不要大厂味，不要广告感，不要炫技，也不要技术炫耀。"
            f"参考语义：{case.get('target_summary', '')}"
        ),
    }


def mixed_paraphrase(stage: str, purposes: str, tags: str) -> str:
    if "技术展示" in stage or "技术入场" in stage:
        return f"真正要的是{stage}：{purposes}。画面可以借用{tags}，但不要做成冷冰冰的功能说明。"
    return f"真正要的是{stage}：{purposes}。画面可以借用{tags}，但这些只是辅助，不要让它变成技术展示。"


def stage_word(stage: str) -> str:
    return STAGE_WORDS.get(stage, stage.replace("_", " ") or "某个段落")


def purpose_text(purposes: list[str]) -> str:
    values = [purpose_word(purpose) for purpose in purposes[:3]]
    return "、".join(values) if values else "明确的创作目的"


def purpose_word(purpose: str) -> str:
    overrides = {
        "opening": "开场建立",
        "setup": "需求铺垫",
        "character_intro": "人物出场",
        "team_work": "团队协作",
        "technology_showcase": "技术能力展示",
        "value_expression": "价值表达",
        "scale_reveal": "规模揭示",
        "outcome": "结果落地",
        "ending": "结尾收束",
        "build_empathy": "建立共情",
        "show_responsibility": "表现责任感",
        "humanize_technology": "让技术更有人味",
    }
    return PURPOSE_WORDS.get(purpose) or STAGE_WORDS.get(purpose) or overrides.get(purpose) or purpose.replace("_", " ")


def tag_hint(case: dict[str, Any]) -> str:
    tags = str(case.get("target_tags_text", "")).split()
    if tags:
        return "、".join(tags[:3])
    return "人物、场景、动作、屏幕"


def purpose_hit_at(results: list[dict[str, Any]], target_purposes: set[str], k: int) -> bool:
    if not target_purposes:
        return False
    for result in results[:k]:
        purposes = set(result.get("metadata", {}).get("creative_purpose", []))
        if target_purposes & purposes:
            return True
    return False


def summarize_paraphrase_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "target_recall_at_1": 0.0,
            "target_recall_at_3": 0.0,
            "target_recall_at_10": 0.0,
            "stage_hit_at_1": 0.0,
            "stage_hit_at_3": 0.0,
            "purpose_hit_at_3": 0.0,
            "low_confidence_rate": 0.0,
            "mean_top1_top2_margin": 0.0,
        }
    return {
        "count": len(rows),
        "target_recall_at_1": round(sum(1 for row in rows if row["target_rank"] == 1) / len(rows), 6),
        "target_recall_at_3": round(sum(1 for row in rows if row["target_rank"] and row["target_rank"] <= 3) / len(rows), 6),
        "target_recall_at_10": round(sum(1 for row in rows if row["target_rank"] and row["target_rank"] <= 10) / len(rows), 6),
        "stage_hit_at_1": round(sum(1 for row in rows if row["stage_hit_at_1"]) / len(rows), 6),
        "stage_hit_at_3": round(sum(1 for row in rows if row["stage_hit_at_3"]) / len(rows), 6),
        "purpose_hit_at_3": round(sum(1 for row in rows if row["purpose_hit_at_3"]) / len(rows), 6),
        "low_confidence_rate": round(sum(1 for row in rows if row["confidence"] == "low") / len(rows), 6),
        "mean_top1_top2_margin": round(mean([float(row["top1_top2_margin"]) for row in rows]), 6),
    }


def style_negative_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "target_recall_at_1": 0.0,
            "target_recall_at_3": 0.0,
            "target_recall_at_10": 0.0,
            "style_negative_violation_at_1": 0.0,
            "style_negative_violation_at_3": 0.0,
            "style_negative_violation_at_10": 0.0,
            "low_confidence_rate": 0.0,
        }
    return {
        "count": len(rows),
        "target_recall_at_1": recall_at(rows, 1),
        "target_recall_at_3": recall_at(rows, 3),
        "target_recall_at_10": recall_at(rows, 10),
        "style_negative_violation_at_1": style_violation_at(rows, 1),
        "style_negative_violation_at_3": style_violation_at(rows, 3),
        "style_negative_violation_at_10": style_violation_at(rows, 10),
        "low_confidence_rate": round(sum(1 for row in rows if confidence_bucket(case_margin(row)) == "low") / len(rows), 6),
    }


def style_violation_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    violations = 0
    for row in rows:
        if any(result.get("constraint_hits", {}).get("negative_style") for result in row.get("top_results", [])[:k]):
            violations += 1
    return round(violations / len(rows), 6)


def summarize_paraphrase_by_source_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = sorted({row["source_case_id"] for row in rows})
    return [
        {
            "source_case_id": source_id,
            **summarize_paraphrase_rows([row for row in rows if row["source_case_id"] == source_id]),
        }
        for source_id in source_ids
    ]


def rerank_with_constraints(
    embedding_ranked: list[dict[str, Any]],
    query_constraints: dict[str, Any],
    constraint_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    reranked = []
    for row in embedding_ranked:
        constraint_score, constraint_hits = score_constraints(
            query_constraints,
            row.get("metadata", {}),
            constraint_profile,
        )
        final_score = float(row["embedding_score"]) + constraint_score
        if has_forbidden_stage_hit(query_constraints, row.get("metadata", {})):
            final_score -= HARD_FORBIDDEN_STAGE_VETO
        reranked.append(
            {
                **row,
                "score": round(final_score, 6),
                "final_score": round(final_score, 6),
                "constraint_score": round(constraint_score, 6),
                "constraint_hits": constraint_hits,
            }
        )
    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked


def possible_overfit(
    dev_baseline_metrics: dict[str, Any],
    dev_tuned_metrics: dict[str, Any],
    test_baseline_metrics: dict[str, Any],
    test_tuned_metrics: dict[str, Any],
) -> bool:
    return (
        tuning_selection_key(dev_tuned_metrics) > tuning_selection_key(dev_baseline_metrics)
        and tuning_selection_key(test_tuned_metrics) < tuning_selection_key(test_baseline_metrics)
    )


def summarize_leave_one_fixture(fixture_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not fixture_reports:
        return {
            "fixture_count": 0,
            "mean_hard_negative_margin_positive_rate": 0.0,
            "min_hard_negative_margin_positive_rate": 0.0,
            "mean_simple_positive_recall_at_3": 0.0,
            "mean_hard_positive_recall_at_10": 0.0,
            "worst_fixture_id": None,
            "possible_overfit": False,
        }
    hard_negative_rates = [
        metric_value(row["test_metrics"], "hard_negative", "hard_negative_expected_prefer_margin_positive_rate")
        for row in fixture_reports
    ]
    simple_recall_at_3 = [
        metric_value(row["test_metrics"], "simple_positive", "recall_at_3")
        for row in fixture_reports
    ]
    hard_positive_recall_at_10 = [
        metric_value(row["test_metrics"], "hard_positive", "recall_at_10")
        for row in fixture_reports
    ]
    worst_index = min(range(len(fixture_reports)), key=lambda index: hard_negative_rates[index])
    return {
        "fixture_count": len(fixture_reports),
        "mean_hard_negative_margin_positive_rate": round(mean(hard_negative_rates), 6),
        "min_hard_negative_margin_positive_rate": round(min(hard_negative_rates), 6),
        "mean_simple_positive_recall_at_3": round(mean(simple_recall_at_3), 6),
        "mean_hard_positive_recall_at_10": round(mean(hard_positive_recall_at_10), 6),
        "worst_fixture_id": fixture_reports[worst_index]["fixture_id"],
        "possible_overfit": min(hard_negative_rates) < 0.5,
    }


def metric_value(metrics: dict[str, Any], case_type: str, metric_name: str) -> float:
    return float(metrics.get("by_case_type", {}).get(case_type, {}).get(metric_name, 0.0))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def search_index(
    index: dict[str, Any],
    query_channels: list[dict[str, Any]],
    cache: EmbeddingCache,
    *,
    top_k: int,
    user_input: str = "",
    constraint_profile: dict[str, Any] | None = None,
    constraints_enabled: bool = True,
) -> list[dict[str, Any]]:
    query_vectors = {
        channel["channel"]: cache.require_embedding(channel["text"])
        for channel in query_channels
        if channel.get("enabled", True)
    }
    query_constraints = (
        parse_query_constraints(user_input, constraint_profile)
        if constraints_enabled and user_input
        else {"desired_stage": [], "forbidden_stage": [], "negative_constraints": [], "visual_hints": []}
    )
    scored = []
    for item in index["items"]:
        embedding_score, channel_scores = score_item(item, query_channels, query_vectors)
        constraint_score, constraint_hits = (
            score_constraints(query_constraints, item.get("metadata", {}), constraint_profile)
            if constraints_enabled
            else (0.0, {})
        )
        final_score = embedding_score + constraint_score
        if constraints_enabled and has_forbidden_stage_hit(query_constraints, item.get("metadata", {})):
            final_score -= HARD_FORBIDDEN_STAGE_VETO
        scored.append(
            {
                "item_id": item["item_id"],
                "score": round(final_score, 6),
                "final_score": round(final_score, 6),
                "embedding_score": round(embedding_score, 6),
                "constraint_score": round(constraint_score, 6),
                "constraint_hits": constraint_hits,
                "channel_scores": channel_scores,
                "metadata": item["metadata"],
                "lexical_text": index_item_lexical_text(item),
            }
        )
    scored.sort(key=lambda row: row["final_score"], reverse=True)
    return scored[:top_k]


def score_item(
    item: dict[str, Any],
    query_channels: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
) -> tuple[float, dict[str, float]]:
    item_channels = {channel["channel"]: channel for channel in item["channels"]}
    total = 0.0
    channel_scores: dict[str, float] = {}
    for query_channel in query_channels:
        if not query_channel.get("enabled", True):
            continue
        target_channel = query_channel.get("target_channel") or target_channel_for_query(query_channel["channel"])
        item_channel = item_channels.get(target_channel)
        query_vector = query_vectors.get(query_channel["channel"])
        if item_channel is None or query_vector is None:
            continue
        weight = float(query_channel.get("weight", DEFAULT_CHANNEL_WEIGHTS.get(target_channel, 0.0)))
        similarity = cosine(query_vector, item_channel["embedding"])
        contribution = weight * similarity
        channel_scores[target_channel] = round(channel_scores.get(target_channel, 0.0) + contribution, 6)
        total += contribution
    return total, channel_scores


def index_item_lexical_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    channel_text = " ".join(
        channel.get("text", "")
        for channel in item.get("channels", [])
        if isinstance(channel, dict)
    )
    parts = [
        metadata.get("script_stage", ""),
        " ".join(metadata.get("creative_purpose", [])),
        metadata.get("script_use_sentence", ""),
        metadata.get("industry", ""),
        metadata.get("style", ""),
        channel_text,
    ]
    return " ".join(str(part) for part in parts if part)


def build_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in case_results:
        by_type.setdefault(row["case_type"], []).append(row)
    return {
        "overall": metric_group(case_results),
        "by_case_type": {case_type: metric_group(rows) for case_type, rows in by_type.items()},
    }


def metric_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if row["expected_relation"] == "should_match"]
    negative_rows = [row for row in rows if row["expected_relation"] == "should_not_match"]
    return {
        "count": len(rows),
        "recall_at_1": recall_at(positive_rows, 1),
        "recall_at_3": recall_at(positive_rows, 3),
        "recall_at_10": recall_at(positive_rows, 10),
        "mrr": mean_reciprocal_rank(positive_rows),
        "hard_negative_expected_prefer_margin_positive_rate": margin_positive_rate(negative_rows),
        "forbidden_stage_violation_at_1": forbidden_stage_violation_at(rows, 1),
        "forbidden_stage_violation_at_3": forbidden_stage_violation_at(rows, 3),
        "forbidden_stage_violation_at_10": forbidden_stage_violation_at(rows, 10),
        "desired_stage_hit_at_1": desired_stage_hit_at(rows, 1),
        "desired_stage_hit_at_3": desired_stage_hit_at(rows, 3),
    }


def recall_at(rows: list[dict[str, Any]], k: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if row["target_rank"] is not None and row["target_rank"] <= k)
    return round(hits / len(rows), 6)


def mean_reciprocal_rank(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = sum(1 / row["target_rank"] for row in rows if row["target_rank"])
    return round(total / len(rows), 6)


def margin_positive_rate(rows: list[dict[str, Any]]) -> float:
    margins = [row["expected_prefer_margin"] for row in rows if row["expected_prefer_margin"] is not None]
    if not margins:
        return 0.0
    return round(sum(1 for margin in margins if margin > 0) / len(margins), 6)


def forbidden_stage_violation_at(rows: list[dict[str, Any]], k: int) -> float:
    constrained = [row for row in rows if row.get("query_constraints", {}).get("forbidden_stage")]
    if not constrained:
        return 0.0
    violations = 0
    for row in constrained:
        forbidden = {canonical_stage(stage) for stage in row.get("query_constraints", {}).get("forbidden_stage", [])}
        if any(result_stage(result) in forbidden for result in row.get("top_results", [])[:k]):
            violations += 1
    return round(violations / len(constrained), 6)


def desired_stage_hit_at(rows: list[dict[str, Any]], k: int) -> float:
    constrained = [row for row in rows if row.get("query_constraints", {}).get("desired_stage")]
    if not constrained:
        return 0.0
    hits = 0
    for row in constrained:
        desired = {canonical_stage(stage) for stage in row.get("query_constraints", {}).get("desired_stage", [])}
        if any(result_stage(result) in desired for result in row.get("top_results", [])[:k]):
            hits += 1
    return round(hits / len(constrained), 6)


def result_stage(result: dict[str, Any]) -> str:
    return canonical_stage(result.get("metadata", {}).get("script_stage", ""))


def tuning_selection_key(metrics: dict[str, Any]) -> list[float]:
    by_type = metrics.get("by_case_type", {})
    hard_negative = by_type.get("hard_negative", {})
    hard_positive = by_type.get("hard_positive", {})
    simple_positive = by_type.get("simple_positive", {})
    return [
        float(hard_negative.get("hard_negative_expected_prefer_margin_positive_rate", 0.0)),
        float(hard_positive.get("recall_at_10", 0.0)),
        float(simple_positive.get("recall_at_3", 0.0)),
    ]


def select_best_tuning_result(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No tuning candidates were produced.")
    return max(candidates, key=lambda row: row["selection_score"])


def rank_of(rows: list[dict[str, Any]], item_id: str | None) -> int | None:
    if item_id is None:
        return None
    for index, row in enumerate(rows, start=1):
        if row["item_id"] == item_id:
            return index
    return None


def score_of(rows: list[dict[str, Any]], item_id: str | None) -> float | None:
    if item_id is None:
        return None
    for row in rows:
        if row["item_id"] == item_id:
            return row["score"]
    return None


def rank_of_prepared_id(prepared: PreparedMockIndex, ranked_indices: list[int], item_id: str | None) -> int | None:
    if item_id is None:
        return None
    target_index = prepared.item_index_by_id.get(item_id)
    if target_index is None:
        return None
    for rank, item_index in enumerate(ranked_indices, start=1):
        if item_index == target_index:
            return rank
    return None


def score_of_prepared_id(prepared: PreparedMockIndex, scores: np.ndarray, item_id: str | None) -> float | None:
    if item_id is None:
        return None
    item_index = prepared.item_index_by_id.get(item_id)
    if item_index is None:
        return None
    return round(float(scores[item_index]), 6)


def cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def read_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_embedding_setup_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "dashscope package is required" in message
        or "DASHSCOPE_API_KEY" in message
        or "embedding missing for text key=" in message
    )


def embedding_setup_error_payload(exc: RuntimeError) -> dict[str, Any]:
    return {
        "error": "embedding_setup_unavailable",
        "message": str(exc),
        "hint": (
            "Activate the environment with dashscope and API credentials before uncached embedding runs, "
            "for example: conda activate video_expert_analyzer and set DASHSCOPE_API_KEY."
        ),
    }


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if not is_embedding_setup_error(exc):
            raise
        print(json.dumps(embedding_setup_error_payload(exc), ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2) from None
