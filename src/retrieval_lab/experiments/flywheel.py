from __future__ import annotations

from pathlib import Path
import json


def retrieval_lab_flywheel_guide() -> dict:
    steps = [
        {
            "step": 1,
            "name": "migration audit",
            "purpose": "Check baseline cleanliness and current Retrieval Lab migration coverage before starting a cycle.",
            "command": "python -m retrieval_lab migration audit --round-id cycle_002 --output .tmp\\retrieval_lab\\migration_audit_next.json --markdown-output .tmp\\retrieval_lab\\migration_audit_next.md",
        },
        {
            "step": 2,
            "name": "planner compare",
            "purpose": "Compare native rule, multi-query, HyDE, and legacy-adapter query understanding without calling real LLMs.",
            "command": "python -m retrieval_lab planner compare --dataset src\\mocktesting\\eval_inputs\\review_generated_inputs.json --split test.md --limit 60 --planners rule,multi_query,hyde_card,legacy_adapter --output .tmp\\retrieval_lab\\planner_compare_next.json",
        },
        {
            "step": 3,
            "name": "qrels build-pooled",
            "purpose": "Build a pooled bootstrap relevance set from multiple workflows.",
            "command": "python -m retrieval_lab qrels build-pooled --split test.md --limit 60 --qrels-output .tmp\\pooled_qrels_next.jsonl",
        },
        {
            "step": 4,
            "name": "qrels audit",
            "purpose": "Audit qrels trust, bootstrap-only rows, low confidence rows, and conflicts.",
            "command": "python -m retrieval_lab qrels audit --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\qrels_audit_next.json --markdown-output .tmp\\qrels_audit_next.md",
        },
        {
            "step": 5,
            "name": "qrels sample-active",
            "purpose": "Sample disagreement, low-confidence, and style-risk examples for review.",
            "command": "python -m retrieval_lab qrels sample-active --split test.md --limit 60 --sample-size 80 --qrels .tmp\\pooled_qrels_next.jsonl --output .tmp\\active_qrels_next.jsonl",
        },
        {
            "step": 6,
            "name": "qrels merge-adjudicated",
            "purpose": "Merge reviewed human/LLM votes back into the qrels registry.",
            "command": "python -m retrieval_lab qrels merge-adjudicated --qrels .tmp\\pooled_qrels_next.jsonl --adjudications .tmp\\active_qrels_reviewed.jsonl --output .tmp\\pooled_qrels_adjudicated.jsonl",
        },
        {
            "step": 7,
            "name": "eval fuzzy",
            "purpose": "Evaluate fuzzy multi-answer retrieval with graded relevance metrics.",
            "command": "python -m retrieval_lab eval fuzzy --split test.md --limit 60 --query-planner multi_query --ranking-key hybrid_rrf_constraints --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\fuzzy_multi_next.json --markdown-output .tmp\\fuzzy_multi_next.md",
        },
        {
            "step": 8,
            "name": "retrieval run",
            "purpose": "Run the native Retrieval Lab in-memory hybrid retrieval runtime and emit reusable run rows.",
            "command": "python -m retrieval_lab retrieval run --split test.md --limit 60 --planner multi_query --output .tmp\\retrieval_lab\\run_rows_next.json",
        },
        {
            "step": 9,
            "name": "eval paraphrase",
            "purpose": "Stress exact scene ranking under natural language paraphrases using the native retrieval runtime.",
            "command": "python -m retrieval_lab eval paraphrase --split test.md --limit 60 --query-planner multi_query --output .tmp\\paraphrase_next.json --markdown-output .tmp\\paraphrase_next.md",
        },
        {
            "step": 10,
            "name": "qrels pool-from-runs",
            "purpose": "Build qrels from a saved run artifact without rerunning ranking.",
            "command": "python -m retrieval_lab qrels pool-from-runs --runs .tmp\\retrieval_lab\\run_rows_next.json --qrels-output .tmp\\pooled_qrels_from_runs_next.jsonl --report-output .tmp\\pooled_qrels_from_runs_next_report.json",
        },
        {
            "step": 11,
            "name": "run rerank",
            "purpose": "Apply native rule or qrels-oracle rerank to a saved run artifact.",
            "command": "python -m retrieval_lab run rerank --runs .tmp\\retrieval_lab\\run_rows_next.json --method rule --rerank-depth 20 --top-k 10 --output .tmp\\retrieval_lab\\run_rows_rule_rerank_next.json",
        },
        {
            "step": 12,
            "name": "workflow compare-runs",
            "purpose": "Compare several native workflow ranking keys over saved run artifacts without rerunning retrieval.",
            "command": "python -m retrieval_lab workflow compare-runs --runs .tmp\\retrieval_lab\\run_rows_next.json --ranking-keys lexical_only,lexical_constraints,hybrid_rrf_constraints_signature,adaptive_signature --output .tmp\\retrieval_lab\\workflow_compare_next.json",
        },
        {
            "step": 13,
            "name": "eval rerank-upper-bound",
            "purpose": "Compare baseline, rule rerank, oracle rerank, and optional LLM sample rerank.",
            "command": "python -m retrieval_lab eval rerank-upper-bound --split test.md --limit 60 --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\rerank_upper_bound_next.json --markdown-output .tmp\\rerank_upper_bound_next.md",
        },
        {
            "step": 14,
            "name": "run evaluate",
            "purpose": "Evaluate saved run artifacts against qrels without rerunning retrieval.",
            "command": "python -m retrieval_lab run evaluate --runs .tmp\\retrieval_lab\\run_rows_next.json --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\retrieval_lab\\run_eval_next.json --markdown-output .tmp\\retrieval_lab\\run_eval_next.md",
        },
        {
            "step": 15,
            "name": "run analyze-failures",
            "purpose": "Attribute failures from saved run artifacts and optional qrels without rerunning ranking.",
            "command": "python -m retrieval_lab run analyze-failures --runs .tmp\\retrieval_lab\\run_rows_next.json --qrels .tmp\\pooled_qrels_adjudicated.jsonl --output .tmp\\retrieval_lab\\failure_analysis_next.json --markdown-output .tmp\\retrieval_lab\\failure_analysis_next.md",
        },
        {
            "step": 16,
            "name": "eval failures",
            "purpose": "Run native fresh failure attribution when saved artifacts are not enough.",
            "command": "python -m retrieval_lab eval failures --split test.md --limit 60 --output .tmp\\failure_analysis_next.json --markdown-output .tmp\\failure_analysis_next.md",
        },
        {
            "step": 17,
            "name": "experiment compare",
            "purpose": "Compare JSON reports, choose the best report by stable metrics, and compute deltas.",
            "command": "python -m retrieval_lab experiment compare --reports .tmp\\fuzzy_multi_next.json .tmp\\rerank_upper_bound_next.json .tmp\\retrieval_lab\\run_eval_next.json .tmp\\retrieval_lab\\failure_analysis_next.json --output .tmp\\retrieval_lab\\experiment_comparison_next.json --markdown-output .tmp\\retrieval_lab\\experiment_comparison_next.md",
        },
        {
            "step": 18,
            "name": "report eval",
            "purpose": "Generate one human-readable Markdown summary from the cycle's JSON reports.",
            "command": "python -m retrieval_lab report eval --inputs .tmp\\qrels_audit_next.json .tmp\\fuzzy_multi_next.json .tmp\\rerank_upper_bound_next.json .tmp\\retrieval_lab\\run_eval_next.json .tmp\\retrieval_lab\\failure_analysis_next.json .tmp\\retrieval_lab\\experiment_comparison_next.json --output .tmp\\retrieval_lab\\eval_report_next.md",
        },
        {
            "step": 19,
            "name": "cycle record",
            "purpose": "Record the cycle and compute deltas against the previous cycle.",
            "command": "python -m retrieval_lab cycle record --cycle-id cycle_002 --reports .tmp\\qrels_audit_next.json .tmp\\fuzzy_multi_next.json .tmp\\rerank_upper_bound_next.json",
        },
        {
            "step": 20,
            "name": "report capability",
            "purpose": "Generate Markdown and SVG capability trend reports.",
            "command": "python -m retrieval_lab report capability --registry .tmp\\capability_cycles.jsonl --output .tmp\\capability_report.md",
        },
        {
            "step": 21,
            "name": "migration certify",
            "purpose": "Certify that the core experiment chain is native and mocktesting stayed clean.",
            "command": "python -m retrieval_lab migration certify --round-id cycle_002 --parity-reports .tmp\\fuzzy_multi_next.json .tmp\\paraphrase_next.json .tmp\\rerank_upper_bound_next.json --output .tmp\\retrieval_lab\\migration_certification_next.json --markdown-output .tmp\\retrieval_lab\\migration_certification_next.md",
        },
    ]
    return {
        "method": "retrieval_lab_flywheel_guide",
        "steps": steps,
        "summary": {
            "step_count": len(steps),
            "default_llm_usage": "off",
            "recommended_first_loop": "migration audit -> planner compare -> native qrels build-pooled -> qrels audit -> qrels sample-active -> retrieval run -> fuzzy/paraphrase eval -> rerank upper bound -> mine hard negatives -> experiment compare -> report eval -> cycle record -> migration certify",
        },
    }


def write_flywheel_guide(path: Path, guide: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(guide, ensure_ascii=False, indent=2), encoding="utf-8")
