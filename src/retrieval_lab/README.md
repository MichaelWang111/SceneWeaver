# Retrieval Lab

Retrieval Lab is SceneWeaver's retrieval experiment platform. It is a new
parallel project next to `src/mocktesting`, so the old mocktesting package stays
available as a behavior baseline while the lab is reorganized into clearer
experiment-platform layers.

The package path is `src/retrieval_lab`; the product name is "Retrieval Lab".

## Architecture

Retrieval Lab follows the same broad shape used by IR and LLM evaluation tools:

```text
datasets/fixtures -> schemas -> planners -> indexes/retrieval -> ranking -> qrels/evaluators -> experiments -> reports/capability
```

The design is based on a few practical patterns:

- OpenAI Evals style: an eval registry, reproducible runs, custom task-specific evaluators.
- BEIR/TREC style: qrels, pooled candidates, graded relevance, NDCG/MRR/Recall style metrics.
- LangSmith/MLflow style: datasets, experiment records, annotations, traces, and run comparison.
- Azure/Pinecone retrieval style: hybrid retrieval, RRF fusion, and optional two-stage reranking.

No external service is required by default. Real LLM use remains opt-in.

## Current Migration State

- Native in Retrieval Lab:
  - Modern CLI groups and aliases.
  - Architecture manifest and path conventions.
  - Flywheel guide.
  - Dataset manifest, split audit, schema summary, and fixture inventory.
  - Qrels audit, pooling from run artifacts, active review sampling from run artifacts, and adjudication merge.
  - Standard IR metrics: nDCG, ERR, MRR, judged coverage, recall, and recall-bound summaries.
  - Run artifact export from reports with reusable `run_rows` schema.
  - Legacy runner bridge that executes `mocktesting` commands while emitting Retrieval Lab reports, run artifacts, and manifests.
  - Native run-artifact reranking: rule rerank and qrels oracle rerank.
  - Native workflow reranking over saved run artifacts: semantic, lexical, RRF, constraints, signature, and adaptive signature keys.
  - Native run-artifact evaluation against qrels.
  - Native failure attribution from saved run artifacts and optional qrels.
  - Native experiment comparison across JSON reports with metric deltas and best-run selection.
  - Native evaluation Markdown reports from one or more JSON reports.
  - Native infra coverage audit for migration status and remaining gaps.
  - Native migration audit gate that checks `src/mocktesting` cleanliness and tracks coverage deltas.
  - Native schema catalog and validation contracts for query plans, run rows, qrels, index manifests, retrieval configs, LLM judgements, and capability cycles.
  - Native planner registry, planner cache, rule/multi-query/HyDE/fake-LLM planners, and a legacy adapter for parity checks.
  - Native in-memory retrieval runtime with BM25-style lexical scoring, local hash-vector semantic fallback, RRF fusion, constraint scoring, style safety, and scene-signature signals.
  - Native core experiment commands for search/evaluate/hybrid workflow comparison, fuzzy/paraphrase validation, graded and pooled evaluation, rerank upper-bound analysis, style-risk validation, and hard-negative mining.
  - Native migration certification for core experiment replacement.
  - Artifact IO, SHA256 fingerprints, and experiment artifact manifests.
  - Capability cycle registry, Markdown reports, and SVG charts.
- Compatibility backend:
  - `mocktesting` remains available as a baseline and parity backend.
  - Low-priority historical commands can remain compatibility-backed when they are outside the core experiment chain.
  - Core daily retrieval experiments are expected to use native Retrieval Lab commands first.

This lets each migration round compare the new lab against `mocktesting` without
changing the baseline.

## CLI Examples

```powershell
python -m retrieval_lab flywheel guide
python -m retrieval_lab dataset inspect --input src\mocktesting\eval_inputs\review_generated_inputs.json --output .tmp\retrieval_lab\dataset_manifest.json
python -m retrieval_lab qrels audit --qrels .tmp\pooled_qrels_next.jsonl
python -m retrieval_lab run export --reports .tmp\fuzzy_multi_next.json --output .tmp\retrieval_lab\run_rows_next.json
python -m retrieval_lab run legacy --legacy-command evaluate-fuzzy-multirelevance --report-output .tmp\fuzzy_multi_next.json --run-output .tmp\retrieval_lab\run_rows_next.json -- --split test --limit 60
python -m retrieval_lab run rerank --runs .tmp\retrieval_lab\run_rows_next.json --method rule --output .tmp\retrieval_lab\run_rows_rule_rerank.json
python -m retrieval_lab run workflow-rerank --runs .tmp\retrieval_lab\run_rows_next.json --ranking-key hybrid_rrf_constraints_signature --output .tmp\retrieval_lab\run_rows_signature.json
python -m retrieval_lab workflow compare-runs --runs .tmp\retrieval_lab\run_rows_next.json --ranking-keys lexical_only,lexical_constraints,hybrid_rrf_constraints_signature,adaptive_signature
python -m retrieval_lab run evaluate --runs .tmp\retrieval_lab\run_rows_next.json --qrels .tmp\pooled_qrels_next.jsonl --markdown-output .tmp\retrieval_lab\run_eval.md
python -m retrieval_lab run analyze-failures --runs .tmp\retrieval_lab\run_rows_next.json --qrels .tmp\pooled_qrels_next.jsonl --markdown-output .tmp\retrieval_lab\failures.md
python -m retrieval_lab experiment compare --reports .tmp\fuzzy_multi_next.json .tmp\strong_baselines_next.json --markdown-output .tmp\retrieval_lab\experiment_compare.md
python -m retrieval_lab report eval --inputs .tmp\qrels_audit_next.json .tmp\fuzzy_multi_next.json .tmp\retrieval_lab\failures.json .tmp\retrieval_lab\experiment_compare.json --output .tmp\retrieval_lab\eval_report.md
python -m retrieval_lab infra audit --markdown-output .tmp\retrieval_lab\infra_audit.md
python -m retrieval_lab migration audit --round-id cycle_002 --markdown-output .tmp\retrieval_lab\migration_audit.md
python -m retrieval_lab planner plan --planner multi_query --query "need grounded setup without product pitch"
python -m retrieval_lab planner compare --dataset src\mocktesting\eval_inputs\review_generated_inputs.json --split test --limit 60 --planners rule,multi_query,hyde_card,legacy_adapter
python -m retrieval_lab planner audit-cache
python -m retrieval_lab evaluate-hybrid --split test --limit 60 --output .tmp\retrieval_lab\evaluate_hybrid_next.json
python -m retrieval_lab validate-fuzzy-understanding --split test --limit 60 --query-planner multi_query --output .tmp\retrieval_lab\fuzzy_next.json
python -m retrieval_lab validate-paraphrase-stress --split test --limit 60 --query-planner multi_query --output .tmp\retrieval_lab\paraphrase_next.json
python -m retrieval_lab compare-rerank-upper-bound --split test --limit 60 --output .tmp\retrieval_lab\rerank_upper_bound_next.json
python -m retrieval_lab mine-hard-negatives --split test --limit 60 --output .tmp\retrieval_lab\hard_negatives_next.json
python -m retrieval_lab schema catalog --include-json-schema --output .tmp\retrieval_lab\schema_catalog.json
python -m retrieval_lab schema show query_plan
python -m retrieval_lab schema validate qrel --input .tmp\pooled_qrels_next.jsonl
python -m retrieval_lab qrels pool-from-runs --runs .tmp\run_rows.json --qrels-output .tmp\pooled_qrels_next.jsonl
python -m retrieval_lab qrels sample-active-from-runs --runs .tmp\run_rows.json --qrels .tmp\pooled_qrels_next.jsonl
python -m retrieval_lab artifact manifest --inputs .tmp\run_rows.json --outputs .tmp\pooled_qrels_next.jsonl
python -m retrieval_lab eval fuzzy --split test --limit 60
python -m retrieval_lab eval rerank-upper-bound --split test --limit 60
python -m retrieval_lab cycle record --cycle-id cycle_002 --reports .tmp\qrels_audit_next.json
python -m retrieval_lab report capability
python -m retrieval_lab migration certify --round-id cycle_002
```

Legacy commands still work through the compatibility bridge:

```powershell
python -m retrieval_lab retrieval-flywheel-guide
python -m retrieval_lab evaluate-fuzzy-multirelevance --split test --limit 60
```

## Migration Rule

Move one subsystem at a time:

1. Extract the stable service code into a Retrieval Lab layer.
2. Add a native CLI route.
3. Compare representative outputs against `mocktesting`.
4. Add tests for the native route and the compatibility boundary.
5. Keep LLM calls disabled unless a command explicitly asks for sampling.

The intent is "replace the cage before moving the bird": migrate stable
experiment-platform boundaries first, then replace heavy retrieval execution
piece by piece. Do not copy the `mocktesting` monolith into a new package.
