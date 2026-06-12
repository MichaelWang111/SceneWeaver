from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabLayer:
    name: str
    responsibility: str
    package: str


LAYERS: tuple[LabLayer, ...] = (
    LabLayer("config", "Project names, path conventions, and artifact roots.", "retreieval_lab.config"),
    LabLayer("datasets", "Cases, splits, variants, and fixture-derived inputs.", "retreieval_lab.datasets"),
    LabLayer("fixtures", "Fixture inventory, lineage, and fixture-level audit helpers.", "retreieval_lab.fixtures"),
    LabLayer("planners", "Query understanding strategies and planner comparisons.", "retreieval_lab.planners"),
    LabLayer("indexes", "Embedding, matrix, lexical, and prepared indexes.", "retreieval_lab.indexes"),
    LabLayer("retrieval", "Recall workflows: semantic, lexical, hybrid, and RRF.", "retreieval_lab.retrieval"),
    LabLayer("ranking", "Constraints, signatures, rerankers, and workflow ranking.", "retreieval_lab.ranking"),
    LabLayer("qrels", "Pooled qrels, adjudication, active sampling, and trust audit.", "retreieval_lab.qrels"),
    LabLayer("evaluators", "IR metrics, failure attribution, and upper-bound analysis.", "retreieval_lab.evaluators"),
    LabLayer("experiments", "Experiment runners, registries, and cycle orchestration.", "retreieval_lab.experiments"),
    LabLayer("artifacts", "Run artifacts, manifests, fingerprints, and JSON/JSONL lifecycle helpers.", "retreieval_lab.artifacts"),
    LabLayer("schemas", "Versioned contracts for query plans, run rows, qrels, indexes, and LLM judgements.", "retreieval_lab.schemas"),
    LabLayer("reports", "JSON, Markdown, SVG, and human-readable summaries.", "retreieval_lab.reports"),
    LabLayer("capability", "Capability cycles and longitudinal experiment tracking.", "retreieval_lab.capability"),
    LabLayer("llm", "Optional LLM adapters, fake clients, and sampled judging helpers.", "retreieval_lab.llm"),
)


def layer_manifest() -> list[dict[str, str]]:
    return [layer.__dict__.copy() for layer in LAYERS]
