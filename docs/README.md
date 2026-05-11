# SceneWeaver Docs

SceneWeaver extracts reusable directing experience from commercial videos and turns it into searchable local knowledge.

## Documents

- [Usage](USAGE.md): CLI commands, keyword loop, streaming, thinking mode, and semantic retrieval.
- [Architecture](ARCHITECTURE.md): pipeline, module boundaries, and retrieval design.
- [Schema](SCHEMA.md): JSON and JSONL artifacts produced by the system.
- [Roadmap](ROADMAP.md): current status, next work, and non-goals.

## Current Direction

The active path is:

```text
video case
-> scene packages
-> scene analysis with tags
-> experience cards
-> keyword loop retrieval
```

Retrieval now has two layers:

1. tag scoring for explainable matching;
2. optional embedding reranking for softer creative similarity.

The first semantic model target is `BAAI/bge-small-zh-v1.5`. Vector databases are intentionally deferred until the experience-card library is large enough to justify the operational cost.
