# SceneWeaver Documentation

SceneWeaver turns commercial video cases into reusable directing knowledge. The docs are split by enterprise documentation role so each file has one job.

## Executive Layer

- [Project Overview](../README.md): product positioning, install path, and the shortest useful workflow.
- [Roadmap](ROADMAP.md): current status, near-term work, non-goals, and deferred platform decisions.

## Architecture Layer

- [Architecture](ARCHITECTURE.md): system boundaries, module ownership, and retrieval design.
- [Schema](SCHEMA.md): JSON / JSONL contracts, artifact shapes, and compatibility notes.

## Operations Layer

- [CLI Command Book](CLI.md): runnable PowerShell commands for environment checks, video processing, retrieval, LLM diagnostics, and maintenance.
- [Usage](USAGE.md): workflow-oriented usage guide that explains when to use each command family.

## Working Notes

The `talking/` folder and `now_talk.md` are historical conversation notes. They are useful for product thinking, but they are not the operational source of truth. If a note becomes active policy, move the decision into one of the docs above.

## Current Operating Model

```text
video case
-> scene package
-> scene analysis
-> tags / fingerprint
-> experience card with script_usecase
-> retrieval by brief, tags, script use case, and optional LLM intent
```

The current system intentionally stays file-first:

- local JSON / JSONL artifacts;
- Pydantic validation;
- in-memory retrieval;
- optional local embeddings;
- no production database or vector store yet.
