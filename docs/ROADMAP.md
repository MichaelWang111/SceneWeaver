# Roadmap

## Current Status

Implemented:

- Bilibili video packaging.
- Scene package creation.
- Vision LLM scene analysis.
- Embedded `tags` in scene analysis.
- Experience card extraction.
- Taxonomy-backed tag normalization.
- Candidate tag logging.
- Keyword loop over one film or all film outputs.
- Streaming and thinking debug output for keyword loop.
- Core creative-intent prompt and intent scoring for retrieval ranking.
- Optional embedding reranking for softer creative matching.

## Recommended Near-Term Work

1. Fix existing Chinese mojibake in prompts, taxonomy labels, old tests, and historical sample outputs.
2. Add a tag-candidate review command:

```text
review-tag-candidates
merge-tag-candidate
reject-tag-candidate
```

3. Add optional LLM rerank pass after tag / intent / semantic preselection for the top 10-20 cards.
4. Add embedding cache for experience cards:

```text
analysis/experience_card_embeddings.jsonl
```

5. Add semantic retrieval smoke tests that run only when `sentence-transformers` is installed.
6. Improve retrieval output for creative review:

- show why tag dimensions matched;
- show semantic score;
- show recommended reuse condition;
- show source scene evidence.

## Deferred

- Web UI.
- Vector database.
- Production database.
- Multi-platform video download.
- Storyboard generation.
- Direct video generation.

## Vector Database Decision

Do not add FAISS, Chroma, Milvus, or another vector store yet.

Reason:

```text
The current card library is small enough for in-memory scoring, and in-memory scoring is easier to debug.
```

Revisit when:

- the card library reaches tens of thousands of cards;
- retrieval latency becomes a real bottleneck;
- multiple projects need a shared persistent semantic index.
