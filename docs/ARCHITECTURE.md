# Architecture

## Product Goal

SceneWeaver turns commercial video cases into reusable directing knowledge. It does not generate video directly. It analyzes existing videos, extracts scene-level directing interpretation, and stores reusable experience cards for future creative retrieval.

## Core Pipeline

```text
Bilibili URL
-> download source video and metadata
-> detect scene spans
-> sample start / middle / end frames
-> write scene packages
-> run Vision LLM scene analysis
-> normalize tags
-> extract experience cards
-> retrieve cards from keyword briefs
```

Main output directory:

```text
outputs/film_analysis/<video_id>/
  source/
  frames/
  packages/
  analysis/
```

## Modules

CLI entry:

```text
src/sceneweaver/cli.py
```

Scene packaging:

```text
src/sceneweaver/analysis/scene_package_builder.py
src/sceneweaver/split/scene_detector.py
```

Scene analysis:

```text
src/sceneweaver/analysis/scene_analyzer.py
```

Tags and taxonomy:

```text
src/sceneweaver/analysis/tags.py
src/sceneweaver/analysis/taxonomy.py
taxonomy/director_tags_v1.json
```

Experience cards:

```text
src/sceneweaver/analysis/experience_extractor.py
src/sceneweaver/schemas/experience_card.py
```

Keyword loop:

```text
src/sceneweaver/analysis/keyword_loop.py
```

Semantic retrieval:

```text
src/sceneweaver/analysis/semantic.py
```

## Retrieval Design

### Tag Scoring

Tag scoring is explainable and deterministic. Query tags and card tags are compared dimension by dimension.

Weights:

```text
audience_projection: 2.0
interaction_mode: 2.0
emotion_core: 1.5
narrative_function: 1.25
symbolic_logic: 1.25
visual_motifs: 1.0
rhythm_pattern: 1.0
```

Formula:

```text
tag_score = sum(matched_tag_count_in_dimension * dimension_weight)
```

### Semantic Reranking

Semantic reranking is optional. It uses local sentence embeddings to make retrieval less brittle when the query and card do not share the same canonical tags.

Default model:

```text
BAAI/bge-small-zh-v1.5
```

Final score:

```text
score = tag_score + max(0, semantic_score) * semantic_weight
```

Default `semantic_weight` is `4.0`.

The system intentionally does not use a vector database yet. For tens, hundreds, or low thousands of cards, encoding and scoring in memory is simpler and easier to inspect. A vector index can be added later when the card library grows substantially.

## Tag Governance

Tags are semi-closed.

Rules:

1. Main artifacts only store canonical tags.
2. Aliases and near-synonyms belong in `taxonomy/director_tags_v1.json`.
3. New expressions are logged to `tag_candidates.jsonl`.
4. Candidate tags are reviewed before becoming aliases or canonical tags.
5. `custom_tags` remains an escape hatch, not the default path.
