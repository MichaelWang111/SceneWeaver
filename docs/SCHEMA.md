# Schema

Code definitions in `src/sceneweaver/schemas/` are authoritative. This document summarizes the active artifacts.

## ScenePackage

Written to:

```text
packages/scene_XXX.json
```

Purpose:

```text
Minimal input unit for Vision LLM scene analysis.
```

Key fields:

- `scene_id`
- `source_video_id`
- `time_range`
- `frames`
- `subtitle_segment`
- `metadata`

## SceneAnalysis

Written to:

```text
analysis/scene_XXX.json
analysis/scenes.json
```

Contains:

- objective visual observation
- director interpretation
- reusable experience candidates
- embedded `tags`

## TagProfile

Used by:

```text
SceneAnalysis.tags
ExperienceCard.tags
AssociationAnalysis.query_tags
RetrievalResult.query_tags
```

Shape:

```json
{
  "emotion_core": [],
  "audience_projection": [],
  "narrative_function": [],
  "interaction_mode": [],
  "visual_motifs": [],
  "symbolic_logic": [],
  "rhythm_pattern": [],
  "custom_tags": [],
  "evidence": [],
  "confidence": 0.75
}
```

Tag dimensions:

- `emotion_core`
- `audience_projection`
- `narrative_function`
- `interaction_mode`
- `visual_motifs`
- `symbolic_logic`
- `rhythm_pattern`

## TagCandidate

Written to:

```text
analysis/tag_candidates.jsonl
keyword_loops/tag_candidates.jsonl
```

Purpose:

```text
Record unmanaged expressions that did not cleanly map to canonical taxonomy aliases.
```

Each row contains:

- `candidate`
- `normalized`
- `source_id`
- `source_type`
- `field`
- `quote`
- `status`
- `suggested_action`
- `note`

## ExperienceCard

Written to:

```text
analysis/experience_cards.jsonl
```

Purpose:

```text
Reusable directing experience unit.
```

Key fields:

- `card_id`
- `source_video_id`
- `source_scene_ids`
- `tags`
- `keywords`
- `underlying_emotion`
- `narrative_logic`
- `director_strategy`
- `shooting_techniques`
- `visual_symbols`
- `copywriting_tone`
- `reuse_condition`
- `confidence`

## KeywordLoopResult

Written when `--result-output` is provided.

Key fields:

- `input_text`
- `association_path`
- `candidate_log_path`
- `experience_cards_paths`
- `searched_card_count`
- `matched_card_count`
- `semantic_enabled`
- `embedding_model`
- `semantic_weight`
- `top_matches`
- `association_analysis`
- `retrieval`

`top_matches` is the compact view for humans. `retrieval.results` keeps the full matched cards.
