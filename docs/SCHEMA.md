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
- `script_usecase`
- `confidence`

## ScriptUseCase

Used by:

```text
ExperienceCard.script_usecase
retrieval results
```

Purpose:

```text
Describe where an experience card fits in a script, not only what tags it matches.
```

Shape:

```json
{
  "script_stage": "team_work",
  "creative_purpose": ["show_team"],
  "best_usage": "Use in a team culture, collaboration, or collective action segment.",
  "risk": "May feel like generic office montage if the collaboration logic is not explicit.",
  "confidence": 0.8
}
```

Allowed `script_stage` values:

- `opening`
- `setup`
- `character_intro`
- `team_work`
- `growth`
- `technology_showcase`
- `value_expression`
- `ending`
- `transition`
- `general`

## KeywordLoopResult

Written when `--result-output` is provided.

Key fields:

- `input_text`
- `mode`
- `association_path`
- `candidate_log_path`
- `experience_cards_paths`
- `unindexed_scene_dirs`
- `searched_card_count`
- `matched_card_count`
- `semantic_enabled`
- `embedding_model`
- `semantic_weight`
- `intent_weight`
- `top_matches`
- `association_analysis`
- `tag_expansion_analysis`
- `intent_analysis`
- `retrieval`

`top_matches` is the compact view for humans. `retrieval.results` keeps the full matched cards.
Each match can include `tag_score`, `usecase_score`, `intent_score`, `quality_score`, `semantic_score`, `matched_usecase`, `script_stage`, `creative_purpose`, `best_usage`, and `risk`.

## TagExpansionAnalysis

Written by `keyword-loop --just-tags`.

Purpose:

```text
Lightweight keyword expansion for retrieval testing.
```

Shape:

```json
{
  "input_text": "...",
  "query_tags": {},
  "expanded_terms": [],
  "tag_hints": {
    "emotion_core": [],
    "audience_projection": [],
    "narrative_function": [],
    "interaction_mode": [],
    "visual_motifs": [],
    "symbolic_logic": [],
    "rhythm_pattern": []
  },
  "expanded_text": "...",
  "avoid_terms": []
}
```

`expanded_text` is derived from `expanded_terms` and `tag_hints`, then passed into `build_query_tags` together with the original input.

## CreativeIntentAnalysis

Written by `keyword-loop --intent` / `--core-intent`.

Purpose:

```text
Compact understanding of what the creator really wants to retrieve, used for ranking experience cards.
```

Shape:

```json
{
  "input_text": "...",
  "query_tags": {},
  "primary_intent": "...",
  "must_match": [],
  "nice_to_have": [],
  "avoid": [],
  "intent_keywords": [],
  "target_audience": [],
  "selection_criteria": [],
  "expanded_text": "..."
}
```

`expanded_text` is derived from the intent fields and passed into `build_query_tags`. The same fields also drive deterministic `intent_score` ranking.
