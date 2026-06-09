# Usage Guide

This guide explains which workflow to use. For runnable one-line commands, use the [CLI Command Book](CLI.md).

## 1. Environment Validation

Use this when setting up a machine, changing model providers, or debugging API timeouts.

Primary command:

```powershell
python -m sceneweaver.cli llm-check "hi" --timeout-seconds 30
```

Expected result:

```json
{
  "reply": "..."
}
```

If this fails, fix API key, base URL, model name, or network routing before running scene analysis or keyword-loop.

## 2. Local Smoke Test

Use this when checking whether the repo and schemas work without external APIs.

Primary command:

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

This should create:

```text
outputs\mock\quick_check\analysis\experience_cards.jsonl
```

## 3. Real Video Ingestion

Use this when adding a new video case into the experience library.

Minimal path:

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 1 --concurrency 1
```

Production-style path:

```text
package-video
-> analyze-scenes
-> extract-experience
```

Run the split steps when you need to inspect artifacts between phases or rerun only one phase.

## 4. Experience Retrieval

Use offline retrieval when validating existing cards or testing retrieval quality without LLM risk.

```powershell
python -m sceneweaver.cli retrieve-cards outputs\film_analysis\BVxxxx "科技向善，可信赖，面对面沟通" --top-k 5
```

Use `keyword-loop` when the user brief needs LLM interpretation before retrieval.

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善，可信赖，面对面沟通" --just-tags --top-k 5 --result-output outputs\keyword_loop_result.json --debug
```

Modes:

- default: full association prompt, richer but heavier.
- `--just-tags`: lighter LLM tag expansion.
- `--intent`: compact creative-intent extraction and ranking.
- `--semantic`: local embedding reranking.

## 5. Streaming And Thinking Diagnostics

Use this when DashScope / Qwen requests appear to hang.

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善，可信赖，面对面沟通" --just-tags --stream --thinking --thinking-budget 4000 --timeout-seconds 180 --retries 0 --debug
```

Behavior:

- `reasoning_content` is written to stderr when the provider sends it.
- streamed answer chunks are written to stderr.
- final validated JSON remains on stdout.
- if no reasoning or answer chunk arrives within the stream idle timeout, the request fails and includes a ping result.

Configurable idle timeout:

```powershell
$env:DASHSCOPE_STREAM_IDLE_TIMEOUT_SECONDS="10"
```

## 6. Retrieval Output Review

Important fields in `keyword-loop` output:

- `searched_card_count`
- `matched_card_count`
- `unindexed_scene_dirs`
- `top_matches[].score`
- `top_matches[].tag_score`
- `top_matches[].usecase_score`
- `top_matches[].intent_score`
- `top_matches[].semantic_score`
- `top_matches[].script_stage`
- `top_matches[].creative_purpose`
- `top_matches[].best_usage`
- `top_matches[].risk`

Review quality by asking:

- Did the card match the requested script stage?
- Is the creative purpose useful, not only semantically similar?
- Does the evidence point back to a real source scene?
- Is the risk acceptable for the intended script segment?

## 7. Legacy Backfill

Use only for old outputs that have `analysis\scenes.json` but no embedded tags.

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BVxxxx
```
