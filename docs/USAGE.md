# Usage

## Install

Base development install:

```powershell
python -m pip install -e ".[dev,video]"
```

Optional semantic retrieval install:

```powershell
python -m pip install -e ".[semantic]"
```

`semantic` installs `sentence-transformers`. It reuses the existing PyTorch/CUDA environment when available.

## Pipeline Commands

Create mock artifacts:

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

Package a real Bilibili video:

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
```

Analyze packaged scenes:

```powershell
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BVxxxx --limit 20 --concurrency 3
```

Extract experience cards:

```powershell
python -m sceneweaver.cli extract-experience outputs\film_analysis\BVxxxx
```

Run the full video pipeline:

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 40 --concurrency 5
```

## Keyword Loop

Search all generated film experience cards:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --result-output outputs\loop_result.json --debug
```

The first argument can be:

- `outputs\film_analysis`: recursively search every `analysis\experience_cards.jsonl` below it.
- `outputs\film_analysis\BVxxxx`: search one film output directory.
- `outputs\film_analysis\BVxxxx\analysis\experience_cards.jsonl`: search one explicit card file.

The loop performs:

```text
keyword brief
-> LLM association analysis
-> query_tags
-> tag candidate logging
-> experience-card retrieval
-> optional semantic reranking
```

Use lightweight tag-only expansion when you want faster retrieval tests and do not need scene ideas or full directing associations:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "成熟大型商业与科技公司全球推广与招聘宣传片，比如甲骨文、腾讯 / 致力于为人们创造更好生活 / 以与视频受众面对面交流对话为主 / 科技向善 / 致力于创意有趣 / 提供发挥潜力的机会" --just-tags --stream --thinking --thinking-budget 1024 --result-output outputs\loop_result.json --debug
```

`--just-tags` asks the LLM only for tag-level expansion:

- expanded terms;
- dimension hints;
- avoid terms.

It does not ask for scenes, story material, shot ideas, or director possibilities.

Use core creative-intent analysis when broad tag expansion is too loose and the ranking should follow the creator's real selection intent:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "成熟大型商业与科技公司全球推广与招聘宣传片，比如甲骨文、腾讯 / 致力于为人们创造更好生活 / 以与视频受众面对面交流对话为主 / 科技向善 / 致力于创意有趣 / 提供发挥潜力的机会" --intent --stream --thinking --thinking-budget 1024 --result-output outputs\loop_result.json --debug
```

`--intent` is also available as `--core-intent`. It asks the LLM for:

- primary creative intent;
- must-match conditions;
- nice-to-have conditions;
- avoid conditions;
- target audience and selection criteria.

It does not ask for scenes, story material, shot ideas, or broad tag expansion. Results include `intent_analysis`, `intent_weight`, and `top_matches[].intent_score`.

## Streaming and Thinking

Stream raw provider chunks to stderr:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --stream --debug
```

Stream provider reasoning content when the model/API supports it:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --thinking --thinking-budget 1024 --debug
```

Both modes keep the final validated JSON on stdout.

## Semantic Retrieval

Enable local embedding reranking:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --semantic --debug
```

Default model:

```text
BAAI/bge-small-zh-v1.5
```

Try the larger model explicitly:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --semantic --embedding-model BAAI/bge-base-zh-v1.5
```

Control semantic influence:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --semantic --semantic-weight 4
```

Output includes:

- `mode`
- `searched_card_count`
- `matched_card_count`
- `unindexed_scene_dirs`
- `semantic_enabled`
- `embedding_model`
- `intent_weight`
- `top_matches[].tag_score`
- `top_matches[].usecase_score`
- `top_matches[].intent_score`
- `top_matches[].quality_score`
- `top_matches[].semantic_score`
- `top_matches[].script_stage`
- `top_matches[].creative_purpose`
- `top_matches[].best_usage`
- `top_matches[].risk`
- `top_matches[].score`

If `unindexed_scene_dirs` is not empty, those videos have `analysis/scenes.json` but do not yet have `analysis/experience_cards.jsonl`. Run `extract-experience` on those film directories before judging retrieval quality.

## Legacy Command

`fingerprint-scenes` is kept only to backfill tags into older `analysis/scenes.json` files:

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BVxxxx
```
