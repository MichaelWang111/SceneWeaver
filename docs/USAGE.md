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

- `searched_card_count`
- `matched_card_count`
- `semantic_enabled`
- `embedding_model`
- `top_matches[].tag_score`
- `top_matches[].semantic_score`
- `top_matches[].score`

## Legacy Command

`fingerprint-scenes` is kept only to backfill tags into older `analysis/scenes.json` files:

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BVxxxx
```
