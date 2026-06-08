# SceneWeaver

SceneWeaver is a local Python toolkit for extracting reusable directing experience from commercial videos, recruiting films, and branded short-form videos.

It does not generate video directly. It turns existing video cases into scene packages, scene analysis, tags, and searchable experience cards.

## Current Pipeline

```text
Bilibili URL
-> video packaging
-> scene packages
-> Vision LLM scene analysis
-> embedded tags
-> experience cards
-> keyword-loop retrieval
```

Core output:

```text
outputs/film_analysis/<video_id>/
  source/
  frames/
  packages/
  analysis/
```

## Install

Base development setup:

```powershell
python -m pip install -e ".[dev,video]"
```

Optional semantic retrieval:

```powershell
python -m pip install -e ".[semantic]"
```

`semantic` installs `sentence-transformers`. The default embedding model is:

```text
BAAI/bge-small-zh-v1.5
```

## Main Commands

Create mock output:

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

Package a real video:

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
```

Analyze scenes:

```powershell
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BVxxxx --limit 20 --concurrency 3
```

Extract experience cards:

```powershell
python -m sceneweaver.cli extract-experience outputs\film_analysis\BVxxxx
```

Run the end-to-end video pipeline:

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 40 --concurrency 5
```

## Keyword Loop

Search experience cards from all film outputs:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --result-output outputs\loop_result.json --debug
```

Use lightweight tag-only expansion for fast retrieval tests:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "成熟大型商业与科技公司全球推广与招聘宣传片，比如甲骨文、腾讯 / 致力于为人们创造更好生活 / 以与视频受众面对面交流对话为主 / 科技向善 / 致力于创意有趣 / 提供发挥潜力的机会" --just-tags --stream --thinking --thinking-budget 1024 --result-output outputs\loop_result.json --debug
```

Use core creative-intent analysis when the query should be ranked by what the creator is really looking for, instead of broad tag expansion:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "成熟大型商业与科技公司全球推广与招聘宣传片，比如甲骨文、腾讯 / 致力于为人们创造更好生活 / 以与视频受众面对面交流对话为主 / 科技向善 / 致力于创意有趣 / 提供发挥潜力的机会" --intent --stream --thinking --thinking-budget 1024 --result-output outputs\loop_result.json --debug
```

Enable local semantic reranking:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --semantic --debug
```

Stream provider output and reasoning traces for testing:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --stream --thinking --thinking-budget 1024 --debug
```

The first `keyword-loop` argument can be:

- a collection directory such as `outputs\film_analysis`;
- one film output directory such as `outputs\film_analysis\BVxxxx`;
- one explicit `experience_cards.jsonl` file.

## Retrieval Model

Retrieval has two layers:

1. deterministic tag scoring for explainable matching;
2. optional core-intent scoring for creator must-match and avoid terms;
3. optional local embedding reranking for softer creative similarity.

No vector database is required at the current scale. Semantic scoring is done in memory.

## Documentation

See [docs/README.md](docs/README.md).

Key files:

- [Usage](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Schema](docs/SCHEMA.md)
- [Roadmap](docs/ROADMAP.md)

## Principles

- Store local JSON / JSONL artifacts first; add databases later only when needed.
- Keep canonical tags semi-closed and governed.
- Put new unmanaged expressions into `tag_candidates.jsonl`.
- Preserve source evidence for every reusable experience.
- Keep LLM outputs behind Pydantic validation.
