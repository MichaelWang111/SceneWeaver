# SceneWeaver

SceneWeaver is a local Python toolkit for turning commercial videos, recruiting films, and branded short-form videos into reusable directing knowledge.

It does not generate video directly. It analyzes existing video cases, extracts scene-level directing interpretation, creates experience cards, and retrieves those cards for future creative work.

## Current Pipeline

```text
Bilibili URL
-> video packaging
-> scene packages
-> Vision LLM scene analysis
-> tags / fingerprint
-> experience cards with script_usecase
-> retrieval by brief
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

## Fast Checks

Create mock output:

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

Check LLM connectivity:

```powershell
python -m sceneweaver.cli llm-check "hi" --timeout-seconds 30
```

Run the minimum real-video pipeline:

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 1 --concurrency 1
```

Run offline retrieval from one analyzed film:

```powershell
python -m sceneweaver.cli retrieve-cards outputs\film_analysis\BVxxxx "科技向善，可信赖，面对面沟通" --top-k 5
```

## Documentation

- [Documentation Index](docs/README.md)
- [CLI Command Book](docs/CLI.md)
- [Usage Guide](docs/USAGE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Schema](docs/SCHEMA.md)
- [Roadmap](docs/ROADMAP.md)

## Principles

- Store local JSON / JSONL artifacts first; add databases later only when needed.
- Keep canonical tags semi-closed and governed.
- Put unmanaged expressions into `tag_candidates.jsonl`.
- Preserve source evidence for every reusable experience.
- Keep LLM outputs behind Pydantic validation.
