# CLI Command Book

This file is the operational command book for SceneWeaver. Commands are kept as single-line PowerShell snippets so they can be copied or run directly from Markdown-capable tooling.

## Environment Checks

Check that the configured LLM endpoint is reachable:

```powershell
python -m sceneweaver.cli llm-check "hi" --timeout-seconds 30
```

Check a specific DashScope / Bailian compatible model:

```powershell
python -m sceneweaver.cli llm-check "hi" --model qwen3.7-max --timeout-seconds 30
```

Useful environment variables:

```powershell
$env:DASHSCOPE_API_KEY="sk-..."
```

```powershell
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

```powershell
$env:DASHSCOPE_MODEL="qwen3.7-max"
```

```powershell
$env:DASHSCOPE_STREAM_IDLE_TIMEOUT_SECONDS="10"
```

## Local Smoke Tests

Create a mock output package:

```powershell
python -m sceneweaver.cli mock-run --output outputs\mock\quick_check
```

Run the core test suite:

```powershell
python -m pytest
```

Run only retrieval tests:

```powershell
python -m pytest tests\test_retrieval_service.py tests\test_retrieval_usecase.py tests\test_keyword_loop.py -q
```

If Windows temp permissions block pytest, route temp files into the workspace:

```powershell
$env:TMP="D:\WorkSpace\Codex\SceneWeaver\.tmp"; $env:TEMP="D:\WorkSpace\Codex\SceneWeaver\.tmp"; python -m pytest -q
```

## Video Pipeline

Package a real Bilibili video:

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
```

Package with a local subtitle file:

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx --subtitle path\to\subtitle.srt
```

Analyze packaged scenes with a limit:

```powershell
python -m sceneweaver.cli analyze-scenes outputs\film_analysis\BVxxxx --limit 3 --concurrency 1
```

Extract experience cards:

```powershell
python -m sceneweaver.cli extract-experience outputs\film_analysis\BVxxxx
```

Run the minimum real-video loop:

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 1 --concurrency 1
```

Run a larger pipeline batch:

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BVxxxx" --limit 40 --concurrency 5
```

## Retrieval

Offline retrieval from one film output directory. This does not call the LLM:

```powershell
python -m sceneweaver.cli retrieve-cards outputs\film_analysis\BVxxxx "科技向善，可信赖，面对面沟通" --top-k 5
```

Keyword loop over all film outputs. Default mode calls the full association LLM prompt:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善，可信赖，面对面沟通" --top-k 5 --result-output outputs\keyword_loop_result.json --debug
```

Lightweight tag expansion. This still calls the LLM, but asks for less than full association:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善，可信赖，面对面沟通" --just-tags --top-k 5 --result-output outputs\keyword_loop_result.json --debug
```

Creative-intent mode:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善，可信赖，面对面沟通" --intent --top-k 5 --result-output outputs\keyword_loop_result.json --debug
```

Stream response and thinking output for diagnostics. If no reasoning or answer chunk arrives within `DASHSCOPE_STREAM_IDLE_TIMEOUT_SECONDS`, the request fails and reports a ping result:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "科技向善，可信赖，面对面沟通" --just-tags --stream --thinking --thinking-budget 4000 --timeout-seconds 180 --retries 0 --debug
```

Enable local semantic reranking:

```powershell
python -m sceneweaver.cli keyword-loop outputs\film_analysis "年轻人逆光奔跑" --semantic --semantic-weight 4 --debug
```

## Creative Association

Generate association material only, without retrieving experience cards:

```powershell
python -m sceneweaver.cli associate "科技向善，可信赖，面对面沟通" --output outputs\key_associates\test.json
```

Stream association with thinking diagnostics:

```powershell
python -m sceneweaver.cli associate "科技向善，可信赖，面对面沟通" --stream --thinking --thinking-budget 4000 --timeout-seconds 180 --retries 0 --debug
```

Reduce association output size when the provider is slow:

```powershell
python -m sceneweaver.cli associate "科技向善，可信赖，面对面沟通" --max-items 16 --timeout-seconds 180 --retries 0 --debug
```

## Legacy Maintenance

Backfill tags into old `analysis\scenes.json` files:

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BVxxxx
```

Force-refresh tags:

```powershell
python -m sceneweaver.cli fingerprint-scenes outputs\film_analysis\BVxxxx --update
```
