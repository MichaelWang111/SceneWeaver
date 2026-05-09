# 开发日志

本文按时间记录已发生的工程推进。当前状态以 `docs/EXECUTION_STATUS.md` 为准。

## 2026-05-08：v1-0 工程骨架、schema 和 mock pipeline

目标：

把 SceneWeaver 从文档规划推进到可运行、可测试、可持续开发的 Python 工程。

完成：

1. 创建 Python 工程骨架：`pyproject.toml`、`src/sceneweaver/`、`examples/`、`tests/`。
2. 实现核心 Pydantic schema：`ScenePackage`、`SceneAnalysis`、`ScenesAnalysis`、`FilmAnalysis`、`ExperienceCard`。
3. 实现 JSON / JSONL 存储 helper。
4. 实现 mock pipeline。
5. 实现 CLI `mock-run`。
6. 完成初始自动测试。

mock 链路：

```text
mock source
→ scene package
→ scene analysis
→ scenes.json
→ film_analysis.json
→ experience_cards.jsonl
→ Pydantic validation
```

阶段判断：

```text
v1-0 完成，内部数据合同成立。
```

## 2026-05-08：v1-1 真实视频 package pipeline

目标：

把真实 Bilibili 视频转换为可验证的 scene packages。

参考：

```text
D:\WorkSpace\github\video-expert-analyzer
```

完成模块：

1. `src/sceneweaver/input/bilibili.py`
2. `src/sceneweaver/input/downloader.py`
3. `src/sceneweaver/split/timecode.py`
4. `src/sceneweaver/split/scene_detector.py`
5. `src/sceneweaver/split/frame_sampler.py`
6. `src/sceneweaver/split/subtitle_segmenter.py`
7. `src/sceneweaver/analysis/scene_package_builder.py`
8. `src/sceneweaver/pipeline/package_video.py`

CLI：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\film_analysis\BVxxxx
```

真实样本验收：

```text
BV1pLqnBWEJC
scene_count: 16
frame_count: 48
package_count: 16
```

阶段判断：

```text
v1-1 完成真实样本验收。
```

限制：

1. 未自动抓取 Bilibili 字幕。
2. 默认不生成 scene clips，需显式传入 `--split-video`。
3. 产物只到 `packages/`，不包含 LLM 分析。

## 2026-05-08：v1-2 Scene-level Vision LLM 分析代码

目标：

将 scene package 和三帧图送入 Vision LLM，输出规格化导演赏析，而不是镜头评分报告。

完成：

1. `src/sceneweaver/llm/client.py`
2. `src/sceneweaver/analysis/scene_analyzer.py`
3. `prompts/scene_analysis.md`
4. CLI `analyze-scenes`

输出：

1. `analysis/scene_XXX.json`
2. `analysis/scenes.json`

设计约束：

1. 不输出评分。
2. 不输出 `weighted_score`。
3. 不输出 `MUST KEEP` / `USABLE` / `DISCARD`。
4. 严格区分客观观察和导演解释。
5. 所有输出通过 `SceneAnalysis` / `ScenesAnalysis` validation。

阶段判断：

```text
scene-level LLM 代码完成，真实 API 小样本验收待执行。
```

## 2026-05-09：关键词联想和文档整理

完成：

1. `associate` 命令支持关键词或粗糙 brief。
2. 支持 debug、stream、thinking 参数。
3. 输出 `AssociationAnalysis`。
4. 重新整理 README 和 docs 分工，减少 PLAN、ROADMAP、STATUS、SUMMARY 之间的重复。

阶段判断：

```text
associate 可作为独立前期创意联想入口；项目文档重新按职责收敛。
```
