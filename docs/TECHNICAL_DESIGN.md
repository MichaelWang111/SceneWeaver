# 技术设计

本文描述当前架构和模块职责。执行状态见 `docs/EXECUTION_STATUS.md`，schema 细节见 `docs/SCHEMA.md`。

## 1. 总体架构

SceneWeaver v1 使用本地 Python pipeline。

```text
Input
→ Split
→ Scene Packaging
→ Scene LLM Analysis
→ Full Film Analysis
→ Experience Card Extraction
→ Local Storage
```

当前真实链路做到：

```text
Input
→ Split
→ Scene Packaging
→ Scene LLM Analysis code
```

Full Film Analysis 和 Experience Card Extraction 仍待接入真实链路。

## 2. 技术栈

1. Python 3.11+
2. Typer：CLI
3. Pydantic：schema validation
4. yt-dlp：Bilibili 视频下载
5. PySceneDetect：scene detection
6. ffmpeg / ffprobe：帧抽取和视频信息读取
7. OpenAI-compatible API：Vision / text LLM 调用
8. JSON / JSONL：v1 本地存储
9. pytest：测试

注意：

当前需要补依赖锁定。已观察到 `typer 0.9.0 + click 8.2.1` 下 CLI `--help` 可能报错。

## 3. 模块职责

### 3.1 CLI

文件：

```text
src/sceneweaver/cli.py
```

命令：

1. `mock-run`
2. `package-video`
3. `analyze-scenes`
4. `associate`
5. `run`

当前 `run` 只串到 scene-level analysis，未包含 full-film 和 card extraction。

### 3.2 Input

文件：

```text
src/sceneweaver/input/bilibili.py
src/sceneweaver/input/downloader.py
```

职责：

1. 从 URL 提取 BV 号。
2. 调用 `yt-dlp` 获取 metadata。
3. 下载视频到 `source/video.mp4`。
4. 写出 `source/metadata.json`。

待补：

1. 自动字幕获取。
2. 字幕失败原因记录。

### 3.3 Split

文件：

```text
src/sceneweaver/split/scene_detector.py
src/sceneweaver/split/frame_sampler.py
src/sceneweaver/split/subtitle_segmenter.py
src/sceneweaver/split/timecode.py
```

职责：

1. 使用 PySceneDetect 检测 scene spans。
2. 使用 ffmpeg 抽取 start / middle / end 三帧。
3. 解析已有 SRT 并按 scene 时间范围匹配字幕。
4. 在没有检测到 scene 时 fallback 到整段视频。

设计约束：

1. 三帧只能辅助判断，不强迫模型确认复杂镜头运动。
2. 字幕缺失不能阻塞 package 生成。

### 3.4 Scene Packaging

文件：

```text
src/sceneweaver/analysis/scene_package_builder.py
src/sceneweaver/pipeline/package_video.py
```

职责：

1. 合并 scene span、frame paths、subtitle segment 和 source metadata。
2. 写出 `packages/scene_XXX.json`。
3. 写出 `packages/scene_packages.json` manifest。

### 3.5 LLM Client

文件：

```text
src/sceneweaver/llm/client.py
```

职责：

1. 从环境变量读取 API key、base URL、model。
2. 支持 text JSON 请求。
3. 支持 Vision image JSON 请求。
4. 提取并解析 LLM 返回的 JSON object。
5. 支持文本流式输出和 reasoning 内容分离。

待补：

1. `analyze_images_json` 的 retry / timeout。
2. 更稳健的 JSON 提取策略。
3. 针对真实 provider 的错误分类。

### 3.6 Scene LLM Analysis

文件：

```text
src/sceneweaver/analysis/scene_analyzer.py
prompts/scene_analysis.md
```

职责：

1. 读取 `packages/scene_XXX.json`。
2. 解析三帧路径。
3. 调用 Vision LLM。
4. 校验 `SceneAnalysis`。
5. 写出 `analysis/scene_XXX.json`。
6. 汇总 `analysis/scenes.json`。
7. 支持并发和已有结果复用。

### 3.7 Associate Analysis

文件：

```text
src/sceneweaver/analysis/associate_analyzer.py
prompts/associate.md
```

职责：

1. 将关键词或粗糙 brief 扩展成导演/编剧联想材料。
2. 输出 `AssociationAnalysis`。
3. 支持 debug、stream、thinking 参数。

该能力当前独立于视频经验库。

### 3.8 Full Film Analysis

计划文件：

```text
src/sceneweaver/analysis/film_analyzer.py
prompts/film_analysis.md
```

职责：

1. 输入 `analysis/scenes.json`。
2. 按时间顺序分析全片。
3. 输出 `analysis/film_analysis.json`。
4. 校验 `FilmAnalysis`。

### 3.9 Experience Card Extraction

计划文件：

```text
src/sceneweaver/analysis/experience_extractor.py
prompts/experience_extraction.md
```

职责：

1. 输入 `analysis/scenes.json` 和 `analysis/film_analysis.json`。
2. 抽取可复用导演经验。
3. 输出 `analysis/experience_cards.jsonl`。
4. 校验每个 `ExperienceCard`。

## 4. 输出目录

```text
outputs/film_analysis/<BV号>/
  source/
    video.mp4
    metadata.json
    subtitles.srt
  scenes/
    scene_001.mp4
  frames/
    scene_001_start.jpg
    scene_001_middle.jpg
    scene_001_end.jpg
  packages/
    scene_001.json
    scene_packages.json
  analysis/
    scene_001.json
    scenes.json
    film_analysis.json
    experience_cards.jsonl
```

说明：

1. `scenes/` 默认可以为空，只有 `--split-video` 时才生成 clips。
2. `subtitles.srt` 目前不会自动生成。
3. `film_analysis.json` 和 `experience_cards.jsonl` 的真实链路待实现。

## 5. 关键设计规则

1. 所有 LLM 输出必须通过 Pydantic validation。
2. `visual_observation` 只写客观观察。
3. `director_interpretation` 写推断和解释。
4. `experience_card` 写可复用知识，不写原始报告摘要。
5. 不把 `scenes.json` 直接当经验库。
6. 所有中间结果都应可落盘、可复跑、可检查。
