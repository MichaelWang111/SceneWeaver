# 技术设计文档

## 1. 总体架构

SceneWeaver v1 使用 Python 技术栈，采用本地 pipeline 架构。

核心链路：

```text
Input Layer
→ Split Layer
→ Scene Packaging
→ Scene LLM Analysis
→ Full Film Analysis
→ Experience Card Extraction
→ Local Storage
```

## 2. 技术栈

1. Python 3.11+
2. Typer：CLI
3. Pydantic：schema validation
4. yt-dlp：Bilibili 视频下载
5. PySceneDetect：scene detection
6. ffmpeg / ffprobe：帧抽取、音频、字幕处理
7. asyncio + httpx：并发 LLM 请求
8. OpenAI-compatible API：Vision LLM 调用
9. JSON / JSONL：v1 本地存储
10. pytest：测试

## 3. 模块设计

### 3.1 输入层 Input Layer

职责：

1. 接收 Bilibili URL。
2. 下载视频文件。
3. 下载或提取字幕。
4. 提取基础 metadata。
5. 输出 video asset 目录。

建议模块：

```text
src/sceneweaver/input/bilibili.py
src/sceneweaver/input/downloader.py
```

### 3.2 拆分层 Split Layer

职责：

1. 使用 PySceneDetect 切分 scene。
2. 记录每个 scene 的 start / end / duration。
3. 生成 scene index。

建议模块：

```text
src/sceneweaver/split/scene_detector.py
```

### 3.3 Frame Sampling Layer

职责：

1. 每个 scene 抽取 3 帧。
2. start frame：scene 开始附近。
3. middle frame：scene 中间。
4. end frame：scene 结束附近。
5. 输出 frame paths。

注意：

1. 三帧只能辅助推断，不应强迫 LLM 确认复杂镜头运动。
2. 输出中需要保留 `confidence_notes`。

建议模块：

```text
src/sceneweaver/split/frame_sampler.py
```

### 3.4 Subtitle Segmenter

职责：

1. 读取 SRT / VTT 字幕。
2. 按 scene time_range 匹配字幕。
3. 输出 subtitle segment。

建议模块：

```text
src/sceneweaver/split/subtitle_segmenter.py
```

### 3.5 Scene Packaging

职责：

将一个 scene 的所有输入合并为 `scene_package`：

1. scene id
2. time_range
3. frame paths
4. subtitle segment
5. source metadata

建议模块：

```text
src/sceneweaver/analysis/scene_package_builder.py
```

### 3.6 Scene LLM Analysis

职责：

1. 并发读取 scene packages。
2. 调用 Vision LLM。
3. 输出 scene analysis。
4. 做 JSON validation。
5. 支持 retry / cache / timeout。

输出重点：

1. `visual_observation`
2. `director_interpretation`
3. `experience_candidates`
4. `emotion_temperature`

建议模块：

```text
src/sceneweaver/analysis/scene_analyzer.py
src/sceneweaver/llm/client.py
src/sceneweaver/llm/retry.py
```

### 3.7 Full Film Analysis

职责：

1. 输入全部 scene analysis。
2. 按时间顺序分析全片。
3. 输出全片层面的导演语言总结。

输出重点：

1. atmosphere
2. tone
3. rhythm
4. emotional_curve
5. visual_language
6. narrative_structure
7. director_language_summary
8. brand_personality
9. audience_projection

建议模块：

```text
src/sceneweaver/analysis/film_analyzer.py
```

### 3.8 Experience Card Extraction

职责：

1. 从 scene analysis 和 film analysis 中抽取可复用经验。
2. 生成 `experience_cards.jsonl`。
3. 为未来向量化和 Graph RAG 做准备。

建议模块：

```text
src/sceneweaver/analysis/experience_extractor.py
src/sceneweaver/storage/json_store.py
```

## 4. 本地输出目录

建议每个视频生成一个独立目录：

```text
outputs/{video_id}/
  source/
    video.mp4
    audio.m4a
    subtitles.srt
    metadata.json
  scenes/
    scene_001.mp4
    scene_002.mp4
  frames/
    scene_001_start.jpg
    scene_001_middle.jpg
    scene_001_end.jpg
  packages/
    scene_001.json
  analysis/
    scene_001.json
    scenes.json
    film_analysis.json
    experience_cards.jsonl
```

## 5. 并发与可靠性

### 5.1 并发

1. scene-level analysis 可以并发。
2. full-film analysis 必须等待 scene analysis 完成。
3. experience extraction 必须等待 film analysis 完成。

### 5.2 retry

LLM 调用失败时：

1. 重试 2-3 次。
2. 指数退避。
3. 仍失败则保留 failed status。
4. 不阻塞已成功 scene 的落盘。

### 5.3 cache

每个 scene analysis 使用输入 hash 缓存。

如果 scene package 未变化，不重复调用 LLM。

## 6. 关键设计规则

1. `visual_observation` 只写客观观察。
2. `director_interpretation` 写推断和解释。
3. `experience_card` 写可复用知识。
4. 不能把 `scenes.json` 直接当经验库。
5. 所有 LLM 输出必须通过 Pydantic validation。
6. 所有中间结果都应可落盘、可复跑、可检查。

## 7. 未来扩展

1. 多平台输入。
2. 智能视频检索。
3. embedding 检索。
4. vector db。
5. Graph RAG。
6. Director Treatment 生成。
7. Web UI。
8. LoRA / fine-tune。
