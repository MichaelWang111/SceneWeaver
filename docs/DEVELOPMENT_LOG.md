# 开发日志

## 2026-05-08：v1-0 工程骨架、schema 和 mock pipeline

### 本次目标

把 SceneWeaver 从文档规划推进到可运行、可测试、可持续开发的 Python 工程。

### 已完成

1. 创建 Python 工程骨架：
   - `pyproject.toml`
   - `src/sceneweaver/`
   - `examples/`
   - `tests/`
   - `.gitignore`

2. 实现核心 Pydantic schema：
   - `ScenePackage`
   - `SceneAnalysis`
   - `ScenesAnalysis`
   - `FilmAnalysis`
   - `ExperienceCard`

3. 实现本地 JSON / JSONL 存储工具：
   - `read_json`
   - `write_json`
   - `read_jsonl`
   - `write_jsonl`

4. 实现 mock pipeline：
   - 生成 `packages/scene_001.json`
   - 生成 `analysis/scene_001.json`
   - 生成 `analysis/scenes.json`
   - 生成 `analysis/film_analysis.json`
   - 生成 `analysis/experience_cards.jsonl`

5. 实现 CLI：

```powershell
python -m sceneweaver.cli mock-run --output outputs\acceptance_mock
```

6. 完成自动测试：

```text
conda env: video_expert_analyzer
Python: 3.11.15
pytest: 8 passed
```

### 功能层面的当前状态

当前项目已经能证明内部数据链路成立：

```text
mock source
→ scene package
→ scene analysis
→ scenes.json
→ film_analysis.json
→ experience_cards.jsonl
→ Pydantic validation
```

这意味着 SceneWeaver 的核心“导演经验结构化数据合同”已经落地。

### 尚未完成

1. 真实 Bilibili 视频下载。
2. PySceneDetect scene 切分。
3. ffmpeg 三帧抽取。
4. 字幕切片。
5. 真实 scene package builder。
6. Vision LLM 调用。
7. full-film LLM 分析。
8. experience card 自动抽取。

### 下一步

进入 v1-1：

```text
真实视频 package pipeline
```

优先实现：

1. `input/downloader.py`
2. `input/bilibili.py`
3. `split/scene_detector.py`
4. `split/frame_sampler.py`
5. `split/subtitle_segmenter.py`
6. `analysis/scene_package_builder.py`
7. CLI `package-video`

## 2026-05-08：v1-1 真实视频 package pipeline 代码实现

### 本次目标

学习并复用本地 `video_expert_analyzer` 的视频前处理实现方式，把真实视频 package pipeline 拆成 SceneWeaver 自己的模块。

### 参考和复用点

参考项目路径：

```text
D:\WorkSpace\github\video-expert-analyzer
```

已复用的工程策略：

1. 使用 `yt-dlp` 获取视频信息。
2. 使用 `yt-dlp` 下载 mp4 优先格式视频。
3. 使用 PySceneDetect 做 content scene detection。
4. 使用 PySceneDetect CLI `split-video` 输出 scene clips。
5. 使用 ffmpeg 按时间点抽帧。
6. 使用 SRT 解析后按 scene 时间范围匹配字幕。

### 已完成代码

新增模块：

1. `src/sceneweaver/input/bilibili.py`
2. `src/sceneweaver/input/downloader.py`
3. `src/sceneweaver/split/timecode.py`
4. `src/sceneweaver/split/scene_detector.py`
5. `src/sceneweaver/split/frame_sampler.py`
6. `src/sceneweaver/split/subtitle_segmenter.py`
7. `src/sceneweaver/analysis/scene_package_builder.py`
8. `src/sceneweaver/pipeline/package_video.py`

新增 CLI：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\BVxxxx
```

当前代码链路：

```text
Bilibili URL
→ yt-dlp metadata
→ yt-dlp video download
→ PySceneDetect scene detection
→ optional PySceneDetect split-video
→ ffmpeg start / middle / end frames
→ optional SRT subtitle segmentation
→ packages/*.json
→ ScenePackage validation
```

### 自动测试

已新增测试覆盖：

1. Bilibili BV 号和 video id 提取。
2. `yt-dlp` metadata / download 命令形态。
3. SRT 字幕解析和 scene overlap 匹配。
4. ffmpeg 三帧抽取命令形态。
5. `ScenePackage` 构建和写盘。
6. CLI `package-video` 调用。

测试结果：

```text
pytest: 15 passed
```

### 当前限制

本次代码测试阶段避免依赖网络、cookies、平台可用性和长视频处理时间。随后已用真实 Bilibili 样本完成 package pipeline 验收。

当前状态应判断为：

```text
v1-1 代码实现完成，真实视频 package 验收已通过
```

### 下一步

继续完善字幕获取或进入 LLM 分析链路：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BVxxxx" --output outputs\BVxxxx
```

## 2026-05-08：v1-1 真实视频 package pipeline 验收

### 验收样本

```text
BV1pLqnBWEJC
标题：智变之时 | 2025腾讯ConTech大会开场视频
时长：95.851 秒
```

### 运行命令

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BV1pLqnBWEJC" --output outputs\BV1pLqnBWEJC
```

### 产物结果

已生成：

1. `source/video.mp4`
2. `source/metadata.json`
3. `frames/scene_001_start.jpg` 等三帧图。
4. `packages/scene_001.json` 到 `packages/scene_016.json`。
5. `packages/scene_packages.json`。

验收结果：

```text
scene_count: 16
frame_count: 48
package_count: 16
pytest: 15 passed
```

### 重要说明

`scenes/` 目录默认可以为空。当前 `package-video` 默认不切出 scene mp4 clips，只检测 scene 时间段并从原视频抽三帧。这样更快，也更符合 v1-1 package 阶段目标。

如需生成 scene clips，需要显式传入：

```powershell
--split-video
```

### 当前缺口

1. `subtitle_segment` 为空：当前未自动抓取 Bilibili 字幕，也未传入 `--subtitle`。
2. 尚未生成 `analysis/scenes.json`、`film_analysis.json`、`experience_cards.jsonl`：这些属于 v1-2 LLM 分析链路。

### 下一步选择

建议优先进入：

```text
v1-2a：字幕获取与字幕 fallback
```

原因：

字幕会显著影响 narrative、brand message 和 audience projection 判断。没有字幕也可以做视觉分析，但导演经验抽取会缺少语义线索。

之后再进入：

```text
v1-2b：Vision LLM scene analysis
```

1. 是否成功下载 `source/video.mp4`。
2. 是否成功生成 scene spans。
3. 是否成功抽取三帧。
4. 是否成功写出 `packages/*.json`。
5. 所有 package 是否通过 `ScenePackage` validation。

## 2026-05-08：v1-2b Vision LLM scene analysis 代码实现

### 本次目标

学习并复用 `video_expert_analyzer` 的视觉 LLM API 调用方式，但将输出目标从“镜头打分”改为 SceneWeaver 的“规格化导演赏析”。

### 参考和复用点

已复用：

1. OpenAI-compatible API client。
2. 通过环境变量配置 API key、base URL 和 model。
3. 将本地帧图转成 base64 data URL。
4. 使用 chat completions 发送 text + image_url。
5. 从模型返回中提取 JSON object。

未复用：

1. 五维评分。
2. weighted_score。
3. MUST KEEP / USABLE / DISCARD。
4. best shots 筛选。

SceneWeaver 的分析目标是：

```text
客观视觉观察
→ 导演解释
→ 可复用经验候选
→ SceneAnalysis validation
```

### 已完成代码

新增：

1. `src/sceneweaver/llm/client.py`
2. `src/sceneweaver/analysis/scene_analyzer.py`
3. `prompts/scene_analysis.md`
4. CLI `analyze-scenes`

CLI：

```powershell
python -m sceneweaver.cli analyze-scenes outputs\BV1pLqnBWEJC --limit 1
```

环境变量：

```powershell
$env:SCENEWEAVER_API_KEY="..."
$env:SCENEWEAVER_BASE_URL="https://..."
$env:SCENEWEAVER_MODEL="..."
```

同时兼容已有环境变量：

```powershell
VIDEO_ANALYZER_API_KEY
VIDEO_ANALYZER_BASE_URL
VIDEO_ANALYZER_MODEL
```

### 输出

生成：

1. `analysis/scene_001.json`
2. `analysis/scenes.json`

所有 LLM 输出必须通过 `SceneAnalysis` / `ScenesAnalysis` Pydantic validation 才会落盘。

### 自动测试

已用 mocked LLM client 验证：

1. 能读取 `packages/scene_001.json`。
2. 能解析 start / middle / end 三帧路径。
3. 能把模型返回验证为 `SceneAnalysis`。
4. 能写出 `analysis/scene_001.json`。
5. 能写出 `analysis/scenes.json`。
6. 已存在分析时默认跳过，支持缓存式复跑。

测试结果：

```text
pytest: 17 passed
```

### 当前状态

```text
v1-2b scene-level LLM 分析代码已完成，真实 API 小样本验收待执行
```

### 下一步

使用真实视觉模型先跑 1 个 scene：

```powershell
python -m sceneweaver.cli analyze-scenes outputs\BV1pLqnBWEJC --limit 1
```

验收重点：

1. 模型是否能读取三帧。
2. 输出是否严格符合 `SceneAnalysis` schema。
3. 是否能区分 `visual_observation` 和 `director_interpretation`。
4. 是否没有退化成打分报告。
5. 是否能写出可复用的 `experience_candidates`。

1. 是否成功下载 `source/video.mp4`。
2. 是否成功生成 scene spans。
3. 是否成功抽取三帧。
4. 是否成功写出 `packages/*.json`。
5. 所有 package 是否通过 `ScenePackage` validation。
