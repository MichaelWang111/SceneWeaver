# SceneWeaver

SceneWeaver 是一个面向商业宣传片、招聘宣传片和品牌短片的导演经验分析系统。

它的目标不是直接生成视频，而是把已有视频中的导演经验拆解、结构化、存储，并在未来支持根据关键词生成多份导演稿。

## 1. 项目定位

SceneWeaver 当前阶段聚焦：

```text
视频输入
→ 场景拆分
→ Scene LLM 解析
→ 全片导演语言分析
→ 导演经验卡片存储
```

长期目标是：

```text
关键词
→ 底层情感
→ 叙事逻辑
→ 拍摄技法
→ 导演稿生成
```

例如用户输入：

```text
青春 / 热情 / 梦想 / 真实感
```

系统应能检索历史视频中的导演经验，推导出对应的情绪、叙事、技法和导演表达，而不是只生成普通广告文案。

## 2. 第一阶段范围

v1 只做“视频到导演经验”的数据生产线。

### 2.1 包含

1. Bilibili 视频下载。
2. 场景切分。
3. 每个 scene 抽取前、中、后 3 帧。
4. 提取或切分字幕片段。
5. 将 3 帧图像、字幕、时间范围合并为 scene package。
6. 并发调用 Vision LLM 进行 scene 解析。
7. 汇总生成 `scenes.json`。
8. 再次调用 LLM 生成全片分析 `film_analysis.json`。
9. 抽取可复用导演经验卡片 `experience_cards.jsonl`。

### 2.2 暂不包含

1. 多平台爬虫。
2. Web UI。
3. 视频生成。
4. LoRA / fine-tune。
5. 完整 Graph RAG。
6. 多 Agent 自动化。

## 3. 核心数据流

```text
Bilibili URL
→ yt-dlp download video / subtitle / audio
→ PySceneDetect split scenes
→ ffmpeg sample start / middle / end frames
→ build scene_package
→ concurrent Vision LLM scene analysis
→ scenes.json
→ chronological full-film analysis
→ film_analysis.json
→ experience card extraction
→ experience_cards.jsonl
```

## 4. 技术栈

项目优先使用 Python 技术栈。

1. Python 3.11+
2. Pydantic：schema validation
3. Typer：CLI
4. yt-dlp：视频下载
5. PySceneDetect：场景切分
6. ffmpeg / ffprobe：帧抽取、音频和字幕处理
7. asyncio + httpx：并发 LLM 调用
8. OpenAI-compatible API client：Vision LLM 接入
9. JSON / JSONL：v1 本地存储
10. pytest：测试

## 5. 文档

1. [开发计划](docs/PLAN.md)
2. [产品需求](docs/PRD.md)
3. [技术设计](docs/TECHNICAL_DESIGN.md)
4. [数据结构](docs/SCHEMA.md)
5. [执行情况](docs/EXECUTION_STATUS.md)
6. [背景摘要](docs/CONTEXT_SUMMARY.md)
7. [路线图](docs/ROADMAP.md)
8. [参考项目笔记](docs/REFERENCE_NOTES.md)

## 6. 当前判断

SceneWeaver 的核心壁垒不是视频下载、切分或调用 LLM，而是“导演经验结构化”。

第一阶段的成功标准是：

```text
能否从视频中稳定提取出可复用的导演经验卡片。
```

只有这个成立，后续的关键词检索、情感温度随机化、多版本导演稿生成才有可靠基础。
