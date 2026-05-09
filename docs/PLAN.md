# SceneWeaver 开发计划

## 1. 总目标

SceneWeaver 要建立一条从视频案例到导演经验记忆的生产线。

第一阶段目标不是做完整产品，而是验证：

```text
视频中的导演语言
→ 是否可以结构化
→ 是否可以存储
→ 是否可以被未来生成系统复用
```

## 2. 阶段划分

### 2.1 v0：项目蓝图

交付物：

1. `README.md`
2. `docs/PLAN.md`
3. `docs/PRD.md`
4. `docs/TECHNICAL_DESIGN.md`
5. `docs/SCHEMA.md`
6. `docs/ROADMAP.md`
7. `docs/REFERENCE_NOTES.md`

验收标准：

1. 项目定位清晰。
2. v1 范围清晰。
3. 数据流清晰。
4. 核心 schema 清晰。
5. 后续开发不需要重新讨论大方向。

### 2.2 v1：视频到导演经验

交付物：

1. v1-0 工程骨架、schema 和 mock pipeline。
2. Bilibili 视频下载。
3. scene 切分。
4. 前、中、后 3 帧抽取。
5. 字幕片段切分。
6. scene package 生成。
7. scene-level LLM 解析。
8. full-film LLM 解析。
9. experience card 抽取。

验收标准：

1. 输入 1 个 Bilibili URL，可以生成完整本地分析产物。
2. 每个 scene 都有独立 JSON 输出。
3. 失败 scene 可重试，不需要重跑整部片。
4. `scenes.json` 可以汇总所有 scene 解析。
5. `film_analysis.json` 可以总结全片氛围、基调、节奏、情绪曲线、视觉语言和叙事结构。
6. `experience_cards.jsonl` 可以被关键词检索。

### 2.2.1 v1-0：工程骨架、schema 和 mock pipeline

状态：

```text
已完成
```

已交付：

1. Python 工程骨架。
2. Pydantic schema。
3. example JSON。
4. mock pipeline。
5. CLI `mock-run`。
6. pytest 自动测试。

验收结果：

```text
8 passed
```

当前能力：

```text
mock source
→ scene package
→ scene analysis
→ scenes.json
→ film_analysis.json
→ experience_cards.jsonl
→ validation
```

### 2.2.2 v1-1：真实视频 package pipeline

状态：

```text
已完成真实样本验收
```

交付物：

1. Bilibili URL 输入。
2. yt-dlp 下载视频和 metadata。
3. PySceneDetect scene 切分。
4. ffmpeg 抽取 start / middle / end 三帧。
5. 字幕按 scene 时间范围切片。
6. 生成真实 `packages/*.json`。
7. CLI `package-video`。

验收标准：

1. 输入 1 个 Bilibili URL，可以生成可验证的 scene packages。
2. 字幕缺失时不阻塞 package 生成。
3. 每个 package 都通过 `ScenePackage` validation。
4. 中间产物落盘，可检查、可复跑。

当前已完成代码：

1. `input/bilibili.py`
2. `input/downloader.py`
3. `split/scene_detector.py`
4. `split/frame_sampler.py`
5. `split/subtitle_segmenter.py`
6. `analysis/scene_package_builder.py`
7. `pipeline/package_video.py`
8. CLI `package-video`

测试结果：

```text
15 passed
```

真实样本验收：

```text
BV1pLqnBWEJC
scene_count: 16
frame_count: 48
package_count: 16
manifest: packages/scene_packages.json
```

### 2.2.3 v1-2：LLM 分析和经验抽取

状态：

```text
scene-level LLM 代码已完成，真实 API 小样本验收待执行
```

交付物：

1. Vision LLM client。
2. scene-level 并发分析。
3. retry / cache / timeout。
4. `scenes.json` 汇总。
5. `film_analysis.json`。
6. `experience_cards.jsonl` 自动抽取。

已完成：

1. Vision LLM client。
2. start / middle / end 三帧上传。
3. `prompts/scene_analysis.md`。
4. `SceneAnalysis` validation。
5. `analysis/scene_XXX.json`。
6. `analysis/scenes.json`。
7. CLI `analyze-scenes`。

测试结果：

```text
17 passed
```

### 2.2.4 v1-2a：字幕获取与字幕 fallback

状态：

```text
下一步
```

交付物：

1. Bilibili 字幕自动获取。
2. `source/subtitles.srt`。
3. package 阶段自动字幕切片。
4. 字幕失败时继续生成无字幕 package。
5. 字幕来源 metadata。

### 2.2.5 v1-2b：Scene-level LLM 真实验收

状态：

```text
下一步
```

验收命令：

```powershell
python -m sceneweaver.cli analyze-scenes outputs\BV1pLqnBWEJC --limit 1
```

验收标准：

1. 输出 `analysis/scene_001.json`。
2. 输出通过 `SceneAnalysis` validation。
3. 内容区分客观观察和导演解释。
4. 不包含评分体系字段。
5. 有可复用 `experience_candidates`。

### 2.3 v2：导演经验检索

交付物：

1. 本地 experience card 检索。
2. 关键词到经验卡片的召回。
3. 情绪、叙事、技法的分层检索。
4. 简单 embedding 存储接口。

验收标准：

1. 输入 `青春 / 热情 / 梦想`，能召回相关经验卡片。
2. 检索结果能解释对应的底层情感、叙事逻辑和拍摄技法。
3. 检索结果不是简单文本相似，而是尽量保留导演逻辑链路。

### 2.4 v3：导演稿生成

交付物：

1. brief 解析。
2. 情感温度参数。
3. 多版本 Director Treatment 生成。
4. shotlist 生成。
5. 文案语气和音乐方向建议。

验收标准：

1. 同一组关键词能生成 3 个不同情绪温度版本。
2. 每个版本都能说明为什么这样拍。
3. 输出结果比普通 LLM 直接生成更有导演意图和执行性。

## 3. 优先级

### 3.1 P0

1. 数据 schema：已完成。
2. scene package：mock 已完成，真实视频 package 已完成。
3. scene analysis：schema、mock 和 LLM 代码已完成，真实 API 验收待执行。
4. film analysis：schema 和 mock 已完成，LLM 待实现。
5. experience card：schema 和 mock 已完成，自动抽取待实现。

### 3.2 P1

1. Bilibili 下载。
2. PySceneDetect 切分。
3. ffmpeg 三帧抽取。
4. 字幕切片：已有 SRT 切片能力，自动获取待实现。
5. 并发 LLM 调用。
6. 本地 JSON / JSONL 存储：已完成基础 helper。

### 3.3 P2

1. 向量检索。
2. Graph RAG。
3. Web UI。
4. 多平台爬虫。
5. 导演稿生成体验。

## 4. 关键风险

1. LLM 可能把观察和推断混在一起。
2. 三帧可能无法准确判断镜头运动。
3. 字幕缺失会影响 narrative 判断。
4. scene 切分过细会导致上下文不足。
5. schema 过大可能导致解析成本和维护成本过高。
6. 经验卡片如果设计不好，后续检索会退化为普通文本搜索。

## 5. 开发原则

1. 先结构化，再自动化。
2. 先本地文件，再数据库。
3. 先小样本验证，再大规模处理。
4. 先保证 LLM 输出可验证，再追求生成质量。
5. 先证明导演经验有复用价值，再做复杂生成。
## CLI 命令大纲

当前推荐的端到端入口：

```powershell
python -m sceneweaver.cli run "https://www.bilibili.com/video/BV1pLqnBWEJC" --limit 20 --concurrency 3
```

语义约定：

1. `run`：从链接直接跑到 scene 级分析完成，不包含后续全片时序分析
2. 默认输出目录固定为 `outputs/<BV号>`
3. `--limit`：限制本次最多处理多少个 scene
4. `--concurrency`：限制 scene LLM 并发数
5. 默认断点续跑：已有 `analysis/scene_XXX.json` 会跳过
6. `--update`：显式覆盖已有结果

拆分命令仍保留，便于调试：

```powershell
python -m sceneweaver.cli package-video "https://www.bilibili.com/video/BV1pLqnBWEJC" --output outputs\BV1pLqnBWEJC
python -m sceneweaver.cli analyze-scenes outputs\BV1pLqnBWEJC --limit 20 --concurrency 3
python -m sceneweaver.cli analyze-scenes outputs\BV1pLqnBWEJC --limit 20 --concurrency 3 --update
```
