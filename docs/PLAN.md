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

1. Bilibili 视频下载。
2. scene 切分。
3. 前、中、后 3 帧抽取。
4. 字幕片段切分。
5. scene package 生成。
6. scene-level LLM 解析。
7. full-film LLM 解析。
8. experience card 抽取。

验收标准：

1. 输入 1 个 Bilibili URL，可以生成完整本地分析产物。
2. 每个 scene 都有独立 JSON 输出。
3. 失败 scene 可重试，不需要重跑整部片。
4. `scenes.json` 可以汇总所有 scene 解析。
5. `film_analysis.json` 可以总结全片氛围、基调、节奏、情绪曲线、视觉语言和叙事结构。
6. `experience_cards.jsonl` 可以被关键词检索。

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

1. 数据 schema。
2. scene package。
3. scene analysis。
4. film analysis。
5. experience card。

### 3.2 P1

1. Bilibili 下载。
2. PySceneDetect 切分。
3. ffmpeg 三帧抽取。
4. 并发 LLM 调用。
5. 本地 JSON / JSONL 存储。

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
