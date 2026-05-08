# 路线图 Roadmap

## 1. v0：文档和方向确认

目标：

建立项目共识和开发蓝图。

内容：

1. README
2. PLAN
3. PRD
4. TECHNICAL_DESIGN
5. SCHEMA
6. REFERENCE_NOTES

状态：

```text
已完成
```

## 2. v1：Director Experience Analyzer

目标：

跑通“视频到导演经验”的核心 pipeline。

当前状态：

```text
v1-0 已完成，v1-1 已完成真实样本验收，v1-2b 代码完成待真实 API 验收
```

能力：

1. 工程骨架、schema、mock pipeline：已完成。
2. Bilibili 视频输入：已完成真实样本验收。
3. 视频下载：已完成真实样本验收。
4. scene 切分：已完成真实样本验收。
5. 三帧抽取：已完成真实样本验收。
6. 字幕切片：已有 SRT 切片能力，自动获取待实现。
7. scene package 生成：已完成真实样本验收。
8. scene LLM 解析：代码已实现，真实 API 小样本验收待执行。
9. full film LLM 解析：待实现。
10. experience card 抽取：mock 已完成，自动抽取待实现。

输出：

1. `scene_packages/*.json`
2. `scene_analysis/*.json`
3. `scenes.json`
4. `film_analysis.json`
5. `experience_cards.jsonl`

### 2.1 v1-0：工程骨架、schema 和 mock pipeline

状态：

```text
已完成
```

输出：

1. `pyproject.toml`
2. `src/sceneweaver/`
3. `examples/*.json`
4. `tests/*.py`
5. `outputs/*/packages/scene_001.json`
6. `outputs/*/analysis/scene_001.json`
7. `outputs/*/analysis/scenes.json`
8. `outputs/*/analysis/film_analysis.json`
9. `outputs/*/analysis/experience_cards.jsonl`

验收：

```text
pytest: 8 passed
```

### 2.2 v1-1：真实视频 package pipeline

状态：

```text
已完成真实样本验收
```

目标：

输入真实 Bilibili URL，生成可验证的 scene packages。

输出：

1. `source/video.mp4`
2. `source/metadata.json`
3. `source/subtitles.srt`
4. `frames/scene_001_start.jpg`
5. `frames/scene_001_middle.jpg`
6. `frames/scene_001_end.jpg`
7. `packages/scene_001.json`
8. `packages/scene_packages.json`

验收样本：

```text
BV1pLqnBWEJC
scene_count: 16
frame_count: 48
package_count: 16
```

说明：

`scenes/` 目录默认不生成 scene clips。需要 clips 时使用 `--split-video`。

### 2.2.1 v1-2a：字幕获取与字幕 fallback

状态：

```text
下一步
```

目标：

自动获取 Bilibili 字幕并注入 scene packages；字幕失败时保留无字幕 package 生成能力。

### 2.3 v1-2：LLM 分析和经验抽取

状态：

```text
scene-level 代码已实现，full-film 和 card extraction 后续
```

目标：

基于真实 scene packages 生成 scene analysis、film analysis 和 experience cards。

已实现：

1. `analyze-scenes`
2. Vision LLM client
3. 三帧上传
4. `SceneAnalysis` validation
5. `analysis/scenes.json`

下一步：

```text
真实 API 小样本验收
```

## 3. v2：Director Memory Retrieval

目标：

让经验卡片可以被关键词检索。

能力：

1. keyword retrieval。
2. embedding retrieval。
3. 按情感、叙事、技法分层检索。
4. 检索结果解释。

示例：

```text
输入：青春 / 热情 / 梦想
输出：底层情感、叙事逻辑、拍摄技法、视觉符号、文案语气
```

## 4. v3：Director Treatment Generation

目标：

根据关键词和 brief 生成多版本导演稿。

能力：

1. brief 解析。
2. 经验卡片召回。
3. 情感温度随机化。
4. 多版本 treatment。
5. shotlist。
6. music direction。
7. copywriting tone。

示例：

```text
输入：新能源车企 / 校招 / 青春 / 热血 / 不要互联网大厂味
输出：3 版不同情绪温度的导演稿
```

## 5. v4：产品化

目标：

把 pipeline 变成可用工具。

能力：

1. Web UI。
2. 多平台输入。
3. 项目管理。
4. 可视化情绪曲线。
5. 可编辑经验卡片。
6. 可导出 treatment / shotlist。

## 6. v5：高级知识系统

目标：

从经验库升级到导演知识系统。

可能方向：

1. Graph RAG。
2. 视觉符号库。
3. 品牌人格 taxonomy。
4. 情绪曲线 taxonomy。
5. narrative pattern library。
6. Taste Engine。
7. LoRA / fine-tune。

## 7. 长期判断

SceneWeaver 的长期价值不在于自动化视频生成，而在于：

```text
把商业视频为什么有效这件事结构化。
```

只有先建立可复用的导演经验，后续生成才不会退化成普通 prompt 工具。
