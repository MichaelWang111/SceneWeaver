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
当前阶段
```

## 2. v1：Director Experience Analyzer

目标：

跑通“视频到导演经验”的核心 pipeline。

能力：

1. Bilibili 视频输入。
2. 视频下载。
3. scene 切分。
4. 三帧抽取。
5. 字幕切片。
6. scene package 生成。
7. scene LLM 解析。
8. full film LLM 解析。
9. experience card 抽取。

输出：

1. `scene_packages/*.json`
2. `scene_analysis/*.json`
3. `scenes.json`
4. `film_analysis.json`
5. `experience_cards.jsonl`

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
