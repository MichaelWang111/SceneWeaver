# 背景摘要

本文用于快速交接。详细计划见 `docs/PLAN.md`，执行状态见 `docs/EXECUTION_STATUS.md`。

## 一句话

SceneWeaver 是一个从商业视频中提炼导演经验，并为未来导演稿生成提供知识基础的 Python 项目。

## 项目本质

它不是普通视频分析器，也不是视频生成器。

它要回答的是：

```text
为什么这里要这样拍？
这个 scene 在全片中承担什么职责？
它制造了什么观众感受？
它塑造了什么品牌人格和身份投射？
这条经验未来如何复用？
```

## 当前 v1 链路

```text
Bilibili URL
→ scene detection
→ start / middle / end frames
→ scene package
→ scene-level Vision LLM analysis
→ scenes.json
→ creative fingerprints
→ film_analysis.json
→ experience_cards.jsonl
```

当前真实链路已做到 scene-level Vision LLM analysis 和 creative fingerprints。后半段 full-film analysis、experience card extraction 和 retrieval 仍待实现。

## 当前状态

```text
已完成：schema、mock pipeline、真实视频 package pipeline、scene-level Vision LLM 分析、creative fingerprint、associate 命令
已验收：BV1cWHyzwEKC 真实视频 40 scene analysis + 40 scene fingerprints
待实现：字幕自动获取、full-film analysis、experience card extraction、experience card retrieval
```

## 核心产物

1. `scene_package`：送入 Vision LLM 的最小输入单元。
2. `scene_analysis`：单个 scene 的导演分析。
3. `scenes.json`：全片所有 scene 分析集合。
4. `scene_fingerprint`：单个 scene 的低维语义坐标。
5. `film_fingerprint.json`：全片 fingerprint 聚合。
6. `film_analysis.json`：全片导演语言总结。
7. `experience_cards.jsonl`：可复用导演经验卡片。

## 下一步

1. 修环境可复现和 CLI help 问题。
2. 抽查并固化 `BV1cWHyzwEKC` 作为回归样本。
3. 实现 `film_analyzer`。
4. 实现 `experience_extractor`。
5. 实现 query fingerprint 到 experience cards 的 top-k retrieval。
