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
→ film_analysis.json
→ experience_cards.jsonl
```

当前只做到 scene-level LLM 代码实现，后半段真实链路仍待实现。

## 当前状态

```text
已完成：schema、mock pipeline、真实视频 package pipeline、scene-level LLM 代码、associate 命令
待验收：真实 API scene 分析小样本
待实现：字幕自动获取、full-film analysis、experience card extraction
```

## 核心产物

1. `scene_package`：送入 Vision LLM 的最小输入单元。
2. `scene_analysis`：单个 scene 的导演分析。
3. `scenes.json`：全片所有 scene 分析集合。
4. `film_analysis.json`：全片导演语言总结。
5. `experience_cards.jsonl`：可复用导演经验卡片。

## 下一步

1. 修环境可复现和 CLI help 问题。
2. 跑真实 API `analyze-scenes --limit 1`。
3. 小样本通过后实现 `film_analyzer`。
4. 再实现 `experience_extractor`。
