# 路线图

本文只保留阶段路线，不记录当前执行细节。当前执行计划见 `docs/PLAN.md`，完成度见 `docs/EXECUTION_STATUS.md`。

## v0：项目蓝图

状态：

```text
已完成
```

目标：

1. 明确 SceneWeaver 的产品定位。
2. 明确 v1 范围和非目标。
3. 建立初版 schema、技术设计和参考边界。

## v1：Director Experience Analyzer

状态：

```text
进行中
```

目标：

```text
真实视频
→ scene package
→ scene analysis
→ film analysis
→ experience cards
```

v1 完成后，系统应能从一个真实 Bilibili 视频中稳定提取可复用导演经验卡片。

关键里程碑：

1. v1-0 工程骨架、schema、mock pipeline：已完成。
2. v1-1 真实视频 package pipeline：已完成真实样本验收。
3. v1-2 scene-level LLM 分析：代码已完成，真实 API 小样本待验收。
4. v1-3 full-film analysis：待实现。
5. v1-4 experience card extraction：待实现。
6. v1-5 完整 `run` 闭环：待实现。

## v2：Director Memory Retrieval

目标：

让 `experience_cards.jsonl` 变成可检索的导演经验库。

能力：

1. 关键词检索。
2. 情绪、叙事、技法分层检索。
3. embedding 检索。
4. 检索结果解释。

示例：

```text
输入：青春 / 热情 / 梦想
输出：底层情感、叙事逻辑、拍摄技法、视觉符号、文案语气
```

## v3：Director Treatment Generation

目标：

基于 brief 和经验库生成多版本导演稿。

能力：

1. brief 解析。
2. 经验卡片召回。
3. 情感温度参数。
4. 多版本 treatment。
5. shotlist。
6. music direction。
7. copywriting tone。

示例：

```text
输入：新能源车企 / 校招 / 青春 / 热血 / 不要互联网大厂味
输出：3 版不同情绪温度的导演稿
```

## v4：产品化

目标：

把本地 pipeline 变成可持续使用的工具。

能力：

1. Web UI。
2. 项目管理。
3. 多平台输入。
4. 可视化情绪曲线。
5. 可编辑经验卡片。
6. 可导出 treatment / shotlist。

## v5：导演知识系统

目标：

从经验库升级为导演知识系统。

可能方向：

1. Graph RAG。
2. 视觉符号库。
3. 品牌人格 taxonomy。
4. 情绪曲线 taxonomy。
5. narrative pattern library。
6. Taste Engine。

## 长期判断

SceneWeaver 的长期价值不在于自动化视频生成，而在于：

```text
把商业视频为什么有效这件事结构化。
```
