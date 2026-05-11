# 产品需求文档

## 1. 产品定位

SceneWeaver 是一个从视频案例中提炼导演经验，并为未来导演稿生成提供知识基础的分析系统。

核心价值：

```text
把商业导演经验变成可检索、可组合、可生成的知识。
```

## 2. 当前 v1 范围

v1 聚焦：

1. Bilibili URL 输入。
2. 视频下载和 metadata 获取。
3. scene 检测。
4. start / middle / end 三帧抽取。
5. scene package 生成。
6. scene-level Vision LLM 分析。
7. analysis 内嵌 tags。
8. experience card 抽取。
9. 本地 JSON / JSONL 存储。

## 3. 核心对象

### SceneAnalysis.tags

用于把 scene analysis 归一到半封闭 canonical tags，支持后续检索和经验卡片匹配。

### ExperienceCard.tags

用于表示可复用导演经验的语义坐标，是检索和生成阶段的核心知识单元。

## 4. Tag 治理原则

1. 主数据只写 canonical tags。
2. 同义词、近义词和中文表达进入 aliases。
3. 新表达先进入 candidate pool。
4. 高频、有区分度、不能被已有 tag 覆盖的 candidate 才升级为 canonical tag。
5. 旧 tag 合并时记录 deprecated mapping。

## 5. 暂不包含

1. Web UI。
2. 多平台下载。
3. 视频生成。
4. 生产级数据库。
5. 向量库。
6. 完整故事板生成。

## 6. 成功标准

1. 一个真实视频可以产出 `analysis/scene_XXX.json` 和 `analysis/experience_cards.jsonl`。
2. 每个 scene analysis 都包含 tags。
3. 每张 experience card 都包含 tags。
4. brief 能转成 query_tags 并召回相关 cards。
5. 标签集可迭代，但不会被同义词和近义词污染。
