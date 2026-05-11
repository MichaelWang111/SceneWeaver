# 背景摘要

本文用于快速交接。

## 一句话

SceneWeaver 是一个从商业视频中提炼导演经验，并把经验沉淀成可检索卡片的 Python 项目。

## 当前核心

项目核心已经从独立 fingerprint 中间层收敛为：

```text
SceneAnalysis.tags
ExperienceCard.tags
```

`tags` 是半封闭标签集：正式数据只写 canonical tags，新表达进入 aliases / candidate pool / deprecated mappings 的治理流程。

## 当前链路

```text
Bilibili URL
→ scene detection
→ start / middle / end frames
→ scene package
→ scene analysis with tags
→ experience_cards.jsonl
→ retrieve cards by query_tags
```

## 当前产物

1. `packages/scene_XXX.json`：Vision LLM 输入包。
2. `analysis/scene_XXX.json`：单 scene 导演分析，内含 `tags`。
3. `analysis/scenes.json`：全片 scene analysis 汇总。
4. `analysis/experience_cards.jsonl`：可复用导演经验卡片，内含 `tags`。

## 过渡说明

1. 旧 `fingerprint` 字段只作为兼容入口。
2. 旧 `fingerprints/` 目录不再作为主数据写出。
3. `fingerprint-scenes` 是 legacy 命令，用于给旧 analysis 补 tags。
4. `FilmAnalysis` 暂停作为核心链路，后续可作为报告型派生产物恢复。

## 下一步

1. 基于真实样本校验 tags 质量。
2. 建立 taxonomy candidate pool。
3. 将代码内 taxonomy 迁移为版本化词表文件。
4. 校准 card retrieval 权重。
