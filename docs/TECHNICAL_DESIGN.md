# 技术设计

本文描述当前架构和模块职责。执行状态见 `docs/EXECUTION_STATUS.md`，schema 细节见 `docs/SCHEMA.md`。

## 1. 总体架构

当前核心链路：

```text
Input
→ Split
→ Scene Packaging
→ Scene Analysis with Tags
→ Experience Card Extraction
→ Local Storage
```

核心输出目录：

```text
source/
frames/
packages/
analysis/
```

`tags` 随 `analysis` 落盘，不再作为独立 `fingerprints/` 目录产物。

## 2. 模块职责

### CLI

入口：`src/sceneweaver/cli.py`

当前命令：

1. `mock-run`
2. `package-video`
3. `analyze-scenes`
4. `fingerprint-scenes`：legacy 过渡命令，用于给旧 analysis 补 tags。
5. `extract-experience`
6. `retrieve-cards`
7. `associate`
8. `run`

`run` 当前链路：

```text
package-video
→ analyze-scenes with tags
→ extract-experience
```

### Scene Packaging

负责合并 scene span、frame paths、subtitle segment 和 source metadata，输出 `packages/scene_XXX.json`。

### Scene Analysis with Tags

入口：`src/sceneweaver/analysis/scene_analyzer.py`

职责：

1. 读取 `packages/scene_XXX.json`。
2. 调用 Vision LLM 生成 scene-level director analysis。
3. 根据 analysis 文本和 taxonomy 生成 `tags`。
4. 写出 `analysis/scene_XXX.json`。
5. 汇总 `analysis/scenes.json`。

### Tags

入口：`src/sceneweaver/analysis/tags.py` 与 `src/sceneweaver/analysis/taxonomy.py`

职责：

1. 将 scene/card/query 文本归一到 canonical tags。
2. 维护当前七个 tag 维度。
3. 支持 brief 到 `query_tags`。
4. 支持 `ExperienceCard.tags` 的 top-k overlap 检索。
5. 兼容旧 fingerprint 命名，但新写出统一使用 tags。

### Experience Card Extraction

入口：`src/sceneweaver/analysis/experience_extractor.py`

职责：

1. 读取 `analysis/scenes.json`。
2. 从 scene candidates 生成第一版 cards。
3. 为每张 card 写入 `tags`。
4. 输出 `analysis/experience_cards.jsonl`。

### Associate Analysis

入口：`src/sceneweaver/analysis/associate_analyzer.py`

该能力独立于视频经验库，用于前期创意联想。它输出 `query_tags`，可用于后续检索 experience cards。

## 3. 标签词表治理

半封闭 tags 的治理规则：

```text
raw phrase
→ alias match
→ canonical tag
→ unmatched phrase enters candidate pool
```

正式数据只写 canonical tags。新表达不能由 LLM 直接进入主标签集，必须先进入 candidate pool，再决定：

1. 并入已有 tag 的 aliases。
2. 升级为新的 canonical tag。
3. 标记 deprecated mapping。
4. 丢弃。

## 4. Legacy 兼容

旧 `fingerprint` 字段可读入并迁移到 `tags`。旧 `fingerprints/` 目录不再作为主数据写出，只作为历史产物参考。
