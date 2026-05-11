# 开发日志

## 2026-05-11：核心架构精简

本轮目标是解决 schema boom，将核心语义链路从旧的 independent fingerprint artifacts 收敛为：

```text
SceneAnalysis.tags
ExperienceCard.tags
```

已完成：

1. 新增主 schema：`TagProfile`、`TagEvidence`。
2. `SceneAnalysis` 内嵌 `tags`。
3. `ExperienceCard` 使用 `tags`，旧 `fingerprint` 字段仅读取兼容。
4. `AssociationAnalysis` 使用 `query_tags`，旧 `query_fingerprint` 仅读取兼容。
5. mock/run 主链路不再写新的 `fingerprints/` 目录。
6. `fingerprint-scenes` 改为 legacy 过渡命令，用于给旧 analysis 补 tags。
7. 新增 `taxonomy/director_tags_v1.json` 作为标签词表治理结构预留。

验证：

```text
50 passed
```

命令：

```powershell
python -m pytest -q --basetemp .tmp\pytest-tags -p no:cacheprovider
```

## 历史说明

2026-05-09 之前曾使用 fingerprint 命名探索低维语义坐标。该命名现在视为 legacy compatibility；新主接口统一为 tags。
