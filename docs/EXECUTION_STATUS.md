# 执行状态

更新时间：2026-05-11

## 1. 当前结论

项目正在完成核心架构精简：

```text
旧：scene analysis → independent fingerprint artifacts → cards
新：scene analysis with tags → experience cards with tags
```

当前代码已经开始使用 `SceneAnalysis.tags` 和 `ExperienceCard.tags`，并停止在 mock/run 主链路中写出新的 `fingerprints/` 目录。

## 2. 已成立

1. 真实视频 package pipeline。
2. scene-level Vision LLM analysis。
3. analysis 内嵌 tags。
4. experience cards 内嵌 tags。
5. brief 到 `query_tags` 的 card 检索。
6. 旧 `fingerprint` 字段读取兼容。

## 3. 过渡状态

1. `fingerprint-scenes` 保留为 legacy 命令，实际行为是给旧 analysis 补 tags。
2. 旧 `fingerprints/` 目录不再是主数据。
3. `FilmAnalysis` 仍有 schema/mock 兼容，但暂停作为核心链路。
4. `AssociationAnalysis` 保留为独立创意工具，输出 `query_tags`。

## 4. 测试状态

已通过：

```text
50 passed
```

命令：

```powershell
python -m pytest -q --basetemp .tmp\pytest-tags -p no:cacheprovider
```

说明：

Windows 默认 Temp 目录在当前环境存在权限问题，测试应使用项目内临时目录。

## 5. 已知问题

1. 需要继续抽查真实样本的 tags 质量。
2. taxonomy 仍在代码中维护，后续应迁移到版本化词表文件。
3. candidate pool 结构尚未落地。
4. Typer / Click 仍有 deprecation warnings，需要后续依赖整理。
