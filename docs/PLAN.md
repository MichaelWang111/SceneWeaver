# 开发计划

本文记录当前执行计划。完成度见 `docs/EXECUTION_STATUS.md`。

## 1. 当前目标

把核心解析架构从旧的独立 fingerprint 中间层收敛为：

```text
SceneAnalysis.tags
ExperienceCard.tags
```

核心链路：

```text
Bilibili URL
→ scene packages
→ scene analysis with tags
→ experience cards
→ retrieval
```

## 2. Step 1：Schema 精简

目标：

```text
移除 fingerprints/ 主产物，让 tags 成为 analysis 的一部分。
```

已实施方向：

1. `TagProfile` 替代 `CreativeFingerprint` 成为主接口。
2. `TagEvidence` 替代 `FingerprintEvidence` 成为主接口。
3. `SceneAnalysis` 包含 `tags`。
4. `ExperienceCard` 包含 `tags`。
5. `query_fingerprint` 改为 `query_tags`。
6. `SceneFingerprint` / `FilmFingerprint` 只作为 legacy wrapper 保留。

验收：

```powershell
python -m pytest -q --basetemp .tmp\pytest-tags -p no:cacheprovider
```

## 3. Step 2：标签词表迭代管理

目标：

```text
让 tags 可控生长，而不是让 LLM 无限造词。
```

规则：

1. 正式 analysis/card 只写 canonical tags。
2. 同义词和近义词进入 aliases。
3. 未命中新表达进入 candidate pool。
4. 高频且有检索区分度的 candidate 才升级为 canonical tag。
5. 旧 tag 合并时记录 deprecated mapping。

第一版继续使用代码内 taxonomy，后续可迁移到：

```text
taxonomy/director_tags_v1.json
```

## 4. 暂停项

1. `FilmAnalysis` 暂不作为核心链路推进，只作为报告型派生产物。
2. `AssociationAnalysis` 保留为独立创意联想工具，不进入视频解析核心层。
3. 不引入数据库、向量库或 Web UI。

## 5. 下一步

1. 抽查真实视频 `analysis/scene_XXX.json` 中 tags 质量。
2. 记录高频 unmatched expressions，形成 candidate pool。
3. 建立 taxonomy review 文件格式。
4. 用真实检索结果校准 tag 权重。
