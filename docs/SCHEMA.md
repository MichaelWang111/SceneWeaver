# 数据结构

本文定义当前核心 JSON / JSONL 数据合同。代码实现以 `src/sceneweaver/schemas/` 为准。

## 1. 当前核心

SceneWeaver 的核心语义层收敛为：

```text
SceneAnalysis.tags
ExperienceCard.tags
```

`tags` 不是独立产物，不再写入 `fingerprints/`。它是 analysis 的一部分，用于检索、归并和经验卡片匹配。

## 2. TagProfile

用途：

```text
半封闭导演语义标签层。
```

字段：

```json
{
  "emotion_core": [],
  "audience_projection": [],
  "narrative_function": [],
  "interaction_mode": [],
  "visual_motifs": [],
  "symbolic_logic": [],
  "rhythm_pattern": [],
  "custom_tags": [],
  "evidence": [],
  "confidence": 0.75
}
```

规则：

1. 主维度只写 canonical tags。
2. 同义词、近义词和中文表达进入 aliases，不直接写入主数据。
3. 未命中新表达进入 candidate pool，后续决定合并、升级或丢弃。
4. `custom_tags` 是半封闭出口，只用于尚未纳入 canonical taxonomy 的少量临时表达。
5. `evidence` 必须指向来源字段。

## 3. TagEvidence

字段：

```json
{
  "source_id": "scene_001",
  "source_type": "scene",
  "field": "director_interpretation.underlying_emotion",
  "quote": "我可以在这里被看见并参与真实工作",
  "note": ""
}
```

## 4. ScenePackage

用途：

```text
送入 Vision LLM 的最小输入单元。
```

关键字段：

1. `scene_id`
2. `source_video_id`
3. `time_range`
4. `frames`
5. `subtitle_segment`
6. `metadata`

## 5. SceneAnalysis

用途：

```text
单个 scene 的导演分析主产物。
```

关键字段：

1. `visual_observation`：客观观察。
2. `director_interpretation`：导演意图、叙事功能、情绪功能、品牌人格和观众投射。
3. `tags`：半封闭 canonical tag profile。
4. `experience_candidates`：候选经验，不等于最终经验卡片。
5. `emotion_temperature`：0 到 1 的情绪强度。

示例结构：

```json
{
  "scene_id": "scene_001",
  "time_range": {},
  "visual_observation": {},
  "director_interpretation": {},
  "tags": {},
  "experience_candidates": [],
  "emotion_temperature": 0.45
}
```

## 6. ScenesAnalysis

用途：

```text
一支视频的 scene analysis 集合。
```

字段：

1. `video_id`
2. `source_url`
3. `scene_count`
4. `scenes`

`scene_count` 必须等于 `scenes` 长度。

## 7. ExperienceCard

用途：

```text
未来检索和生成的核心知识单元。
```

关键字段：

1. `card_id`
2. `source_video_id`
3. `source_scene_ids`
4. `tags`
5. `keywords`
6. `underlying_emotion`
7. `narrative_logic`
8. `director_strategy`
9. `shooting_techniques`
10. `visual_symbols`
11. `copywriting_tone`
12. `avoid`
13. `emotion_temperature_range`
14. `reuse_condition`
15. `confidence`

说明：

1. card 不是 scene 摘要，而是可迁移经验。
2. card 的 `tags` 用于检索和匹配。
3. 旧字段 `fingerprint` 只作为读取兼容，新写出统一使用 `tags`。

## 8. AssociationAnalysis

用途：

```text
关键词或 brief 的导演/编剧前期联想材料。
```

它是独立创意工具，不属于视频解析核心层。

当前字段中使用：

```text
query_tags
```

旧字段 `query_fingerprint` 只作为读取兼容。

## 9. Legacy

旧 `CreativeFingerprint`、`FingerprintEvidence`、`SceneFingerprint`、`FilmFingerprint` 名称仅为兼容旧数据和旧测试语境保留。

新主接口为：

```text
TagProfile
TagEvidence
```
