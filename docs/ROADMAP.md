# 路线图

## v1：核心闭环

目标：

```text
真实视频 → scene analysis with tags → experience cards → retrieval
```

当前优先级：

1. 稳定 `SceneAnalysis.tags`。
2. 稳定 `ExperienceCard.tags`。
3. 建立 taxonomy candidate pool。
4. 用真实检索结果校准 tag 权重。

## v1.5：标签治理

目标：

```text
canonical tags + aliases + candidate pool + deprecated mappings
```

任务：

1. 将代码内 taxonomy 迁移到版本化 JSON。
2. 增加 candidate tags 输出和 review 流程。
3. 建立同义词归并和 deprecated mapping。
4. 保证主数据只写 canonical tags。

## v2：生成辅助

在 experience cards 稳定后，再推进：

1. brief 到 cards 的检索增强。
2. 基于 cards 的导演稿生成。
3. 报告型 `FilmAnalysis` 派生产物。
4. 更大样本的经验库整理。
