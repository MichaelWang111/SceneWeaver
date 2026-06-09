# 观看中检索样本

`retrieval_viewing` 表示第一轮“观看中”分析。这个阶段模型还没有完整视频记忆，只能根据当前关键帧或短 scene 做观察和猜测。

## 在工作流里的作用

这是“观看中”阶段：

- 每一帧或每个 scene 基本独立分析。
- 模型可以观察环境、人物、动作、构图、光线和颜色。
- 模型可以猜测可能含义，但不应该断言最终场景作用。
- 解读中应该保留 `confidence`、`evidence`、`alternatives` 和 `risk_of_overread`。

这一层不应该作为最终检索标签来源。它的用途是保留不确定性，并且用于和后续“全片复盘”结果做对照。

## 数据层

- `source.json`：视觉观察事实、帧路径和 package 信息。
- `knowledge.json`：解释假设、可复用技巧候选、风险和置信度。
- `retrieval.json`：浅层标签、候选脚本用途句子和空 embedding 向量。

## 当前样本

- `tech_recruitment_minimal_uncertain_001`：一个 10 scene 的抽象科技招聘片样本，画面极简且语义不完全明确。

## 命名说明

这个目录替代旧的 `retrieval_prototype_v2` 名称。旧名字容易误导，因为这一层其实发生在全片复盘之前。
