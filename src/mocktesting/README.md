# MockTesting

这里放检索原型用的 mock 数据和测试工具。它和正式生产输出目录 `outputs/film_analysis` 是分开的，不依赖旧的数据目录范式。

## 检索数据工作流

mock 检索数据分成两个分析阶段：

1. `retrieval_viewing`
   - 表示“观看中”的第一轮分析。
   - 每个关键帧或 scene 基本是独立理解的，没有完整视频记忆。
   - 输出应该以视觉观察为主，并保留不确定性。
   - 导演解读只是猜测，不是最终结论。
   - 这一层用来检查系统到底猜了多少。

2. `retrieval_review`
   - 表示“复盘”的第二轮分析。
   - 模型已经看过完整视频，可以重新理解每个 scene 在全片中的作用。
   - 这一层更接近可复用的导演经验。
   - 默认应该用这一层生成检索标签。
   - 这一层也可以作为验证目标，用来比较“观看中猜测”和“全片复盘结论”的差异。

## 设计原则

不要把第一轮视觉猜测直接当成最终检索知识。先用 `retrieval_viewing` 保留不确定性，再用 `retrieval_review` 生成面向检索的标签、脚本用途描述和经验候选。

## 检索测试输入生成

`eval_input_generator.py` 会根据 `retrieval_review` 的内容，反向生成用户输入测试用例：

- `simple_positive`：简单正例，和复盘层的一句话用途比较接近。
- `hard_positive`：困难正例，不直接复制总结，而是根据阶段、中文创作目的、标签、行业语境改写。
- `hard_negative`：困难负例，保留相似视觉元素，但明确要求另一个 scene 或脚本阶段；不会把另一个 scene 的总结原文塞进输入。

重新生成全部测试输入：

```powershell
python -m mocktesting.eval_input_generator
```

当前生成文件：

```text
src/mocktesting/eval_inputs/review_generated_inputs.json
```

当前数据集包含 450 条 case：

- 150 条 `simple_positive`
- 150 条 `hard_positive`
- 150 条 `hard_negative`

## 评判方法

`eval_methods.py` 里有三种原型评分方法：

- `summary-embedding`：把 `user_input` 和复盘层的一句话用途 `script_use_sentence` 做 embedding 余弦相似度。
- `tags-embedding`：把 `user_input` 和复盘层展开后的标签文本做 embedding 余弦相似度。
- `llm-judge-lite`：快速 LLM 评价，只判断是否匹配和分数。
- `llm-judge-batch`：批量 LLM 快速评价，一次请求判断多条 case。
- `llm-judge`：LLM 点评模式，判断是否符合目标 scene，并给出简短理由。

embedding 评估使用 DashScope text embedding v4，思路和根目录的 `test.py` 一致，但这里从环境变量读取 API Key，不再硬编码。

示例：

```powershell
python -m mocktesting.eval_methods --method summary-embedding --limit 10
python -m mocktesting.eval_methods --method tags-embedding --limit 10
python -m mocktesting.eval_methods --method llm-judge-lite --limit 10
python -m mocktesting.eval_methods --method llm-judge-batch --limit 60 --batch-size 10 --llm-timeout-seconds 90
python -m mocktesting.eval_methods --method llm-judge --limit 10
python -m mocktesting.eval_methods --method all --limit 10
python -m mocktesting.eval_methods --method llm-judge-batch --limit 60 --batch-size 10 --concurrency 6 --llm-timeout-seconds 90
```

LLM 响应慢时可以加大单条请求超时：

```powershell
python -m mocktesting.eval_methods --method llm-judge-lite --limit 10 --llm-timeout-seconds 90
```

如果要跑几十条以上，优先试 batch + 并发：

```powershell
python -m mocktesting.eval_methods --method llm-judge-batch --limit 60 --batch-size 10 --concurrency 3 --llm-timeout-seconds 90
```

或者直接并发跑 lite：

```powershell
python -m mocktesting.eval_methods --method llm-judge-lite --limit 60 --concurrency 6 --llm-timeout-seconds 90
```

默认输出文件：

- `src/mocktesting/eval_outputs/summary_scores.json`
- `src/mocktesting/eval_outputs/tag_scores.json`
- `src/mocktesting/eval_outputs/llm_lite_scores.json`
- `src/mocktesting/eval_outputs/llm_batch_scores.json`
- `src/mocktesting/eval_outputs/llm_scores.json`
- `src/mocktesting/eval_outputs/all_scores.json`

embedding 所需环境变量：

```powershell
$env:DASHSCOPE_API_KEY="sk-..."
```

LLM judge 复用 SceneWeaver 之前的 LLM 配置：

```powershell
$env:VIDEO_ANALYZER_API_KEY="sk-..."
$env:VIDEO_ANALYZER_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:VIDEO_ANALYZER_MODEL="qwen3.6-plus"
```

## 多文本 Embedding 检索原型

`mock_retriever.py` 用 `retrieval_review` 构建多文本通道 embedding index。每个 retrieval item 默认包含四个 channel：

- `script_use`：脚本用途。
- `experience`：可复用导演经验。
- `visual_tags`：画面和浅层标签。
- `combined`：组合兜底文本。

先查看需要生成多少 embedding：

```powershell
python -m mocktesting.mock_retriever build-index --dry-run
```

构建 index：

```powershell
python -m mocktesting.mock_retriever build-index --embedding-batch-size 10
```

搜索：

```powershell
python -m mocktesting.mock_retriever search "真实但有力量的开场，科技向善，不要大厂味" --top-k 10
```

评测：

```powershell
python -m mocktesting.mock_retriever evaluate --limit 450 --top-k 10
```

`--embedding-batch-size` 控制每次向 DashScope embedding API 提交多少条文本。百炼当前限制单批不能超过 10，因此默认是 10，传更大的值也会被缓存层自动压到 10。

默认输出：

- `src/mocktesting/eval_outputs/mock_embedding_index.json`
- `src/mocktesting/eval_outputs/mock_search_result.json`
- `src/mocktesting/eval_outputs/mock_retrieval_report.json`

embedding 缓存默认写入：

```text
src/mocktesting/embedding_cache/qwen_text_embedding_v4_1024.jsonl
```

## 约束层与动态调参

`mock_retriever.py` 现在默认启用轻量约束层。它会从用户输入中解析：

- `desired_stage`：用户真正想要的脚本阶段，例如“我真正要的是铺垫”。
- `forbidden_stage`：用户明确不要的脚本阶段，例如“不要做成开场”“避免技术展示”。
- `negative_constraints`：暂时无法结构化的负面要求，例如“不要互联网大厂味”。
- `visual_hints`：用户允许借用的画面元素。

默认 profile：

```text
src/mocktesting/eval_outputs/mock_constraint_profile.json
```

约束前后对照：

```powershell
python -m mocktesting.mock_retriever evaluate --limit 60 --top-k 10 --no-constraints --output src\mocktesting\eval_outputs\mock_retrieval_report_no_constraints.json
python -m mocktesting.mock_retriever evaluate --limit 60 --top-k 10
```

动态调参：

```powershell
python -m mocktesting.mock_retriever tune-constraints --limit 60 --top-k 10
```

调参会写出：

```text
src/mocktesting/eval_outputs/mock_constraint_tuning_report.json
src/mocktesting/eval_outputs/mock_constraint_profile.json
```

完整评测：

```powershell
python -m mocktesting.mock_retriever evaluate --limit 450 --top-k 10
```

如果想回到纯 embedding 排序，所有 search / evaluate 命令都可以加：

```powershell
--no-constraints
```
