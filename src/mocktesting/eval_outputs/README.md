# 评测输出

这个目录用于存放本地评测分数输出。生成出来的 score JSON 不是源数据样本，主要用于临时查看和比较。

默认输出文件按评判方法拆开：

- `summary_scores.json`：一句话用途 embedding 分数。
- `tag_scores.json`：标签 embedding 分数。
- `llm_lite_scores.json`：LLM 快速评价分数，只判断是否匹配和分数。
- `llm_batch_scores.json`：LLM 批量快速评价分数，一次请求判断多条 case。
- `llm_scores.json`：LLM 判断分数。
- `all_scores.json`：三种方法一起跑时的综合结果。

## 评判方法

一句话用途 embedding：

```powershell
python -m mocktesting.eval_methods --method summary-embedding --limit 10
```

默认输出：

```text
src/mocktesting/eval_outputs/summary_scores.json
```

标签 embedding：

```powershell
python -m mocktesting.eval_methods --method tags-embedding --limit 10
```

默认输出：

```text
src/mocktesting/eval_outputs/tag_scores.json
```

LLM 快速评价：

```powershell
python -m mocktesting.eval_methods --method llm-judge-lite --limit 10
```

如果模型响应慢，可以加大单条请求超时：

```powershell
python -m mocktesting.eval_methods --method llm-judge-lite --limit 10 --llm-timeout-seconds 90
```

默认输出：

```text
src/mocktesting/eval_outputs/llm_lite_scores.json
```

LLM 批量快速评价：

```powershell
python -m mocktesting.eval_methods --method llm-judge-batch --limit 60 --batch-size 10 --llm-timeout-seconds 90
```

如果 batch 仍然不快，说明模型处理长 prompt 的时间抵消了减少请求次数的收益。可以加并发：

```powershell
python -m mocktesting.eval_methods --method llm-judge-batch --limit 60 --batch-size 10 --concurrency 3 --llm-timeout-seconds 90
```

也可以并发跑逐条 lite：

```powershell
python -m mocktesting.eval_methods --method llm-judge-lite --limit 60 --concurrency 6 --llm-timeout-seconds 90
```

默认输出：

```text
src/mocktesting/eval_outputs/llm_batch_scores.json
```

`--batch-size` 控制一次请求评几条 case。建议先用 10，稳定后再试 20。`--concurrency` 控制同时发几个 LLM 请求，建议从 3 开始，太高可能触发限流或超时。

LLM 点评：

```powershell
python -m mocktesting.eval_methods --method llm-judge --limit 10
```

默认输出：

```text
src/mocktesting/eval_outputs/llm_scores.json
```

`llm-judge-lite` 逐条请求，只输出 `fits` 和 `score`。`llm-judge-batch` 一次请求判断多条 case，适合跑几十到几百条。`llm-judge` 会额外输出 `reason`，更适合人工检查为什么这么判。

LLM 请求失败时，评测不会整批中断，而是把 `error` 写进对应 case，方便继续看其他结果。

全部方法一起跑：

```powershell
python -m mocktesting.eval_methods --method all --limit 10
```

默认输出：

```text
src/mocktesting/eval_outputs/all_scores.json
```

也可以用 `--output` 手动指定输出路径。
