# 评测输入

这个目录存放从 `retrieval_review` 反向生成的用户输入测试用例。

## Case 类型

- `simple_positive`：简单正例，和复盘层的一句话用途比较接近。
- `hard_positive`：困难正例，不直接复制总结，而是用中文改写脚本阶段、标签和创作目的。
- `hard_negative`：困难负例，和目标有相似的表面视觉标签，但用户真正要的是另一个脚本作用；输入里不会引用另一个目标总结的原文。

## 重新生成

```powershell
python -m mocktesting.eval_input_generator
```

调试时只生成第一个样本：

```powershell
python -m mocktesting.eval_input_generator --limit-fixtures 1 --output src\mocktesting\eval_inputs\debug_inputs.json
```
