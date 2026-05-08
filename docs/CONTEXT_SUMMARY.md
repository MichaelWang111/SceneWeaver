# 背景摘要

## 1. 项目一句话

SceneWeaver 是一个从商业视频中提炼导演经验，并为未来导演稿生成提供知识基础的 Python 项目。

## 2. 当前核心目标

当前阶段不做完整视频生成，也不做复杂 Web 产品。

第一目标是跑通：

```text
Bilibili 视频
→ 场景拆分
→ 三帧 + 字幕片段
→ Scene LLM 解析
→ 全片 LLM 总结
→ 导演经验卡片
```

## 3. 项目本质

项目不是普通的视频分析器。

真正目标是建立：

```text
导演经验检索系统
```

也就是把视频里的导演决策结构化，让未来可以根据关键词检索并生成导演稿。

## 4. 关键认知

导演不是在“拍镜头”，而是在控制观众感受。

所以系统不应只记录：

```text
镜头类型 / 构图 / 光线 / 色彩
```

更应记录：

```text
这个 scene 让观众感受到什么
这个 scene 在全片中承担什么职责
它塑造了什么品牌人格
它制造了什么身份投射
这条经验未来如何复用
```

## 5. 当前 v1 数据链路

```text
Input Layer
→ Split Layer
→ Scene Packaging
→ Scene LLM Analysis
→ Full Film Analysis
→ Experience Card Extraction
→ Local JSON / JSONL Storage
```

## 6. 核心数据产物

1. `scene_package`：送入 Vision LLM 的最小输入单元。
2. `scene_analysis`：单个 scene 的导演分析。
3. `scenes.json`：全片所有 scene 分析集合。
4. `film_analysis.json`：全片导演语言总结。
5. `experience_cards.jsonl`：可复用导演经验卡片。

## 7. 第一版技术栈

1. Python 3.11+
2. Pydantic
3. Typer
4. yt-dlp
5. PySceneDetect
6. ffmpeg / ffprobe
7. asyncio + httpx
8. OpenAI-compatible Vision LLM API
9. JSON / JSONL
10. pytest

## 8. 参考项目

本地参考项目：

```text
D:\WorkSpace\github\video-expert-analyzer
```

它适合参考：

1. 视频下载。
2. scene 切分。
3. frame extraction。
4. 字幕 / OCR / ASR 处理。
5. JSON 和 Markdown 报告输出。

但它本质是高级剪辑分析器，SceneWeaver 需要在它之上增加导演认知层。

## 9. 当前边界

v1 不做：

1. Web UI。
2. 多平台输入。
3. 视频生成。
4. LoRA / fine-tune。
5. 完整 Graph RAG。
6. 多 Agent 自动化。

## 10. 下一步重点

下一步应从文档进入 Python 工程骨架：

```text
pyproject.toml
src/sceneweaver/
prompts/
examples/
tests/
```

然后优先实现 schema 和本地 mock pipeline，再接真实视频处理和真实 LLM。
