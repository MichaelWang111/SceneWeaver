# 产品需求文档 PRD

## 1. 产品名称

SceneWeaver

## 2. 一句话说明

SceneWeaver 是一个从视频案例中提炼导演经验，并为未来导演稿生成提供知识基础的分析系统。

## 3. 目标用户

### 3.1 第一阶段用户

1. 项目创建者。
2. AI 创意工具开发者。
3. 对商业视频、招聘宣传片、品牌片进行研究的人。

### 3.2 未来用户

1. 广告创意策划。
2. 商业导演。
3. 制片团队。
4. 雇主品牌团队。
5. 短视频和品牌内容团队。

## 4. 核心场景

### 4.1 视频经验沉淀

用户输入一个 Bilibili 视频 URL，系统自动下载并拆分视频，调用 LLM 分析每个 scene 的画面、字幕、导演意图和可复用经验。

### 4.2 全片导演语言总结

系统按时间顺序读取全部 scene analysis，生成全片层面的导演语言总结。

需要包含：

1. 氛围 atmosphere
2. 基调 tone
3. 节奏 rhythm
4. 情绪曲线 emotional curve
5. 视觉风格 visual language
6. 叙事结构 narrative structure
7. 导演语言 director language
8. 品牌人格 brand personality
9. 观众身份投射 audience projection

### 4.3 经验卡片抽取

系统将 scene analysis 和 film analysis 抽取为可复用的经验卡片。

经验卡片面向未来检索和生成，不是面向阅读报告。

## 5. v1 功能范围

### 5.1 输入层

1. 支持 Bilibili URL。
2. 使用 `yt-dlp` 下载视频。
3. 尝试获取字幕。
4. 保留音频提取扩展点。

### 5.2 拆分层

1. 使用 PySceneDetect 进行 scene 切分。
2. 记录每个 scene 的 `time_range`。
3. 使用 ffmpeg 抽取 start / middle / end 3 帧。
4. 按 scene 时间范围匹配字幕片段。

### 5.3 Scene 解析层

1. 每个 scene package 独立调用 Vision LLM。
2. 支持并发。
3. 支持 retry。
4. 支持 cache。
5. 支持 JSON validation。

### 5.4 全片解析层

1. 输入 `scenes.json`。
2. 按时间顺序分析全片结构。
3. 输出 `film_analysis.json`。

### 5.5 经验存储层

1. 从 scene 和 film 分析中抽取 experience cards。
2. 以 JSONL 存储。
3. 为未来 embedding 和 vector db 预留接口。

## 6. 非目标

v1 不做：

1. Web UI。
2. 多平台下载。
3. 智能搜索视频。
4. 视频生成。
5. LoRA / fine-tune。
6. 自动剪辑。
7. 完整故事板生成。
8. 生产级数据库。

## 7. 成功标准

### 7.1 功能成功

1. 一个 Bilibili URL 可以跑出完整分析链路。
2. 每个 scene 都有结构化分析结果。
3. 全片总结能体现整体导演语言。
4. experience cards 能表达可复用的导演经验。

### 7.2 质量成功

1. 输出不只是画面描述，而能解释“为什么这样拍”。
2. 输出能区分客观观察和导演推断。
3. 输出能体现情绪、叙事、品牌人格和身份投射。
4. 经验卡片能被关键词检索复用。

### 7.3 长期成功

未来输入关键词：

```text
青春 / 热情 / 梦想
```

系统应能召回：

1. 底层情感。
2. 叙事逻辑。
3. 拍摄技法。
4. 视觉符号。
5. 文案语气。
6. 可执行导演稿。

## 8. 核心判断

SceneWeaver 的产品价值不在于“分析视频”，而在于：

```text
把商业导演经验变成可检索、可组合、可生成的知识。
```
