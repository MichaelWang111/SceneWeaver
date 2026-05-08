# 参考项目笔记

## 1. 本地参考项目

路径：

```text
D:\WorkSpace\github\video-expert-analyzer
```

该项目是 SceneWeaver v1 的重要参考，但不是直接复制对象。

## 2. 可借鉴能力

### 2.1 视频输入

参考项目已经覆盖：

1. yt-dlp 下载。
2. Bilibili / YouTube / Douyin / Xiaohongshu 等平台入口。
3. 视频 metadata 获取。

SceneWeaver v1 只需要先保留 Bilibili。

### 2.2 视频拆分

参考项目使用：

1. PySceneDetect。
2. ffmpeg。
3. scene clips。
4. preview frames。

SceneWeaver 可以借鉴其工程流程，但输出目标不同。

### 2.3 字幕和音频

参考项目包含：

1. Bilibili 字幕。
2. OCR。
3. ASR。
4. 多级 fallback。

SceneWeaver v1 可以先做最小字幕链路，复杂 fallback 后置。

### 2.4 报告生成

参考项目会生成 Markdown 报告和 JSON 分析文件。

SceneWeaver 也应保留：

1. JSON 中间产物。
2. Markdown 可读报告。
3. 独立输出目录。

## 3. 不应照搬的部分

参考项目本质更接近：

```text
高级剪辑分析器
```

它主要关注：

1. aesthetic beauty
2. credibility
3. impact
4. memorability
5. fun / interest
6. best shot selection

这些字段有价值，但不是 SceneWeaver 的核心壁垒。

SceneWeaver 要新增的是：

1. scene function
2. emotional function
3. underlying emotion
4. brand personality signal
5. audience projection
6. narrative logic
7. director strategy
8. reusable experience card

## 4. 架构差异

### 4.1 参考项目

```text
video
→ scenes
→ frames
→ visual scoring
→ best shots
→ report
```

### 4.2 SceneWeaver

```text
video
→ scenes
→ scene packages
→ director interpretation
→ full film analysis
→ experience cards
→ future retrieval / generation
```

## 5. 核心差异

参考项目回答：

```text
这个镜头好不好？
```

SceneWeaver 要回答：

```text
为什么这里要这样拍？
这个 scene 在全片中承担什么职责？
它制造了什么观众感受？
它塑造了什么身份投射？
这条经验未来如何复用？
```

## 6. 使用建议

1. 借鉴参考项目的视频下载和拆分思路。
2. 借鉴它的 JSON / Markdown 输出方式。
3. 不要沿用它的评分体系作为主 schema。
4. 不要把 best shot 作为核心目标。
5. 将重点放在导演语义层和经验卡片层。

## 7. 当前注意事项

参考项目本地存在未提交修改：

```text
scripts/pipeline_enhanced.py
```

后续如果需要复制或参考代码，应先确认该修改是否需要保留。
