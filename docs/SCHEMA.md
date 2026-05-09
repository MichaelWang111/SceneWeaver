# 数据结构

本文说明核心 JSON / JSONL 数据合同。代码实现以 `src/sceneweaver/schemas/` 为准。

## 1. 设计原则

SceneWeaver 的 schema 区分三类信息：

1. 客观观察：画面中看到了什么。
2. 导演解释：为什么这样拍。
3. 可复用经验：未来如何被检索和生成系统复用。

禁止把所有内容塞进一个泛化的 `director_insight` 字段。

## 2. ScenePackage

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

示例：

```json
{
  "scene_id": "scene_001",
  "source_video_id": "bilibili_BVxxxx",
  "time_range": {
    "start": "00:00:03.200",
    "end": "00:00:07.800",
    "duration_seconds": 4.6
  },
  "frames": {
    "start": "frames/scene_001_start.jpg",
    "middle": "frames/scene_001_middle.jpg",
    "end": "frames/scene_001_end.jpg"
  },
  "subtitle_segment": {
    "text": "这里是字幕片段",
    "items": []
  },
  "metadata": {
    "scene_index": 1,
    "source_url": "https://www.bilibili.com/video/...",
    "language": "zh-CN"
  }
}
```

## 3. SceneAnalysis

用途：

```text
单个 scene 的结构化导演分析。
```

关键字段：

1. `visual_observation`：只写客观观察。
2. `director_interpretation`：写导演意图、叙事功能、情绪功能、品牌人格和观众投射。
3. `experience_candidates`：候选可复用经验，不等于最终经验卡片。
4. `emotion_temperature`：0 到 1 的情绪强度。

示例：

```json
{
  "scene_id": "scene_001",
  "time_range": {
    "start": "00:00:03.200",
    "end": "00:00:07.800",
    "duration_seconds": 4.6
  },
  "visual_observation": {
    "setting": "办公室或工位环境",
    "characters": "年轻员工或团队成员",
    "action_change": "从独立工作转向团队互动",
    "composition": "中近景为主，强调人物状态",
    "lighting": "自然光或柔和人工光",
    "color": "低饱和、真实感",
    "camera_motion": "无法从三帧确定，可能为静态或轻微移动",
    "confidence_notes": "三帧不足以确认完整镜头运动"
  },
  "director_interpretation": {
    "narrative_function": "建立真实工作状态",
    "emotional_function": "降低广告感，建立可信度",
    "brand_personality_signal": "真实、温暖、可靠",
    "underlying_emotion": "我可以在这里被看见并参与真实工作",
    "audience_projection": "年轻人可以成为团队中被需要的一员",
    "shooting_techniques": ["中近景", "自然光", "生活化动作"],
    "why_it_works": "通过真实工作细节替代口号，使观众更容易相信品牌表达"
  },
  "experience_candidates": [
    {
      "keywords": ["真实感", "青年", "团队"],
      "emotion": "被需要",
      "narrative_logic": "先建立真实日常，再导向团队归属",
      "techniques": ["自然光", "中近景", "微行为"],
      "reuse_condition": "适合需要弱化广告感、强调真实工作氛围的招聘宣传片"
    }
  ],
  "emotion_temperature": 0.45
}
```

## 4. ScenesAnalysis

用途：

```text
一支视频所有 scene analysis 的集合。
```

示例：

```json
{
  "video_id": "bilibili_BVxxxx",
  "source_url": "https://www.bilibili.com/video/...",
  "scene_count": 12,
  "scenes": []
}
```

校验：

1. `scene_count` 必须等于 `scenes` 长度。
2. `scene_id` 使用 `scene_001` 格式。

## 5. FilmAnalysis

用途：

```text
全片层面的导演语言总结。
```

关键字段：

1. `atmosphere`
2. `tone`
3. `rhythm`
4. `emotional_curve`
5. `visual_language`
6. `narrative_structure`
7. `brand_personality`
8. `audience_projection`
9. `director_language_summary`

示例：

```json
{
  "video_id": "bilibili_BVxxxx",
  "atmosphere": "真实、温暖、轻度热血",
  "tone": "纪录片式招聘宣传片",
  "rhythm": {
    "overall": "前慢后快",
    "description": "前半段建立真实感，后半段释放团队能量"
  },
  "emotional_curve": [
    {
      "phase": "start",
      "emotion": "好奇",
      "function": "引入人物和场景"
    }
  ],
  "visual_language": {
    "camera": ["手持", "中近景"],
    "lighting": ["自然光", "晨光"],
    "symbolism": ["团队协作", "工作空间"]
  },
  "narrative_structure": "真实日常 -> 团队协作 -> 未来召唤",
  "brand_personality": ["真实", "年轻", "可信"],
  "audience_projection": "年轻人可以在这里成为被需要、有成长空间的团队成员",
  "director_language_summary": "影片通过生活化工作细节和群像推进建立雇主品牌信任"
}
```

## 6. ExperienceCard

用途：

```text
未来检索和生成的核心知识单元。
```

示例：

```json
{
  "card_id": "exp_000001",
  "source_video_id": "bilibili_BVxxxx",
  "source_scene_ids": ["scene_001", "scene_002"],
  "keywords": ["青春", "热情", "梦想", "团队"],
  "underlying_emotion": "年轻人正在共同创造未来",
  "narrative_logic": "个体日常逐渐汇入团队群像",
  "director_strategy": "先建立真实工作状态，再通过群像剪辑释放集体能量",
  "shooting_techniques": ["手持跟拍", "逆光", "中近景"],
  "visual_symbols": ["晨光", "团队协作", "工作空间"],
  "copywriting_tone": "少口号，多第一人称和真实动作",
  "avoid": ["空泛梦想口号", "过度互联网大厂感"],
  "emotion_temperature_range": [0.55, 0.85],
  "reuse_condition": "适合希望表达青年热情、团队归属和未来感的招聘宣传片",
  "confidence": 0.82
}
```

校验：

1. `card_id` 使用 `exp_000001` 格式。
2. `source_scene_ids` 使用 `scene_001` 格式。
3. `emotion_temperature_range` 必须在 0..1 内且从低到高。
4. `confidence` 必须在 0..1 内。

## 7. AssociationAnalysis

用途：

```text
关键词或 brief 的导演/编剧前期联想材料。
```

该结构由 `associate` 命令生成，当前不依赖历史视频经验库。

关键字段：

1. `input_text`
2. `core_reading`
3. `emotional_arc`
4. `association_map`
5. `director_possibilities`
6. `avoid_cliches`

校验：

1. `association_count` 必须等于所有 association item 总数。
2. 每个 association 类别至少 1 条。
3. `director_possibilities` 必须有 3 到 5 个方向。

## 8. 命名规范

1. 字段名使用英文 snake_case。
2. 文档说明使用中文。
3. 输出内容默认中文。
4. 时间统一使用 `HH:MM:SS.mmm`。
5. scene id 使用 `scene_001` 格式。
6. experience id 使用 `exp_000001` 格式。

## 9. 必须避免

1. 不要把客观观察写成导演判断。
2. 不要把推断当成确定事实。
3. 不要把单个 scene 的判断直接上升为全片结论。
4. 不要把原始分析报告直接存成经验库。
5. 不要让未验证 JSON 进入存储层。
