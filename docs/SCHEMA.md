# 数据结构说明

## 1. 设计原则

SceneWeaver 的 schema 需要区分 3 类信息：

1. 客观观察：画面中看到了什么。
2. 导演解释：为什么这样拍。
3. 可复用经验：未来如何被检索和生成系统复用。

不要把所有字段塞进一个 `director_insight`。

## 2. scene_package

`scene_package` 是送入 Vision LLM 的最小输入单元。

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

## 3. scene_analysis

`scene_analysis` 是单个 scene 的结构化导演分析。

```json
{
  "scene_id": "scene_001",
  "time_range": {
    "start": "00:00:03.200",
    "end": "00:00:07.800"
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

## 4. scenes.json

`scenes.json` 是一支视频所有 scene analysis 的集合。

```json
{
  "video_id": "bilibili_BVxxxx",
  "source_url": "https://www.bilibili.com/video/...",
  "scene_count": 12,
  "scenes": []
}
```

## 5. film_analysis

`film_analysis` 是全片层面的导演语言总结。

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
    },
    {
      "phase": "middle",
      "emotion": "归属",
      "function": "建立团队关系"
    },
    {
      "phase": "ending",
      "emotion": "未来感",
      "function": "完成品牌召唤"
    }
  ],
  "visual_language": {
    "camera": ["手持", "中近景", "群像剪辑"],
    "lighting": ["自然光", "晨光", "低饱和"],
    "symbolism": ["团队协作", "工作空间", "城市生活"]
  },
  "narrative_structure": "真实日常 → 团队协作 → 未来召唤",
  "brand_personality": ["真实", "年轻", "可信", "温暖"],
  "audience_projection": "年轻人可以在这里成为被需要、有成长空间的团队成员",
  "director_language_summary": "影片通过生活化工作细节和群像推进，避免空泛口号，用真实感建立雇主品牌信任"
}
```

## 6. experience_card

`experience_card` 是未来检索和生成的核心知识单元。

```json
{
  "card_id": "exp_000001",
  "source_video_id": "bilibili_BVxxxx",
  "source_scene_ids": ["scene_001", "scene_002"],
  "keywords": ["青春", "热情", "梦想", "团队"],
  "underlying_emotion": "年轻人正在共同创造未来",
  "narrative_logic": "个体日常逐渐汇入团队群像",
  "director_strategy": "先建立真实工作状态，再通过群像剪辑释放集体能量",
  "shooting_techniques": ["手持跟拍", "逆光", "中近景", "群像快剪"],
  "visual_symbols": ["晨光", "团队协作", "奔跑", "工作空间"],
  "copywriting_tone": "少口号，多第一人称和真实动作",
  "avoid": ["空泛梦想口号", "过度互联网大厂感", "纯炫技快剪"],
  "emotion_temperature_range": [0.55, 0.85],
  "reuse_condition": "适合希望表达青年热情、团队归属和未来感的招聘宣传片",
  "confidence": 0.82
}
```

## 7. 命名规范

1. 字段名使用英文 snake_case。
2. 文档说明使用中文。
3. 输出内容可以中英混合，但 v1 默认中文。
4. 时间统一使用 `HH:MM:SS.mmm`。
5. scene id 使用 `scene_001` 格式。
6. experience id 使用 `exp_000001` 格式。

## 8. 必须避免

1. 不要把客观观察写成导演判断。
2. 不要把推断当成确定事实。
3. 不要把单个 scene 的判断直接上升为全片结论。
4. 不要把原始分析报告直接存成经验库。
5. 不要让 LLM 输出未验证 JSON 进入存储层。
