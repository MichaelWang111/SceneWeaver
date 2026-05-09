from __future__ import annotations

from pathlib import Path

from sceneweaver.analysis.fingerprint import build_film_fingerprint, build_scene_fingerprint
from sceneweaver.schemas import (
    ExperienceCard,
    FilmAnalysis,
    FrameSet,
    SceneAnalysis,
    SceneMetadata,
    ScenePackage,
    ScenesAnalysis,
    SubtitleSegment,
    TimeRange,
)
from sceneweaver.schemas.film_analysis import EmotionalCurvePoint, Rhythm, VisualLanguage
from sceneweaver.schemas.scene_analysis import (
    DirectorInterpretation,
    ExperienceCandidate,
    VisualObservation,
)
from sceneweaver.storage.json_store import write_json, write_jsonl


def build_mock_artifacts(source_url: str = "https://www.bilibili.com/video/BVxxxx") -> tuple[
    ScenePackage,
    SceneAnalysis,
    ScenesAnalysis,
    FilmAnalysis,
    list[ExperienceCard],
]:
    video_id = "bilibili_BVxxxx"
    time_range = TimeRange(start="00:00:03.200", end="00:00:07.800", duration_seconds=4.6)

    package = ScenePackage(
        scene_id="scene_001",
        source_video_id=video_id,
        time_range=time_range,
        frames=FrameSet(
            start="frames/scene_001_start.jpg",
            middle="frames/scene_001_middle.jpg",
            end="frames/scene_001_end.jpg",
        ),
        subtitle_segment=SubtitleSegment(text="这里是字幕片段", items=[]),
        metadata=SceneMetadata(scene_index=1, source_url=source_url, language="zh-CN"),
    )

    scene_analysis = SceneAnalysis(
        scene_id="scene_001",
        time_range=time_range,
        visual_observation=VisualObservation(
            setting="办公室或工位环境",
            characters="年轻员工或团队成员",
            action_change="从独立工作转向团队互动",
            composition="中近景为主，强调人物状态",
            lighting="自然光或柔和人工光",
            color="低饱和、真实感",
            camera_motion="无法从三帧确定，可能为静态或轻微移动",
            confidence_notes="三帧不足以确认完整镜头运动",
        ),
        director_interpretation=DirectorInterpretation(
            narrative_function="建立真实工作状态",
            emotional_function="降低广告感，建立可信度",
            brand_personality_signal="真实、温暖、可靠",
            underlying_emotion="我可以在这里被看见并参与真实工作",
            audience_projection="年轻人可以成为团队中被需要的一员",
            shooting_techniques=["中近景", "自然光", "生活化动作"],
            why_it_works="通过真实工作细节替代口号，使观众更容易相信品牌表达",
        ),
        experience_candidates=[
            ExperienceCandidate(
                keywords=["真实感", "青年", "团队"],
                emotion="被需要",
                narrative_logic="先建立真实日常，再导向团队归属",
                techniques=["自然光", "中近景", "微行为"],
                reuse_condition="适合需要弱化广告感、强调真实工作氛围的招聘宣传片",
            )
        ],
        emotion_temperature=0.45,
    )

    scenes = ScenesAnalysis(
        video_id=video_id,
        source_url=source_url,
        scene_count=1,
        scenes=[scene_analysis],
    )
    scene_fingerprint = build_scene_fingerprint(scene_analysis, source_video_id=video_id)
    film_fingerprint = build_film_fingerprint(
        video_id=video_id,
        scene_fingerprints=[scene_fingerprint],
    )

    film_analysis = FilmAnalysis(
        video_id=video_id,
        fingerprint=film_fingerprint.fingerprint,
        atmosphere="真实、温暖、轻度热血",
        tone="纪录片式招聘宣传片",
        rhythm=Rhythm(overall="前慢后快", description="前半段建立真实感，后半段释放团队能量"),
        emotional_curve=[
            EmotionalCurvePoint(phase="start", emotion="好奇", function="引入人物和场景"),
            EmotionalCurvePoint(phase="middle", emotion="归属", function="建立团队关系"),
            EmotionalCurvePoint(phase="ending", emotion="未来感", function="完成品牌召唤"),
        ],
        visual_language=VisualLanguage(
            camera=["手持", "中近景", "群像剪辑"],
            lighting=["自然光", "晨光", "低饱和"],
            symbolism=["团队协作", "工作空间", "城市生活"],
        ),
        narrative_structure="真实日常 -> 团队协作 -> 未来召唤",
        brand_personality=["真实", "年轻", "可信", "温暖"],
        audience_projection="年轻人可以在这里成为被需要、有成长空间的团队成员",
        director_language_summary="影片通过生活化工作细节和群像推进，避免空泛口号，用真实感建立雇主品牌信任",
    )

    cards = [
        ExperienceCard(
            card_id="exp_000001",
            source_video_id=video_id,
            source_scene_ids=["scene_001"],
            fingerprint=scene_fingerprint.fingerprint,
            keywords=["青春", "热情", "梦想", "团队"],
            underlying_emotion="年轻人正在共同创造未来",
            narrative_logic="个体日常逐渐汇入团队群像",
            director_strategy="先建立真实工作状态，再通过群像剪辑释放集体能量",
            shooting_techniques=["手持跟拍", "逆光", "中近景", "群像快剪"],
            visual_symbols=["晨光", "团队协作", "奔跑", "工作空间"],
            copywriting_tone="少口号，多第一人称和真实动作",
            avoid=["空泛梦想口号", "过度互联网大厂感", "纯炫技快剪"],
            emotion_temperature_range=(0.55, 0.85),
            reuse_condition="适合希望表达青年热情、团队归属和未来感的招聘宣传片",
            confidence=0.82,
        )
    ]
    return package, scene_analysis, scenes, film_analysis, cards


def run_mock_pipeline(output_dir: Path, source_url: str = "https://www.bilibili.com/video/BVxxxx") -> Path:
    package, scene_analysis, scenes, film_analysis, cards = build_mock_artifacts(source_url=source_url)
    scene_fingerprint = build_scene_fingerprint(scene_analysis, source_video_id=scenes.video_id)
    film_fingerprint = build_film_fingerprint(
        video_id=scenes.video_id,
        scene_fingerprints=[scene_fingerprint],
    )

    write_json(output_dir / "packages" / "scene_001.json", package)
    write_json(output_dir / "analysis" / "scene_001.json", scene_analysis)
    write_json(output_dir / "analysis" / "scenes.json", scenes)
    write_json(output_dir / "analysis" / "film_analysis.json", film_analysis)
    write_jsonl(output_dir / "analysis" / "experience_cards.jsonl", cards)
    write_json(output_dir / "fingerprints" / "scene_001.json", scene_fingerprint)
    write_json(output_dir / "fingerprints" / "film_fingerprint.json", film_fingerprint)
    return output_dir
