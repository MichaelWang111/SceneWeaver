from __future__ import annotations

import shutil
from pathlib import Path

from sceneweaver.analysis.scene_analyzer import analyze_scene_packages
from sceneweaver.pipeline.mock_pipeline import build_mock_artifacts
from sceneweaver.schemas import SceneAnalysis, ScenesAnalysis
from sceneweaver.storage.json_store import read_json, write_json


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls = []

    def analyze_images_json(self, *, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> dict:
        self.calls.append((system_prompt, user_prompt, image_paths))
        return {
            "scene_id": "scene_001",
            "time_range": {
                "start": "00:00:03.200",
                "end": "00:00:07.800",
                "duration_seconds": 4.6,
            },
            "visual_observation": {
                "setting": "办公室或工位环境",
                "characters": "年轻员工或团队成员",
                "action_change": "从独立工作转向团队互动",
                "composition": "中近景为主",
                "lighting": "柔和人工光",
                "color": "低饱和、真实感",
                "camera_motion": "三帧不足以确认完整镜头运动",
                "confidence_notes": "画面主体和环境较明确，运镜判断不确定",
            },
            "director_interpretation": {
                "narrative_function": "建立真实工作状态",
                "emotional_function": "降低广告感，建立可信度",
                "brand_personality_signal": "真实、温暖、可靠",
                "underlying_emotion": "我可以在这里参与真实工作",
                "audience_projection": "年轻人可以成为团队中被需要的一员",
                "shooting_techniques": ["中近景", "自然光", "生活化动作"],
                "why_it_works": "用真实细节替代口号，使观众更容易相信表达",
            },
            "experience_candidates": [
                {
                    "keywords": ["真实感", "青年", "团队"],
                    "emotion": "被需要",
                    "narrative_logic": "先建立真实日常，再导向团队归属",
                    "techniques": ["自然光", "中近景"],
                    "reuse_condition": "适合招聘宣传片",
                }
            ],
            "emotion_temperature": 0.45,
        }


def test_analyze_scene_packages_writes_valid_outputs(tmp_path):
    package, *_ = build_mock_artifacts()
    output_dir = tmp_path / "video"
    write_json(output_dir / "packages" / "scene_001.json", package)
    for frame in [package.frames.start, package.frames.middle, package.frames.end]:
        frame_path = output_dir / frame
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"fake image")

    client = FakeVisionClient()
    scenes = analyze_scene_packages(output_dir, client=client)

    assert scenes.scene_count == 1
    assert len(client.calls) == 1
    assert len(client.calls[0][2]) == 3
    assert read_json(output_dir / "analysis" / "scene_001.json", SceneAnalysis)
    assert read_json(output_dir / "analysis" / "scenes.json", ScenesAnalysis)


def test_analyze_scene_packages_reuses_existing_analysis(tmp_path):
    package, scene_analysis, *_ = build_mock_artifacts()
    output_dir = tmp_path / "video"
    write_json(output_dir / "packages" / "scene_001.json", package)
    write_json(output_dir / "analysis" / "scene_001.json", scene_analysis)

    client = FakeVisionClient()
    scenes = analyze_scene_packages(output_dir, client=client)

    assert scenes.scene_count == 1
    assert client.calls == []

