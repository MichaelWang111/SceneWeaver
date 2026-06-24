from __future__ import annotations

import json
from argparse import Namespace

from sceneweaver.pipeline.mock_pipeline import run_mock_pipeline
from sceneweaver.user_api import (
    USER_INGEST_SCENE_ANALYSIS_MODEL,
    build_scene_analysis_client,
    generate_script,
    search_scenes,
)
from sceneweaver.user_cli import run_ingest, main


class FakeScriptClient:
    def __init__(self):
        self.calls = []

    def analyze_images_json(self, **kwargs):
        raise AssertionError("script generation must not send images to the LLM")

    def analyze_text_json(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "title": "成长的真实一刻",
            "logline": "用真实工作细节写一次团队成长。",
            "creative_strategy": "借鉴参考中的真实办公质感和团队归属情绪，但生成新的脚本画面。",
            "script_markdown": "## Beat 1｜进入真实问题\nVO: 我们在真实的问题里相遇，也在一次次并肩里看见自己的成长。\nVisual: 清晨团队进入项目室，电脑屏幕亮起，白板上留着昨晚讨论的方案。\nShot: 手持跟拍进入空间，用中近景捕捉互相补位的眼神和动作。\n\n## Beat 2｜协作推进\nVO: 不是每一步都完美，但每一次讨论都让答案更清楚。\nVisual: 成员围绕问题拆解方案，有人补充数据，有人修改原型，有人把复杂需求讲给新人听。\nShot: 交叉剪辑键盘、草图、会议中的短句，节奏逐渐升温。\n\n## Beat 3｜成长落点\nVO: 在这里，梦想不是一句口号，而是每天被共同推进的具体事情。\nVisual: 团队完成展示后相视一笑，镜头落在新成员继续写下下一步计划。\nShot: 稳定推近，保留真实环境声，收在温和但有力量的团队合影感。",
            "beats": [
                {
                    "beat_id": "beat_001",
                    "purpose": "建立真实感",
                    "voiceover": "我们在真实的问题里相遇。",
                    "visual_direction": "清晨团队进入项目室。",
                    "shot_notes": "中近景跟随人物动作。",
                    "inspired_by_reference_ids": ["reference_001"],
                }
            ],
            "reference_takeaways": [
                {"reference_id": "reference_001", "takeaway": "用真实工作状态建立信任", "used_as": "emotion"}
            ],
            "risks": ["避免空泛口号"],
        }


def test_search_scenes_returns_frame_context_from_sceneweaver_package(tmp_path):
    output_dir = run_mock_pipeline(tmp_path / "mock_video")

    result = search_scenes("technology capability interface", [output_dir], top_k=1)

    assert result["matches"]
    match = result["matches"][0]
    assert match["scene_id"] == "scene_001"
    assert match["time_range"]["start"] == "00:00:03.200"
    assert match["frames"]["middle"]["relative_path"] == "frames/scene_001_middle.jpg"
    assert match["source"]["package_path"].endswith("scene_001.json")


def test_generate_script_uses_retrieved_reference_frames(tmp_path):
    output_dir = run_mock_pipeline(tmp_path / "mock_video")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "scene_001_middle.jpg").write_bytes(b"fake image")
    client = FakeScriptClient()

    result = generate_script(
        "technology capability interface",
        [output_dir],
        script_brief="招聘短片",
        top_k=1,
        client=client,
    )

    assert result["script"]["beats"][0]["inspired_by_reference_ids"] == ["reference_001"]
    assert result["reference_items"][0]["scene_id"] == "scene_001"
    assert result["reference_items"][0]["frame_label"] == "middle"
    assert "creative_reference_soft_constraint" in client.calls[0]["user_prompt"]
    assert "scene_card_text_only_no_images" in client.calls[0]["user_prompt"]
    assert "image_paths" not in client.calls[0]
    assert "frame_path" not in client.calls[0]["user_prompt"]
    assert "scene_001_middle.jpg" not in client.calls[0]["user_prompt"]
    assert client.calls[0]["max_tokens"] == 6000


def test_generate_script_repairs_empty_script_output(tmp_path):
    output_dir = run_mock_pipeline(tmp_path / "mock_video")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "scene_001_middle.jpg").write_bytes(b"fake image")

    class RepairingClient(FakeScriptClient):
        def analyze_text_json(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {"title": "空结果"}
            return {
                "title": "成长的真实一刻",
                "logline": "用真实工作细节写一次团队成长。",
                "creative_strategy": "把参考 scene 的真实工作压力转化为招聘片中的团队协作和成长弧线。",
                "script_markdown": "## Beat 1\nVO: 我们在真实的问题里相遇。\nVisual: 清晨团队进入项目室，白板上留下昨日讨论的痕迹。\nShot: 手持跟拍，进入人物关系。\n\n## Beat 2\nVO: 每一次并肩，都是向梦想靠近的一步。\nVisual: 成员围绕问题拆解方案，互相补位，最终完成展示。\nShot: 中近景交替，节奏逐步升温。",
                "beats": [
                    {
                        "beat_id": "beat_001",
                        "purpose": "建立真实团队感",
                        "voiceover": "我们在真实的问题里相遇。",
                        "visual_direction": "清晨团队进入项目室，白板上留下昨日讨论的痕迹。",
                        "shot_notes": "手持跟拍，进入人物关系。",
                        "inspired_by_reference_ids": ["reference_001"],
                    }
                ],
                "reference_takeaways": [
                    {"reference_id": "reference_001", "takeaway": "用真实工作状态建立信任", "used_as": "emotion"}
                ],
                "risks": ["避免空泛口号"],
            }

    client = RepairingClient()

    result = generate_script("technology capability interface", [output_dir], top_k=1, client=client)

    assert len(client.calls) == 2
    assert "Repair the previous response" in client.calls[1]["user_prompt"]
    assert result["script"]["script_markdown"].startswith("## Beat 1")


def test_user_cli_search_prints_json(capsys, tmp_path):
    output_dir = run_mock_pipeline(tmp_path / "mock_video")

    exit_code = main(["search", "technology capability interface", "--source", str(output_dir), "--top-k", "1"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["matches"][0]["scene_id"] == "scene_001"


def test_user_cli_script_prints_json(monkeypatch, capsys, tmp_path):
    output_dir = run_mock_pipeline(tmp_path / "mock_video")

    monkeypatch.setattr("sceneweaver.user_cli.generate_script", lambda *args, **kwargs: {"status": "ok", "script": {"script_markdown": "draft"}})

    exit_code = main([
        "script",
        "technology capability interface",
        "--source",
        str(output_dir),
        "--top-k",
        "1",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["script"]["script_markdown"] == "draft"


def test_run_ingest_no_analyze_does_not_extract_cards(monkeypatch, tmp_path):
    called = {}

    def fake_ingest_video(*args, **kwargs):
        called.update(kwargs)
        return {"status": "packaged"}

    monkeypatch.setattr("sceneweaver.user_cli.ingest_video", fake_ingest_video)
    args = Namespace(
        source="video.mp4",
        source_type="file",
        output_root=tmp_path,
        output_dir=None,
        video_id="video_001",
        scene_threshold=27.0,
        subtitle=None,
        split_video=False,
        force=False,
        frame_workers=None,
        burn_subtitles=False,
        no_analyze=True,
        no_extract_cards=False,
        limit=None,
        concurrency=1,
        timeout_seconds=180.0,
        retries=0,
        scene_analysis_model=USER_INGEST_SCENE_ANALYSIS_MODEL,
        quiet=True,
    )

    result = run_ingest(args)

    assert result == {"status": "packaged"}
    assert called["analyze"] is False
    assert called["extract_cards"] is False
    assert called["scene_analysis_model"] == "qwen3.7-plus"


def test_user_ingest_scene_analysis_client_forces_qwen_plus_on_dashscope(monkeypatch, tmp_path):
    monkeypatch.setattr("sceneweaver.llm.settings.DEFAULT_LLM_CONFIG_PATH", tmp_path / "llm_config.json")
    monkeypatch.setenv("SCENEWEAVER_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")

    client = build_scene_analysis_client()

    assert client.config.provider == "dashscope"
    assert client.config.api_key == "dashscope-key"
    assert client.config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert client.config.model == "qwen3.7-plus"
