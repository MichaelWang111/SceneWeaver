from __future__ import annotations

import subprocess
from pathlib import Path

from typer.testing import CliRunner

from sceneweaver.analysis.scene_package_builder import build_scene_packages, write_scene_packages
from sceneweaver.cli import app
from sceneweaver.input.bilibili import extract_bvid, extract_video_id
from sceneweaver.input.downloader import download_video
from sceneweaver.schemas import ScenePackage
from sceneweaver.split.frame_sampler import sample_scene_frames
from sceneweaver.split.scene_detector import SceneSpan
from sceneweaver.split.scene_detector import _probe_duration_seconds
from sceneweaver.split.subtitle_segmenter import parse_srt, segment_subtitles_for_scenes
from sceneweaver.storage.json_store import read_json


def test_extract_bilibili_ids():
    url = "https://www.bilibili.com/video/BV1vdZ6BJEcQ/?spm_id_from=333"
    assert extract_bvid(url) == "BV1vdZ6BJEcQ"
    assert extract_video_id(url) == "bilibili_BV1vdZ6BJEcQ"


def test_download_video_uses_reference_ytdlp_pattern(tmp_path):
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "--no-download" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="标题\n作者\n频道\n12\n99\n")
        return subprocess.CompletedProcess(cmd, 0)

    asset = download_video(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
    )

    assert asset.video_id == "bilibili_BV1vdZ6BJEcQ"
    assert asset.metadata_path.exists()
    download_cmd = calls[1][0]
    assert download_cmd[1:4] == ["-m", "yt_dlp", "-f"]
    assert "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" in download_cmd
    assert str(tmp_path / "source" / "video.mp4") in download_cmd


def test_subtitle_segmenter_matches_overlapping_cues(tmp_path):
    srt = tmp_path / "sample.srt"
    srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,500\n"
        "第一句\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:06,000\n"
        "第二句\n",
        encoding="utf-8",
    )
    scenes = [
        SceneSpan("scene_001", 1, 0.0, 3.0),
        SceneSpan("scene_002", 2, 3.0, 7.0),
    ]

    cues = parse_srt(srt)
    segments = segment_subtitles_for_scenes(scenes, cues)

    assert segments["scene_001"].text == "第一句"
    assert segments["scene_001"].items[0].start == "00:00:01.000"
    assert segments["scene_002"].text == "第二句"


def test_frame_sampler_extracts_three_named_frames(tmp_path):
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_text("fake jpg", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    scenes = [SceneSpan("scene_001", 1, 0.0, 4.0)]
    frames = sample_scene_frames(tmp_path / "video.mp4", scenes, tmp_path / "frames", runner=fake_runner)

    assert set(frames["scene_001"]) == {"start", "middle", "end"}
    assert frames["scene_001"]["middle"] == "frames/scene_001_middle.jpg"
    assert len(calls) == 3
    assert calls[0][0] == "ffmpeg"


def test_probe_duration_seconds_uses_ffprobe(tmp_path):
    def fake_runner(cmd, **kwargs):
        assert cmd[0] == "ffprobe"
        return subprocess.CompletedProcess(cmd, 0, stdout="1121.500590\n")

    assert _probe_duration_seconds(tmp_path / "video.mp4", runner=fake_runner) == 1121.500590


def test_build_and_write_scene_packages(tmp_path):
    scenes = [SceneSpan("scene_001", 1, 1.0, 3.5)]
    frame_paths = {
        "scene_001": {
            "start": "frames/scene_001_start.jpg",
            "middle": "frames/scene_001_middle.jpg",
            "end": "frames/scene_001_end.jpg",
        }
    }
    subtitle_segments = segment_subtitles_for_scenes(scenes, [])

    packages = build_scene_packages(
        video_id="bilibili_BV1vdZ6BJEcQ",
        source_url="https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        scenes=scenes,
        frame_paths=frame_paths,
        subtitle_segments=subtitle_segments,
    )
    write_scene_packages(tmp_path, packages)

    package = read_json(tmp_path / "packages" / "scene_001.json", ScenePackage)
    assert package.time_range.start == "00:00:01.000"
    assert package.time_range.duration_seconds == 2.5
    assert (tmp_path / "packages" / "scene_packages.json").exists()


def test_package_video_cli_invokes_pipeline(monkeypatch, tmp_path):
    called = {}

    def fake_run_package_video(**kwargs):
        called.update(kwargs)
        return tmp_path

    monkeypatch.setattr("sceneweaver.cli.run_package_video", fake_run_package_video)
    result = CliRunner().invoke(
        app,
        [
            "package-video",
            "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
            "--output",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert called["url"].endswith("BV1vdZ6BJEcQ/")
    assert called["output_dir"] == tmp_path
    assert called["split_video"] is False
