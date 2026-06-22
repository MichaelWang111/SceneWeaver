from __future__ import annotations

import subprocess
import json
from pathlib import Path

from typer.testing import CliRunner

from sceneweaver.analysis.scene_package_builder import build_scene_packages, write_scene_packages
from sceneweaver.cli import app
from sceneweaver.input.bilibili import extract_bvid, extract_video_id
from sceneweaver.input.downloader import (
    BILIBILI_FORMAT,
    download_video,
    ytdlp_browser_cookie_sources,
    ytdlp_extra_args,
    _select_bilibili_streams,
)
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


def test_download_video_uses_browser_cookies_for_bilibili(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "chrome")
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_BROWSER_DOWNLOAD", "off")

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
    assert download_cmd[1:3] == ["-m", "yt_dlp"]
    assert download_cmd[download_cmd.index("--cookies-from-browser") + 1] == "chrome"
    assert "--user-agent" in download_cmd
    assert "Referer:https://www.bilibili.com/video/BV1vdZ6BJEcQ/" in download_cmd
    assert "Origin:https://www.bilibili.com" in download_cmd
    assert download_cmd[download_cmd.index("-f") + 1] == BILIBILI_FORMAT
    assert str(tmp_path / "source" / "video.mp4") in download_cmd


def test_download_video_does_not_read_browser_cookie_db_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.delenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", raising=False)
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_BROWSER_DOWNLOAD", "off")

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "--no-download" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="title\nuploader\nchannel\n12\n99\n")
        return subprocess.CompletedProcess(cmd, 0)

    download_video(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
    )

    assert calls
    assert all("--cookies-from-browser" not in cmd for cmd, _kwargs in calls)


def test_download_video_accepts_extra_ytdlp_args_from_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("SCENEWEAVER_YTDLP_ARGS", "--proxy http://127.0.0.1:7890 --impersonate chrome")
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "none")
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_BROWSER_DOWNLOAD", "off")

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "--no-download" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="标题\n作者\n频道\n12\n99\n")
        return subprocess.CompletedProcess(cmd, 0)

    download_video(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
    )

    assert ytdlp_extra_args() == ["--proxy", "http://127.0.0.1:7890", "--impersonate", "chrome"]
    assert ytdlp_browser_cookie_sources() == []
    for cmd, _kwargs in calls:
        assert "--proxy" in cmd
        assert "http://127.0.0.1:7890" in cmd
        assert "--impersonate" in cmd
        assert "chrome" in cmd
        assert "--cookies-from-browser" not in cmd


def test_download_video_force_overwrites_existing_file(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "none")
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_BROWSER_DOWNLOAD", "off")
    video_path = tmp_path / "source" / "video.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_text("old", encoding="utf-8")

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "--no-download" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="title\nuploader\nchannel\n12\n99\n")
        return subprocess.CompletedProcess(cmd, 0)

    download_video(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
        force=True,
    )

    download_cmd = calls[1][0]
    assert "--force-overwrites" in download_cmd


def test_download_video_uses_explicit_bilibili_cookie_file(monkeypatch, tmp_path):
    calls = []
    cookies_paths = []
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_COOKIE", "SESSDATA=fake; bili_jct=fake")
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "none")
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_BROWSER_DOWNLOAD", "off")

    def fake_runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "--no-download" in cmd:
            cookies_path = Path(cmd[cmd.index("--cookies") + 1])
            cookies_paths.append(cookies_path)
            assert cookies_path.exists()
            return subprocess.CompletedProcess(cmd, 0, stdout="title\nuploader\nchannel\n12\n99\n")
        cookies_path = Path(cmd[cmd.index("--cookies") + 1])
        cookies_paths.append(cookies_path)
        assert cookies_path.exists()
        assert "SESSDATA" in cookies_path.read_text(encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    download_video(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
    )

    assert cookies_paths
    assert len(cookies_paths) == 2
    assert not any(path.exists() for path in cookies_paths)


def test_select_bilibili_streams_prefers_highest_quality():
    streams = _select_bilibili_streams(
        {
            "data": {
                "dash": {
                    "video": [
                        {"id": 64, "bandwidth": 2000, "baseUrl": "https://video.example/64.m4s"},
                        {"id": 80, "bandwidth": 1000, "baseUrl": "https://video.example/80.m4s"},
                    ],
                    "audio": [
                        {"id": 30216, "bandwidth": 90000, "baseUrl": "https://audio.example/low.m4s"},
                        {"id": 30280, "bandwidth": 192000, "backupUrl": ["https://audio.example/high.m4s"]},
                    ],
                }
            }
        }
    )

    assert streams["video_url"] == "https://video.example/80.m4s"
    assert streams["audio_url"] == "https://audio.example/high.m4s"
    assert streams["video_quality"] == "80"


def test_download_video_uses_chrome_cdp_when_available(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "none")
    monkeypatch.setattr("sceneweaver.input.downloader._chrome_cdp_available", lambda cdp_url: True)
    monkeypatch.setattr(
        "sceneweaver.input.downloader._bilibili_streams_from_cdp",
        lambda url, cdp_url, log=None: {
            "video_url": "https://video.example/high.m4s",
            "audio_url": "https://audio.example/high.m4s",
            "video_quality": "80",
            "audio_quality": "30280",
        },
    )
    monkeypatch.setattr(
        "sceneweaver.input.downloader._download_url",
        lambda url, path, referer, log=None: path.write_text("stream", encoding="utf-8"),
    )

    def fake_runner(cmd, **kwargs):
        if "--no-download" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="title\nuploader\nchannel\n12\n99\n")
        Path(cmd[-1]).write_text("merged", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    asset = download_video(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
        browser_cdp_url="http://127.0.0.1:9223",
    )

    metadata = json.loads(asset.metadata_path.read_text(encoding="utf-8"))
    assert asset.video_path.read_text(encoding="utf-8") == "merged"
    assert metadata["auth_source"] == "chrome_cdp:http://127.0.0.1:9223"
    assert metadata["download_method"] == "chrome_cdp_playinfo"
    assert metadata["video_quality"] == "80"


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


def test_frame_sampler_accepts_parallel_workers(tmp_path):
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_text("fake jpg", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    scenes = [
        SceneSpan("scene_001", 1, 0.0, 4.0),
        SceneSpan("scene_002", 2, 4.0, 8.0),
    ]
    frames = sample_scene_frames(
        tmp_path / "video.mp4",
        scenes,
        tmp_path / "frames",
        runner=fake_runner,
        frame_workers=2,
    )

    assert len(calls) == 6
    assert frames["scene_002"]["end"] == "frames/scene_002_end.jpg"


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
            "--frame-workers",
            "3",
            "--burn-subtitles",
            "--browser-profile-subtitles",
            "--browser-cdp-url",
            "http://127.0.0.1:9223",
            "--browser-profile-directory",
            "Default",
        ],
    )

    assert result.exit_code == 0
    assert called["url"].endswith("BV1vdZ6BJEcQ/")
    assert called["output_dir"] == tmp_path
    assert called["split_video"] is False
    assert called["frame_workers"] == 3
    assert called["burn_subtitles"] is True
    assert called["browser_profile_subtitles"] is True
    assert called["browser_cdp_url"] == "http://127.0.0.1:9223"
    assert called["browser_profile_directory"] == "Default"
