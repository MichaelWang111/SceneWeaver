from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sceneweaver.input.subtitles import download_bilibili_subtitles
from sceneweaver.input.subtitles import _page_fetch_bytes
from sceneweaver.split.frame_sampler import sample_scene_frames
from sceneweaver.split.scene_detector import SceneSpan
from sceneweaver.split.subtitle_segmenter import SubtitleCue, parse_subtitle_file


def test_parse_bilibili_subtitle_json(tmp_path):
    subtitle_path = tmp_path / "subtitle.json"
    subtitle_path.write_text(
        json.dumps(
            {
                "body": [
                    {"from": 0.5, "to": 2.0, "content": "欢迎加入我们"},
                    {"from": 2.0, "to": 3.5, "content": "一起工作"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cues = parse_subtitle_file(subtitle_path)

    assert cues == [
        SubtitleCue(start_seconds=0.5, end_seconds=2.0, text="欢迎加入我们"),
        SubtitleCue(start_seconds=2.0, end_seconds=3.5, text="一起工作"),
    ]


def test_download_bilibili_subtitles_writes_canonical_srt(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "none")

    def fake_runner(cmd, **kwargs):
        output_template = Path(cmd[cmd.index("-o") + 1])
        subtitle_path = output_template.parent / "raw.BV1vdZ6BJEcQ.zh-Hans.json"
        subtitle_path.write_text(
            json.dumps({"body": [{"from": 1, "to": 2.5, "content": "AI字幕"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    cues = download_bilibili_subtitles(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
        force=True,
    )

    srt_path = tmp_path / "source" / "subtitles" / "subtitles.srt"
    metadata_path = tmp_path / "source" / "subtitles" / "metadata.json"
    assert cues == [SubtitleCue(start_seconds=1.0, end_seconds=2.5, text="AI字幕")]
    assert srt_path.exists()
    assert "AI字幕" in srt_path.read_text(encoding="utf-8")
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["cue_count"] == 1


def test_download_bilibili_subtitles_uses_cookie_header_api(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENEWEAVER_BILIBILI_COOKIE", "SESSDATA=fake; bili_jct=fake")
    calls = []

    def fake_http_get(url, headers):
        calls.append((url, headers))
        assert headers["Cookie"] == "SESSDATA=fake; bili_jct=fake"
        if "x/web-interface/view" in url:
            return json.dumps({"code": 0, "data": {"aid": 123, "pages": [{"cid": 456}]}}).encode("utf-8")
        if "x/player/v2" in url:
            return json.dumps(
                {
                    "code": 0,
                    "data": {
                        "subtitle": {
                            "subtitles": [
                                {
                                    "lan": "zh-CN",
                                    "lan_doc": "中文（AI生成）",
                                    "subtitle_url": "//subtitle.example.com/a.json",
                                }
                            ]
                        }
                    },
                }
            ).encode("utf-8")
        if "subtitle.example.com" in url:
            return json.dumps({"body": [{"from": 0.5, "to": 2.0, "content": "AI字幕"}]}).encode("utf-8")
        raise AssertionError(url)

    def fake_runner(cmd, **kwargs):
        raise AssertionError("yt-dlp fallback should not run when API subtitles succeed")

    cues = download_bilibili_subtitles(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=fake_runner,
        force=True,
        http_get=fake_http_get,
    )

    srt_path = tmp_path / "source" / "subtitles" / "subtitles.srt"
    metadata_path = tmp_path / "source" / "subtitles" / "metadata.json"
    assert cues == [SubtitleCue(start_seconds=0.5, end_seconds=2.0, text="AI字幕")]
    assert "AI字幕" in srt_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "bilibili_api"
    assert metadata["auth_source"] == "env:SCENEWEAVER_BILIBILI_COOKIE"
    assert len(calls) == 3


def test_download_bilibili_subtitles_uses_browser_cdp_after_anonymous_api(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER", "none")

    def fake_http_get(url, headers):
        if "x/web-interface/view" in url:
            return json.dumps({"code": 0, "data": {"aid": 123, "pages": [{"cid": 456}]}}).encode("utf-8")
        if "x/player" in url:
            return json.dumps({"code": 0, "data": {"subtitle": {"subtitles": []}}}).encode("utf-8")
        raise AssertionError(url)

    def fake_browser_subtitles(url, subtitles_dir, **kwargs):
        raw_path = subtitles_dir / "raw.BV1vdZ6BJEcQ.zh-CN.json"
        raw_path.write_text(
            json.dumps({"body": [{"from": 3, "to": 4, "content": "鐧诲綍AI瀛楀箷"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        return [SubtitleCue(start_seconds=3.0, end_seconds=4.0, text="鐧诲綍AI瀛楀箷")], {
            "method": "chrome_cdp",
            "auth_source": "chrome_cdp:http://127.0.0.1:9223",
            "source_path": raw_path.name,
            "language": "zh-CN",
            "attempts": [
                {
                    "method": "chrome_cdp",
                    "auth_source": "chrome_cdp:http://127.0.0.1:9223",
                    "returncode": 0,
                    "stderr_tail": "",
                }
            ],
        }

    monkeypatch.setattr("sceneweaver.input.subtitles._download_bilibili_browser_subtitles", fake_browser_subtitles)

    cues = download_bilibili_subtitles(
        "https://www.bilibili.com/video/BV1vdZ6BJEcQ/",
        tmp_path,
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("yt-dlp should not run")),
        force=True,
        http_get=fake_http_get,
        browser_profile=True,
        browser_cdp_url="http://127.0.0.1:9223",
    )

    metadata = json.loads((tmp_path / "source" / "subtitles" / "metadata.json").read_text(encoding="utf-8"))
    assert cues == [SubtitleCue(start_seconds=3.0, end_seconds=4.0, text="鐧诲綍AI瀛楀箷")]
    assert metadata["method"] == "chrome_cdp"
    assert metadata["auth_source"] == "chrome_cdp:http://127.0.0.1:9223"
    assert [attempt["method"] for attempt in metadata["attempts"]] == ["bilibili_api", "chrome_cdp"]


def test_page_fetch_bytes_does_not_forward_cookie_header():
    seen = {}

    class FakePage:
        def evaluate(self, script, payload):
            seen.update(payload["headers"])
            return {"ok": True, "status": 200, "text": "{}"}

    assert _page_fetch_bytes(
        FakePage(),
        "https://api.bilibili.com/x/web-interface/nav",
        {"Accept": "application/json", "Cookie": "SESSDATA=secret", "Referer": "https://example.com"},
        timeout_seconds=1.0,
    ) == b"{}"
    assert seen == {"Accept": "application/json"}


def test_frame_sampler_burns_subtitles_into_sampled_frames(tmp_path):
    from PIL import Image

    def fake_runner(cmd, **kwargs):
        Image.new("RGB", (640, 360), "white").save(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0)

    scenes = [SceneSpan("scene_001", 1, 0.0, 4.0)]
    cues = [SubtitleCue(start_seconds=0.0, end_seconds=4.0, text="同步字幕")]

    frames = sample_scene_frames(
        tmp_path / "video.mp4",
        scenes,
        tmp_path / "frames",
        runner=fake_runner,
        subtitle_cues=cues,
        burn_subtitles=True,
    )

    frame_path = tmp_path / frames["scene_001"]["middle"]
    assert frame_path.exists()
    assert frame_path.with_suffix(frame_path.suffix + ".subtitle").exists()


def test_frame_sampler_does_not_burn_subtitles_by_default(tmp_path):
    from PIL import Image

    def fake_runner(cmd, **kwargs):
        Image.new("RGB", (640, 360), "white").save(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0)

    scenes = [SceneSpan("scene_001", 1, 0.0, 4.0)]
    cues = [SubtitleCue(start_seconds=0.0, end_seconds=4.0, text="同步字幕")]

    frames = sample_scene_frames(
        tmp_path / "video.mp4",
        scenes,
        tmp_path / "frames",
        runner=fake_runner,
        subtitle_cues=cues,
    )

    frame_path = tmp_path / frames["scene_001"]["middle"]
    assert frame_path.exists()
    assert not frame_path.with_suffix(frame_path.suffix + ".subtitle").exists()
