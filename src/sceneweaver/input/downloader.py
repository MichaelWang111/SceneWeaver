from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sceneweaver.input.bilibili_auth import bilibili_cookie_header, temporary_bilibili_cookies_file
from sceneweaver.input.bilibili import extract_video_id, is_bilibili_url

Runner = Callable[..., subprocess.CompletedProcess]
LogFn = Callable[[str], None]

BILIBILI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
DEFAULT_CHROME_CDP_URL = "http://127.0.0.1:9222"
BILIBILI_BROWSER_DOWNLOAD_ENV = "SCENEWEAVER_BILIBILI_BROWSER_DOWNLOAD"
BILIBILI_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
BILIBILI_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
)
DEFAULT_BROWSER_COOKIE_SOURCES: tuple[str, ...] = ()


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    source_url: str
    video_path: Path
    metadata_path: Path
    title: str = ""
    uploader: str = ""


def get_video_info(url: str, runner: Runner = subprocess.run) -> dict:
    """Fetch lightweight metadata with yt-dlp, reusing browser cookies when available."""
    explicit_cookie_header, explicit_cookie_source = bilibili_cookie_header()
    if is_bilibili_url(url) and explicit_cookie_header:
        with temporary_bilibili_cookies_file(explicit_cookie_header) as cookies_path:
            cmd = build_ytdlp_cmd(url, _metadata_args(cookies_path=cookies_path))
            result = runner(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return _metadata_from_ytdlp_stdout(result.stdout, auth_source=explicit_cookie_source)

    for cookie_source in ytdlp_cookie_candidates(url):
        cmd = build_ytdlp_cmd(
            url,
            _metadata_args(),
            cookie_source=cookie_source,
        )
        result = runner(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            continue

        return _metadata_from_ytdlp_stdout(result.stdout, auth_source=auth_source_label(cookie_source))

    return {"success": False, "title": "", "uploader": "", "auth_source": "unavailable"}


def download_video(
    url: str,
    output_dir: Path,
    runner: Runner = subprocess.run,
    force: bool = False,
    browser_cdp_url: str | None = None,
    log: LogFn | None = None,
) -> VideoAsset:
    """Download one Bilibili video and persist basic metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_video_id(url)
    video_path = output_dir / "source" / "video.mp4"
    metadata_path = output_dir / "source" / "metadata.json"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    info = get_video_info(url, runner=runner)
    browser_download_error = ""
    metadata = {"video_id": video_id, "source_url": url, **info}
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if force or not video_path.exists():
        browser_downloaded = False
        if _should_try_browser_download(url):
            browser_metadata = try_download_bilibili_video_from_browser(
                url,
                video_path,
                runner=runner,
                force=force,
                cdp_url=browser_cdp_url,
                log=log,
            )
            if browser_metadata.get("success"):
                browser_downloaded = video_path.exists()
                metadata.update(browser_metadata)
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                browser_download_error = str(browser_metadata.get("error") or "")
        if not browser_downloaded:
            run_ytdlp_download(
                url=url,
                video_path=video_path,
                auth_source=metadata.get("auth_source", "unavailable"),
                force=force,
                runner=runner,
            )
            if browser_download_error:
                metadata["browser_download_error"] = browser_download_error
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    return VideoAsset(
        video_id=video_id,
        source_url=url,
        video_path=video_path,
        metadata_path=metadata_path,
        title=metadata.get("title", ""),
        uploader=metadata.get("uploader", ""),
    )


def _metadata_args(cookies_path: Path | None = None) -> list[str]:
    args = [
        "--print",
        "%(title)s",
        "--print",
        "%(uploader)s",
        "--print",
        "%(channel)s",
        "--print",
        "%(duration)s",
        "--print",
        "%(view_count)s",
        "--no-download",
    ]
    if cookies_path is not None:
        args[0:0] = ["--cookies", str(cookies_path)]
    return args


def _metadata_from_ytdlp_stdout(stdout: str, *, auth_source: str) -> dict[str, object]:
    lines = stdout.strip().splitlines()
    return {
        "success": True,
        "title": lines[0] if len(lines) > 0 else "",
        "uploader": lines[1] if len(lines) > 1 else "",
        "channel": lines[2] if len(lines) > 2 else "",
        "duration": lines[3] if len(lines) > 3 else "",
        "view_count": lines[4] if len(lines) > 4 else "",
        "auth_source": auth_source,
    }


def try_download_bilibili_video_from_browser(
    url: str,
    video_path: Path,
    *,
    runner: Runner = subprocess.run,
    force: bool = False,
    cdp_url: str | None = None,
    log: LogFn | None = None,
) -> dict[str, object]:
    cdp_endpoint = _resolve_chrome_cdp_url(cdp_url)
    if not _chrome_cdp_available(cdp_endpoint):
        return {"success": False, "error": f"Chrome CDP is not reachable: {cdp_endpoint}"}
    try:
        streams = _bilibili_streams_from_cdp(url, cdp_endpoint, log=log)
        _download_bilibili_streams(streams, video_path, source_url=url, runner=runner, force=force, log=log)
        return {
            "success": True,
            "auth_source": f"chrome_cdp:{cdp_endpoint}",
            "download_method": "chrome_cdp_playinfo",
            "video_quality": streams.get("video_quality", ""),
            "audio_quality": streams.get("audio_quality", ""),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _should_try_browser_download(url: str) -> bool:
    if not is_bilibili_url(url):
        return False
    value = os.environ.get(BILIBILI_BROWSER_DOWNLOAD_ENV, "auto").strip().lower()
    return value not in {"", "0", "false", "no", "none", "off"}


def _resolve_chrome_cdp_url(value: str | None) -> str:
    env_value = os.environ.get("SCENEWEAVER_CHROME_CDP_URL") or os.environ.get("SCENEWEAVER_BILIBILI_CHROME_CDP_URL")
    return (value or env_value or DEFAULT_CHROME_CDP_URL).strip()


def _chrome_cdp_available(cdp_url: str) -> bool:
    try:
        request = urllib.request.Request(f"{cdp_url.rstrip('/')}/json/version", headers={"User-Agent": BILIBILI_USER_AGENT})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=1.5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _bilibili_streams_from_cdp(url: str, cdp_url: str, *, log: LogFn | None = None) -> dict[str, object]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required for Chrome CDP Bilibili downloads") from exc

    if log:
        log(f"Connecting to Chrome CDP for Bilibili video streams: {cdp_url}.")
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url, timeout=10_000)
        page = None
        try:
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_function(
                "() => window.__playinfo__ || Array.from(document.scripts).some((script) => (script.textContent || '').includes('__playinfo__'))",
                timeout=15_000,
            )
            playinfo = page.evaluate(
                r"""() => {
                    if (window.__playinfo__) return window.__playinfo__;
                    for (const script of document.scripts) {
                        const text = script.textContent || '';
                        const marker = 'window.__playinfo__=';
                        const index = text.indexOf(marker);
                        if (index >= 0) {
                            const raw = text.slice(index + marker.length).replace(/;\s*$/, '');
                            return JSON.parse(raw);
                        }
                    }
                    return null;
                }"""
            )
        finally:
            if page is not None:
                page.close()
            browser.close()

    return _select_bilibili_streams(playinfo)


def _select_bilibili_streams(playinfo: object) -> dict[str, object]:
    if not isinstance(playinfo, dict):
        raise RuntimeError("Bilibili playinfo was not available from the browser page")
    data = playinfo.get("data") if isinstance(playinfo.get("data"), dict) else playinfo
    dash = data.get("dash") if isinstance(data, dict) else None
    if not isinstance(dash, dict):
        raise RuntimeError("Bilibili playinfo did not include DASH streams")
    videos = [item for item in dash.get("video", []) if isinstance(item, dict)]
    audios = [item for item in dash.get("audio", []) if isinstance(item, dict)]
    if not videos or not audios:
        raise RuntimeError("Bilibili playinfo did not include both video and audio streams")
    video = max(videos, key=_stream_rank)
    audio = max(audios, key=_stream_rank)
    video_url = _stream_url(video)
    audio_url = _stream_url(audio)
    if not video_url or not audio_url:
        raise RuntimeError("Bilibili playinfo streams did not include usable URLs")
    return {
        "video_url": video_url,
        "audio_url": audio_url,
        "video_quality": str(video.get("id") or video.get("bandwidth") or ""),
        "audio_quality": str(audio.get("id") or audio.get("bandwidth") or ""),
    }


def _stream_rank(item: dict[str, object]) -> tuple[int, int]:
    return int(item.get("id") or 0), int(item.get("bandwidth") or 0)


def _stream_url(item: dict[str, object]) -> str:
    for key in ("baseUrl", "base_url"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    backups = item.get("backupUrl") or item.get("backup_url")
    if isinstance(backups, list):
        for value in backups:
            if isinstance(value, str) and value:
                return value
    return ""


def _download_bilibili_streams(
    streams: dict[str, object],
    video_path: Path,
    *,
    source_url: str,
    runner: Runner,
    force: bool,
    log: LogFn | None,
) -> None:
    tmp_dir = video_path.parent / ".browser_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    video_part = tmp_dir / "video.m4s"
    audio_part = tmp_dir / "audio.m4s"
    if force or not video_part.exists():
        _download_url(str(streams["video_url"]), video_part, referer=source_url, log=log)
    if force or not audio_part.exists():
        _download_url(str(streams["audio_url"]), audio_part, referer=source_url, log=log)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_part),
        "-i",
        str(audio_part),
        "-c",
        "copy",
        str(video_path),
    ]
    runner(cmd, check=True)


def _download_url(url: str, path: Path, *, referer: str, log: LogFn | None) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": BILIBILI_USER_AGENT,
            "Referer": referer,
            "Origin": "https://www.bilibili.com",
        },
    )
    if log:
        parsed = urllib.parse.urlparse(url)
        log(f"Downloading Bilibili stream from {parsed.netloc} to {path.name}...")
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
        while True:
            chunk = response.read(BILIBILI_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)


def ytdlp_extra_args(env: dict[str, str] | None = None) -> list[str]:
    raw = (env or os.environ).get("SCENEWEAVER_YTDLP_ARGS", "")
    return shlex.split(raw) if raw.strip() else []


def ytdlp_browser_cookie_sources(env: dict[str, str] | None = None) -> list[str]:
    raw = (env or os.environ).get("SCENEWEAVER_YTDLP_COOKIES_FROM_BROWSER")
    if raw is None:
        return list(DEFAULT_BROWSER_COOKIE_SOURCES)

    normalized = raw.strip().lower()
    if normalized in {"", "0", "false", "no", "none", "off"}:
        return []
    if normalized == "auto":
        return ["chrome", "edge", "firefox", "brave", "chromium"]

    return [source.strip() for source in raw.split(",") if source.strip()]


def ytdlp_cookie_candidates(url: str) -> list[str | None]:
    if not is_bilibili_url(url):
        return [None]
    return [*ytdlp_browser_cookie_sources(), None]


def build_ytdlp_cmd(
    url: str,
    args: list[str],
    cookie_source: str | None = None,
) -> list[str]:
    cmd = [sys.executable, "-m", "yt_dlp"]
    if is_bilibili_url(url):
        cmd.extend(
            [
                "--user-agent",
                BILIBILI_USER_AGENT,
                "--add-header",
                f"Referer:{url}",
                "--add-header",
                "Origin:https://www.bilibili.com",
            ]
        )
    if cookie_source:
        cmd.extend(["--cookies-from-browser", cookie_source])
    cmd.extend(ytdlp_extra_args())
    cmd.extend(args)
    cmd.append(url)
    return cmd


def run_ytdlp_download(
    url: str,
    video_path: Path,
    auth_source: str,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> None:
    last_result: subprocess.CompletedProcess | None = None
    explicit_cookie_header, _ = bilibili_cookie_header()
    if is_bilibili_url(url) and explicit_cookie_header:
        with temporary_bilibili_cookies_file(explicit_cookie_header) as cookies_path:
            cmd = build_ytdlp_cmd(
                url,
                _download_args(video_path, force=force, cookies_path=cookies_path),
            )
            result = runner(cmd, check=False)
            if result.returncode == 0:
                return
            last_result = result
    for cookie_source in download_cookie_candidates(url, auth_source):
        cmd = build_ytdlp_cmd(
            url,
            _download_args(video_path, force=force),
            cookie_source=cookie_source,
        )
        result = runner(cmd, check=False)
        if result.returncode == 0:
            return
        last_result = result

    if last_result is None:
        return
    raise subprocess.CalledProcessError(last_result.returncode, last_result.args)


def _download_args(video_path: Path, *, force: bool = False, cookies_path: Path | None = None) -> list[str]:
    args = ["-f", BILIBILI_FORMAT, "-o", str(video_path)]
    if force:
        args.insert(0, "--force-overwrites")
    if cookies_path is not None:
        args[0:0] = ["--cookies", str(cookies_path)]
    return args


def download_cookie_candidates(url: str, auth_source: str) -> list[str | None]:
    if not is_bilibili_url(url):
        return [None]
    if auth_source.startswith("browser:"):
        return [auth_source.removeprefix("browser:"), None]
    if auth_source == "anonymous":
        return [None]
    return ytdlp_cookie_candidates(url)


def auth_source_label(cookie_source: str | None) -> str:
    return f"browser:{cookie_source}" if cookie_source else "anonymous"
