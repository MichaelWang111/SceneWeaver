from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from sceneweaver.input.bilibili import extract_bvid, is_bilibili_url
from sceneweaver.input.bilibili_auth import bilibili_cookie_header
from sceneweaver.input.downloader import (
    BILIBILI_USER_AGENT,
    auth_source_label,
    build_ytdlp_cmd,
    ytdlp_cookie_candidates,
)
from sceneweaver.split.subtitle_segmenter import SubtitleCue, parse_subtitle_file, write_srt

Runner = Callable[..., subprocess.CompletedProcess]
HttpGetter = Callable[[str, dict[str, str]], bytes]
LogFn = Callable[[str], None]

SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".json"}
DEFAULT_CHROME_PROFILE_DIRECTORY = "Default"
DEFAULT_CHROME_CDP_URL = "http://127.0.0.1:9222"
BROWSER_FETCH_HEADER_ALLOWLIST = {"accept"}


class SubtitleFetchError(RuntimeError):
    pass


class BrowserSubtitleFetchError(SubtitleFetchError):
    def __init__(self, message: str, attempts: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.attempts = attempts


def download_bilibili_subtitles(
    url: str,
    output_dir: Path,
    runner: Runner = subprocess.run,
    force: bool = False,
    http_get: HttpGetter | None = None,
    browser_profile: bool = False,
    browser_cdp_url: str | None = None,
    browser_profile_timeout_seconds: float = 30.0,
    browser_user_data_dir: Path | None = None,
    browser_profile_directory: str | None = None,
    log: LogFn | None = None,
) -> list[SubtitleCue]:
    """Download Bilibili AI subtitles and persist a canonical SRT."""
    if not is_bilibili_url(url):
        return []

    subtitles_dir = output_dir / "source" / "subtitles"
    subtitles_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = subtitles_dir / "subtitles.srt"
    metadata_path = subtitles_dir / "metadata.json"

    if canonical_path.exists() and not force:
        return parse_subtitle_file(canonical_path)

    if force:
        _cleanup_raw_subtitle_files(subtitles_dir)

    attempts = []
    cookie_header, cookie_source = bilibili_cookie_header()
    api_cookie_header = cookie_header
    api_auth_source = cookie_source or "anonymous"
    cues, api_metadata = _download_bilibili_api_subtitles(
        url,
        subtitles_dir,
        api_cookie_header,
        http_get=http_get or _http_get,
    )
    attempts.append(
        {
            "method": "bilibili_api",
            "auth_source": api_auth_source,
            "returncode": 0 if cues else 1,
            "stderr_tail": _tail_text(api_metadata.get("error", "")),
        }
    )
    if cues:
        write_srt(canonical_path, cues)
        _write_metadata(
            metadata_path,
            {
                "source_url": url,
                "method": "bilibili_api",
                "auth_source": api_auth_source,
                "cue_count": len(cues),
                "source_path": str(api_metadata.get("source_path", "")),
                "language": api_metadata.get("language", ""),
                "attempts": attempts,
            },
        )
        return cues

    if browser_profile:
        browser_cues, browser_metadata = _download_bilibili_browser_subtitles(
            url,
            subtitles_dir,
            cdp_url=browser_cdp_url,
            timeout_seconds=browser_profile_timeout_seconds,
            user_data_dir=browser_user_data_dir,
            profile_directory=browser_profile_directory,
            log=log,
        )
        browser_attempts = browser_metadata.get("attempts")
        if isinstance(browser_attempts, list):
            attempts.extend(browser_attempts)
        else:
            attempts.append(
                {
                    "method": browser_metadata.get("method", "chrome_cdp"),
                    "auth_source": browser_metadata.get("auth_source", "chrome_cdp"),
                    "returncode": 0 if browser_cues else 1,
                    "stderr_tail": _tail_text(browser_metadata.get("error", "")),
                }
            )
        if browser_cues:
            browser_method = str(browser_metadata.get("method") or "chrome_cdp")
            browser_auth_source = str(browser_metadata.get("auth_source") or browser_method)
            write_srt(canonical_path, browser_cues)
            _write_metadata(
                metadata_path,
                {
                    "source_url": url,
                    "method": browser_method,
                    "auth_source": browser_auth_source,
                    "cue_count": len(browser_cues),
                    "source_path": str(browser_metadata.get("source_path", "")),
                    "language": browser_metadata.get("language", ""),
                    "attempts": attempts,
                },
            )
            return browser_cues

    for cookie_source in ytdlp_cookie_candidates(url):
        before = _known_subtitle_files(subtitles_dir)
        cmd = build_ytdlp_cmd(
            url,
            [
                "--skip-download",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "all,-danmaku",
                "--sub-format",
                "srt/vtt/best",
                "--convert-subs",
                "srt",
                "-o",
                str(subtitles_dir / "raw.%(id)s.%(ext)s"),
            ],
            cookie_source=cookie_source,
        )
        result = runner(cmd, check=False, capture_output=True, text=True)
        attempts.append(_attempt_metadata(cookie_source, result))
        if result.returncode != 0:
            continue

        cues, source_path = _load_downloaded_cues(subtitles_dir, before)
        if not cues:
            continue

        write_srt(canonical_path, cues)
        _write_metadata(
            metadata_path,
            {
                "source_url": url,
                "method": "yt_dlp",
                "auth_source": auth_source_label(cookie_source),
                "cue_count": len(cues),
                "source_path": source_path.name,
                "attempts": attempts,
            },
        )
        return cues

    _write_metadata(
        metadata_path,
        {
            "source_url": url,
            "method": "unavailable",
            "auth_source": "unavailable",
            "cue_count": 0,
            "source_path": "",
            "note": "No Bilibili AI subtitles were available. Logged-in browser cookies may be required.",
            "attempts": attempts,
        },
    )
    return []


def _download_bilibili_api_subtitles(
    url: str,
    subtitles_dir: Path,
    cookie_header: str,
    *,
    http_get: HttpGetter,
) -> tuple[list[SubtitleCue], dict[str, object]]:
    try:
        return _download_bilibili_api_subtitles_or_raise(url, subtitles_dir, cookie_header, http_get=http_get)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, SubtitleFetchError) as exc:
        return [], {"error": str(exc)}


def _download_bilibili_browser_subtitles(
    url: str,
    subtitles_dir: Path,
    *,
    cdp_url: str | None,
    timeout_seconds: float,
    user_data_dir: Path | None,
    profile_directory: str | None,
    log: LogFn | None,
) -> tuple[list[SubtitleCue], dict[str, object]]:
    try:
        return _download_bilibili_browser_subtitles_or_raise(
            url,
            subtitles_dir,
            cdp_url=cdp_url,
            timeout_seconds=timeout_seconds,
            user_data_dir=user_data_dir,
            profile_directory=profile_directory,
            log=log,
        )
    except BrowserSubtitleFetchError as exc:
        return [], {"error": str(exc), "attempts": exc.attempts}
    except (OSError, json.JSONDecodeError, SubtitleFetchError, ImportError) as exc:
        return [], {"error": str(exc)}


def _download_bilibili_browser_subtitles_or_raise(
    url: str,
    subtitles_dir: Path,
    *,
    cdp_url: str | None,
    timeout_seconds: float,
    user_data_dir: Path | None,
    profile_directory: str | None,
    log: LogFn | None,
) -> tuple[list[SubtitleCue], dict[str, object]]:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError("Playwright is required for browser subtitle fetching") from exc

    attempts: list[dict[str, object]] = []
    with sync_playwright() as playwright:
        cdp_endpoint = _resolve_chrome_cdp_url(cdp_url)
        if cdp_endpoint:
            if not _chrome_cdp_available(cdp_endpoint):
                attempts.append(
                    _browser_attempt_metadata(
                        "chrome_cdp",
                        f"chrome_cdp:{cdp_endpoint}",
                        1,
                        f"Chrome CDP is not reachable: {cdp_endpoint}",
                    )
                )
            else:
                try:
                    cues, metadata = _download_bilibili_cdp_subtitles_or_raise(
                        playwright,
                        url,
                        subtitles_dir,
                        cdp_url=cdp_endpoint,
                        timeout_seconds=timeout_seconds,
                        log=log,
                    )
                    attempts.append(_browser_attempt_metadata("chrome_cdp", f"chrome_cdp:{cdp_endpoint}", 0, ""))
                    metadata.update(
                        {
                            "method": "chrome_cdp",
                            "auth_source": f"chrome_cdp:{cdp_endpoint}",
                            "attempts": attempts,
                        }
                    )
                    return cues, metadata
                except (OSError, json.JSONDecodeError, SubtitleFetchError, PlaywrightError) as exc:
                    attempts.append(_browser_attempt_metadata("chrome_cdp", f"chrome_cdp:{cdp_endpoint}", 1, str(exc)))

        try:
            cues, metadata = _download_bilibili_profile_subtitles_or_raise(
                playwright,
                url,
                subtitles_dir,
                timeout_seconds=timeout_seconds,
                user_data_dir=user_data_dir,
                profile_directory=profile_directory,
                log=log,
            )
            attempts.append(
                _browser_attempt_metadata(
                    "playwright_chrome_profile",
                    metadata.get("auth_source", "playwright_chrome_profile"),
                    0,
                    "",
                )
            )
            metadata.update(
                {
                    "method": "playwright_chrome_profile",
                    "auth_source": metadata.get("auth_source", "playwright_chrome_profile"),
                    "attempts": attempts,
                }
            )
            return cues, metadata
        except (OSError, json.JSONDecodeError, SubtitleFetchError, PlaywrightError) as exc:
            attempts.append(_browser_attempt_metadata("playwright_chrome_profile", "playwright_chrome_profile", 1, str(exc)))

    raise BrowserSubtitleFetchError("browser subtitle fetch failed", attempts)


def _download_bilibili_cdp_subtitles_or_raise(
    playwright,
    url: str,
    subtitles_dir: Path,
    *,
    cdp_url: str,
    timeout_seconds: float,
    log: LogFn | None,
) -> tuple[list[SubtitleCue], dict[str, object]]:
    if log:
        log(f"Connecting to Chrome CDP for Bilibili subtitles: {cdp_url}.")
    browser = playwright.chromium.connect_over_cdp(cdp_url, timeout=timeout_seconds * 1000)
    page = None
    try:
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        _assert_bilibili_login_page(page, timeout_seconds=timeout_seconds, log=log)

        def browser_http_get(request_url: str, headers: dict[str, str]) -> bytes:
            return _page_fetch_bytes(page, request_url, headers, timeout_seconds=timeout_seconds)

        return _download_bilibili_api_subtitles_or_raise(
            url,
            subtitles_dir,
            "",
            http_get=browser_http_get,
        )
    finally:
        if page is not None:
            page.close()
        browser.close()


def _download_bilibili_profile_subtitles_or_raise(
    playwright,
    url: str,
    subtitles_dir: Path,
    *,
    timeout_seconds: float,
    user_data_dir: Path | None,
    profile_directory: str | None,
    log: LogFn | None,
) -> tuple[list[SubtitleCue], dict[str, object]]:
    profile_dir = _resolve_chrome_user_data_dir(user_data_dir)
    profile_name = _resolve_chrome_profile_directory(profile_directory)
    if not profile_dir.exists():
        raise SubtitleFetchError(f"Chrome user data directory does not exist: {profile_dir}")
    if log:
        log(f"Reusing Chrome profile for Bilibili subtitles: {profile_dir} ({profile_name}).")
    context = _launch_persistent_browser(playwright, profile_dir, profile_name)
    try:
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        _assert_bilibili_login(context, timeout_seconds=timeout_seconds, log=log)

        def browser_http_get(request_url: str, headers: dict[str, str]) -> bytes:
            safe_headers = {key: value for key, value in headers.items() if key.lower() != "cookie"}
            response = context.request.get(request_url, headers=safe_headers, timeout=30_000)
            if not response.ok:
                raise SubtitleFetchError(f"browser request failed: {response.status} {request_url}")
            return response.body()

        cues, metadata = _download_bilibili_api_subtitles_or_raise(
            url,
            subtitles_dir,
            "",
            http_get=browser_http_get,
        )
        metadata["auth_source"] = f"playwright_chrome_profile:{profile_dir}:{profile_name}"
        return cues, metadata
    finally:
        context.close()


def _resolve_chrome_cdp_url(value: str | None) -> str:
    env_value = os.environ.get("SCENEWEAVER_CHROME_CDP_URL") or os.environ.get("SCENEWEAVER_BILIBILI_CHROME_CDP_URL")
    resolved = value or env_value or DEFAULT_CHROME_CDP_URL
    return resolved.strip()


def _chrome_cdp_available(cdp_url: str) -> bool:
    if cdp_url.startswith("ws://") or cdp_url.startswith("wss://"):
        return True
    try:
        request = urllib.request.Request(
            f"{cdp_url.rstrip('/')}/json/version",
            headers={"User-Agent": BILIBILI_USER_AGENT},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=1.5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _assert_bilibili_login_page(page, *, timeout_seconds: float, log: LogFn | None) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            payload = _page_fetch_json(
                page,
                "https://api.bilibili.com/x/web-interface/nav",
                {"Accept": "application/json, text/plain, */*"},
                timeout_seconds=10.0,
            )
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict) and data.get("isLogin"):
                if log:
                    log("Bilibili login state found in connected Chrome.")
                return
            last_error = "connected Chrome is not logged in to Bilibili"
        except (json.JSONDecodeError, SubtitleFetchError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise SubtitleFetchError(last_error or "Bilibili login state was not available in connected Chrome")


def _page_fetch_json(page, request_url: str, headers: dict[str, str], *, timeout_seconds: float) -> dict:
    return json.loads(_page_fetch_bytes(page, request_url, headers, timeout_seconds=timeout_seconds).decode("utf-8-sig"))


def _page_fetch_bytes(page, request_url: str, headers: dict[str, str], *, timeout_seconds: float) -> bytes:
    safe_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() in BROWSER_FETCH_HEADER_ALLOWLIST
    }
    payload = page.evaluate(
        """async ({ url, headers, timeoutMs }) => {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const response = await fetch(url, {
                    method: "GET",
                    credentials: "include",
                    headers,
                    signal: controller.signal,
                });
                const text = await response.text();
                return { ok: response.ok, status: response.status, text };
            } finally {
                clearTimeout(timeoutId);
            }
        }""",
        {
            "url": request_url,
            "headers": safe_headers,
            "timeoutMs": int(timeout_seconds * 1000),
        },
    )
    if not isinstance(payload, dict):
        raise SubtitleFetchError(f"browser fetch returned an invalid response for {request_url}")
    if not payload.get("ok"):
        raise SubtitleFetchError(f"browser fetch failed: HTTP {payload.get('status')} {request_url}")
    text = payload.get("text")
    if not isinstance(text, str):
        raise SubtitleFetchError(f"browser fetch returned no text for {request_url}")
    return text.encode("utf-8")


def _browser_attempt_metadata(method: str, auth_source: object, returncode: int, error: str) -> dict[str, object]:
    return {
        "method": method,
        "auth_source": str(auth_source or method),
        "returncode": returncode,
        "stderr_tail": _tail_text(error),
    }


def _resolve_chrome_user_data_dir(user_data_dir: Path | None) -> Path:
    if user_data_dir is not None:
        return user_data_dir.resolve()
    env_value = os.environ.get("SCENEWEAVER_CHROME_USER_DATA_DIR") or os.environ.get(
        "SCENEWEAVER_BILIBILI_CHROME_USER_DATA_DIR"
    )
    if env_value:
        return Path(env_value).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (Path(local_app_data) / "Google" / "Chrome" / "User Data").resolve()
    return (Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data").resolve()


def _resolve_chrome_profile_directory(profile_directory: str | None) -> str:
    return (
        profile_directory
        or os.environ.get("SCENEWEAVER_CHROME_PROFILE_DIRECTORY")
        or os.environ.get("SCENEWEAVER_BILIBILI_CHROME_PROFILE_DIRECTORY")
        or DEFAULT_CHROME_PROFILE_DIRECTORY
    )


def _launch_persistent_browser(playwright, profile_dir: Path, profile_name: str):
    return playwright.chromium.launch_persistent_context(
        str(profile_dir),
        channel="chrome",
        headless=False,
        args=[f"--profile-directory={profile_name}"],
    )


def _assert_bilibili_login(context, *, timeout_seconds: float, log: LogFn | None) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        response = context.request.get("https://api.bilibili.com/x/web-interface/nav", timeout=10_000)
        if response.ok:
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict) and data.get("isLogin"):
                if log:
                    log("Bilibili login state found in reused Chrome profile.")
                return
            last_error = "reused Chrome profile is not logged in to Bilibili"
        else:
            last_error = f"Bilibili nav check failed: HTTP {response.status}"
        time.sleep(1.0)
    raise SubtitleFetchError(last_error or "Bilibili login state was not available in reused Chrome profile")


def _download_bilibili_api_subtitles_or_raise(
    url: str,
    subtitles_dir: Path,
    cookie_header: str,
    *,
    http_get: HttpGetter,
) -> tuple[list[SubtitleCue], dict[str, object]]:
    bvid = extract_bvid(url)
    if not bvid:
        raise SubtitleFetchError("missing BVID")

    headers = _bilibili_headers(url, cookie_header)
    view_url = f"https://api.bilibili.com/x/web-interface/view?bvid={urllib.parse.quote(bvid)}"
    view = _fetch_json(view_url, headers, http_get=http_get)
    view_data = _api_data(view, "view")
    aid = str(view_data.get("aid") or "")
    cid = _first_cid(view_data)
    if not cid:
        raise SubtitleFetchError("missing cid from Bilibili view API")

    player_errors = []
    for player_url in _player_api_urls(bvid, aid, cid):
        try:
            player = _fetch_json(player_url, headers, http_get=http_get)
            player_data = _api_data(player, "player")
            subtitle_item = _preferred_subtitle_item(_subtitle_items(player_data))
            if not subtitle_item:
                player_errors.append("player API returned no subtitle items")
                continue

            subtitle_url = _normalize_subtitle_url(str(subtitle_item.get("subtitle_url") or ""))
            if not subtitle_url:
                player_errors.append("subtitle item did not include subtitle_url")
                continue

            subtitle_json = _fetch_json(subtitle_url, headers, http_get=http_get)
            language = str(subtitle_item.get("lan") or subtitle_item.get("lan_doc") or "subtitle")
            raw_path = subtitles_dir / f"raw.{bvid}.{_safe_subtitle_name(language)}.json"
            raw_path.write_text(json.dumps(subtitle_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            cues = parse_subtitle_file(raw_path)
            if cues:
                return cues, {"source_path": raw_path.name, "language": language}
            player_errors.append("subtitle JSON did not contain timed cues")
        except (OSError, urllib.error.URLError, json.JSONDecodeError, SubtitleFetchError) as exc:
            player_errors.append(str(exc))

    raise SubtitleFetchError("; ".join(player_errors) or "Bilibili player API returned no usable subtitles")


def _bilibili_headers(url: str, cookie_header: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cookie": cookie_header,
        "Origin": "https://www.bilibili.com",
        "Referer": url,
        "User-Agent": BILIBILI_USER_AGENT,
    }


def _fetch_json(url: str, headers: dict[str, str], *, http_get: HttpGetter) -> dict:
    data = http_get(url, headers)
    return json.loads(data.decode("utf-8-sig"))


def _http_get(url: str, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _api_data(payload: dict, label: str) -> dict:
    code = payload.get("code")
    if code not in (0, None):
        raise SubtitleFetchError(f"{label} API failed: code={code}, message={payload.get('message') or payload.get('msg')}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise SubtitleFetchError(f"{label} API returned no data")
    return data


def _first_cid(view_data: dict) -> str:
    pages = view_data.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict) and page.get("cid"):
                return str(page["cid"])
    return str(view_data.get("cid") or "")


def _player_api_urls(bvid: str, aid: str, cid: str) -> list[str]:
    query = {"bvid": bvid, "cid": cid}
    if aid:
        query["aid"] = aid
    encoded = urllib.parse.urlencode(query)
    return [
        f"https://api.bilibili.com/x/player/v2?{encoded}",
        f"https://api.bilibili.com/x/player/wbi/v2?{encoded}",
    ]


def _subtitle_items(player_data: dict) -> list[dict]:
    subtitle = player_data.get("subtitle")
    if not isinstance(subtitle, dict):
        return []
    for key in ("subtitles", "list"):
        items = subtitle.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _preferred_subtitle_item(items: list[dict]) -> dict | None:
    usable = [item for item in items if item.get("subtitle_url")]
    if not usable:
        return None
    preferred_languages = ("zh-CN", "zh-Hans", "zh", "ai-zh")
    for language in preferred_languages:
        for item in usable:
            values = {str(item.get("lan") or ""), str(item.get("lan_doc") or "")}
            if any(language.lower() in value.lower() for value in values):
                return item
    return usable[0]


def _normalize_subtitle_url(value: str) -> str:
    if value.startswith("//"):
        return f"https:{value}"
    if value.startswith("http://"):
        return "https://" + value.removeprefix("http://")
    return value


def _safe_subtitle_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_.") or "subtitle"


def _write_metadata(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _known_subtitle_files(subtitles_dir: Path) -> set[Path]:
    return {
        path.resolve()
        for path in subtitles_dir.glob("*")
        if path.is_file() and path.suffix.lower() in SUBTITLE_EXTENSIONS
    }


def _cleanup_raw_subtitle_files(subtitles_dir: Path) -> None:
    for path in subtitles_dir.glob("raw.*"):
        if path.is_file():
            path.unlink()


def _attempt_metadata(cookie_source: str | None, result: subprocess.CompletedProcess) -> dict[str, object]:
    return {
        "method": "yt_dlp",
        "auth_source": auth_source_label(cookie_source),
        "returncode": result.returncode,
        "stderr_tail": _tail_text(getattr(result, "stderr", "")),
    }


def _tail_text(value: object, limit: int = 800) -> str:
    text = value if isinstance(value, str) else ""
    text = text.strip()
    lowered = text.lower()
    if "user data directory is already in use" in lowered or "please specify a unique value for --user-data-dir" in lowered:
        return "Chrome profile is already in use. Close Chrome or use a Chrome instance launched for automation before --browser-profile-subtitles."
    if "--user-data-dir=" in lowered and "--remote-debugging-pipe" in lowered and "<process did exit: exitcode=0" in lowered:
        return "Chrome profile was opened by an existing Chrome session, so Playwright could not attach to it. Close Chrome first, or use a Chrome instance launched for automation."
    return text[-limit:] if len(text) > limit else text


def _load_downloaded_cues(subtitles_dir: Path, before: set[Path]) -> tuple[list[SubtitleCue], Path]:
    candidates = [
        path
        for path in subtitles_dir.glob("*")
        if path.is_file()
        and path.suffix.lower() in SUBTITLE_EXTENSIONS
        and path.name != "metadata.json"
        and path.resolve() not in before
    ]
    if not candidates:
        candidates = [
            path
            for path in subtitles_dir.glob("*")
            if path.is_file() and path.suffix.lower() in SUBTITLE_EXTENSIONS and path.name != "metadata.json"
        ]

    for path in sorted(candidates, key=_subtitle_preference):
        cues = parse_subtitle_file(path)
        if cues:
            return cues, path
    return [], Path("")


def _subtitle_preference(path: Path) -> tuple[int, str]:
    suffix_rank = {".srt": 0, ".vtt": 1, ".json": 2}.get(path.suffix.lower(), 9)
    return suffix_rank, path.name
