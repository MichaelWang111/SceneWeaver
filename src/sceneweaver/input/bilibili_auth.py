from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator

COOKIE_ENV_NAMES = ("SCENEWEAVER_BILIBILI_COOKIE", "BILIBILI_COOKIE")
SESSDATA_ENV_NAMES = ("SCENEWEAVER_BILIBILI_SESSDATA", "BILIBILI_SESSDATA")
BILI_JCT_ENV_NAMES = ("SCENEWEAVER_BILIBILI_BILI_JCT", "BILIBILI_BILI_JCT")
DEDE_USER_ID_ENV_NAMES = ("SCENEWEAVER_BILIBILI_DEDEUSERID", "BILIBILI_DEDEUSERID")


def bilibili_cookie_header(env: dict[str, str] | None = None) -> tuple[str, str]:
    env = env or os.environ
    for name in COOKIE_ENV_NAMES:
        value = (env.get(name) or "").strip()
        if value:
            return normalize_cookie_header(value), f"env:{name}"

    pairs = []
    sessdata, sessdata_name = _first_env_value(env, SESSDATA_ENV_NAMES)
    if sessdata:
        pairs.append(("SESSDATA", sessdata))
    bili_jct, _ = _first_env_value(env, BILI_JCT_ENV_NAMES)
    if bili_jct:
        pairs.append(("bili_jct", bili_jct))
    dede_user_id, _ = _first_env_value(env, DEDE_USER_ID_ENV_NAMES)
    if dede_user_id:
        pairs.append(("DedeUserID", dede_user_id))

    if not pairs:
        return "", ""
    return "; ".join(f"{key}={value}" for key, value in pairs), f"env:{sessdata_name or 'SCENEWEAVER_BILIBILI_SESSDATA'}"


def normalize_cookie_header(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().startswith("cookie:"):
        return stripped.split(":", 1)[1].strip()
    return stripped


@contextmanager
def temporary_bilibili_cookies_file(cookie_header: str) -> Iterator[Path]:
    cookies = parse_cookie_header(cookie_header)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".cookies.txt", delete=False)
    path = Path(handle.name)
    try:
        with handle:
            handle.write("# Netscape HTTP Cookie File\n")
            for name, value in cookies.items():
                handle.write(f".bilibili.com\tTRUE\t/\tTRUE\t2147483647\t{name}\t{value}\n")
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in normalize_cookie_header(cookie_header).split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or "\n" in name or "\r" in name or "\t" in name:
            continue
        if "\n" in value or "\r" in value or "\t" in value:
            continue
        cookies[name] = value
    return cookies


def _first_env_value(env: dict[str, str], names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value, name
    return "", ""
