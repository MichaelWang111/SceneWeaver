from __future__ import annotations

import re

BVID_RE = re.compile(r"(BV[a-zA-Z0-9]{10})")


def extract_bvid(value: str) -> str:
    match = BVID_RE.search(value)
    return match.group(1) if match else ""


def is_bilibili_url(value: str) -> bool:
    return "bilibili.com" in value or "b23.tv" in value or bool(extract_bvid(value))


def extract_video_id(value: str) -> str:
    bvid = extract_bvid(value)
    if bvid:
        return f"bilibili_{bvid}"
    return "bilibili_unknown"

